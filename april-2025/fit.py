import copy

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel,
                                       SelectKBest, VarianceThreshold, chi2,
                                       f_classif, f_regression,
                                       mutual_info_regression)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   RobustScaler, StandardScaler, TargetEncoder)
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from tqdm.auto import tqdm
from xgboost import XGBRegressor


def oof_target_encode(
    train_df,
    test_df,
    features,
    target_col,
    smooth=0.0,
    n_splits=5,
    seed=42,
    ordinal=False,
    prefix="TE_",
    batch_size=5,
):
    """Leakage-free (OOF) target encoding, now with optional batching."""
    train_out = train_df.copy()
    test_out = test_df.copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def batch(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    feature_batches = list(batch(features, batch_size or len(features)))

    for feature_batch in tqdm(feature_batches, desc="Encoding feature batches"):
        te_cols = [f"{prefix}{feature}" for feature in feature_batch]
        oof_encoded_batch = pd.DataFrame(index=train_df.index, columns=te_cols, dtype=float)

        # OOF encoding per batch
        for tr_idx, val_idx in kf.split(train_df):
            tr_part, val_part = train_df.iloc[tr_idx], train_df.iloc[val_idx]

            enc = TargetEncoder(smooth=smooth, random_state=seed)
            enc.fit(tr_part[feature_batch], tr_part[target_col])

            val_encoded = enc.transform(val_part[feature_batch])
            if not isinstance(val_encoded, pd.DataFrame):
                val_encoded = pd.DataFrame(val_encoded, index=val_part.index, columns=feature_batch)

            val_encoded.columns = te_cols
            val_encoded.index = val_part.index

            oof_encoded_batch.iloc[val_idx] = val_encoded.values

        if ordinal:
            oof_encoded_batch = oof_encoded_batch.rank(method="dense").astype(int)

        train_out[te_cols] = oof_encoded_batch

        # Full encoding on test set
        final_enc = TargetEncoder(smooth=smooth, random_state=seed)
        final_enc.fit(train_df[feature_batch], train_df[target_col])

        test_encoded = final_enc.transform(test_df[feature_batch])
        if not isinstance(test_encoded, pd.DataFrame):
            test_encoded = pd.DataFrame(test_encoded, index=test_df.index, columns=feature_batch)

        test_encoded.columns = te_cols
        test_encoded.index = test_df.index

        if ordinal:
            test_encoded = test_encoded.rank(method="dense").astype(int)

        test_out[te_cols] = test_encoded

    return train_out, test_out


def cast_columns(df: pd.DataFrame, columns: list, target_dtype: str) -> pd.DataFrame:
    """
    Casts selected columns of a DataFrame to a target dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to cast.
    columns : list of str
        List of column names to cast.
    target_dtype : str
        The desired target dtype (e.g., 'float32', 'int32', 'category').

    Returns
    -------
    pd.DataFrame
        New DataFrame with selected columns casted to the target dtype.
    """
    df = df.copy()
    df[columns] = df[columns].astype(target_dtype)
    return df


class Solver:
    def __init__(
        self,
        train_df,
        test_df,
        imputer,
        outliers_detector,
        scaler,
        feature_selector,
        preload_folds=True,
        n_folds=10,
        seed=42,
        verbose=True,
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.seed = seed
        self.verbose = verbose
        self.n_folds = n_folds
        self.preload_folds = preload_folds

        self.imputer = imputer
        self.outliers_detector = outliers_detector
        self.scaler = scaler
        self.feature_selector = feature_selector

    def _print(self, *args):
        if self.verbose:
            print(*args)
        else:
            pass

    def _count_missing_values(self, df):
        """
        Count the number of missing cells in each column of the DataFrame.
        """
        missing_count = df.isnull().sum().sum()
        total_entries = df.size
        missing_pct = missing_count / total_entries
        self._print(
            f"Total missing values: {missing_count}/{total_entries} ({missing_pct*100:.3f}%)"
        )
        return missing_pct

    def kfold_preparation(self):
        """
        k-fold logic:
        - Since the target is listening time minutes, we can discretize the target into bins.
        - We define bins by the quantiles of the target variable.
        - We use StratifiedKFold to ensure that each fold has a similar distribution of the target variable.
        """
        y = self.train_df["Listening_Time_minutes"]
        self.train_df["target"] = pd.qcut(y, q=10, labels=False, duplicates="drop")
        self.train_df["kfold"] = -1
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(self.train_df, self.train_df["target"])
        ):
            self.train_df.loc[val_idx, "kfold"] = fold
        self.train_df.drop(columns=["target", "id"], inplace=True)

        self.folds = []
        for fold in range(self.n_folds):
            val_mask = self.train_df["kfold"] == fold
            x_train = self.train_df[~val_mask].drop(columns=["kfold"])
            x_val = self.train_df[val_mask].drop(columns=["kfold"])
            pd.DataFrame(x_train).to_csv(f"data/train_fold_{fold}.csv", index=False)
            pd.DataFrame(x_val).to_csv(f"data/val_fold_{fold}.csv", index=False)
            y_train = x_train["Listening_Time_minutes"]
            y_val = x_val["Listening_Time_minutes"]
            self.folds.append((x_train, y_train, x_val, y_val))

        # # Double-check to see if the distriubution of the target variable is similar across folds
        # for fold in range(self.n_folds):
        #     self._print(f"Fold {fold} distribution:")
        #     self._print(
        #         self.train_df[self.train_df["kfold"] == fold][
        #             "Listening_Time_minutes"
        #         ].describe()
        #     )

    def transform_features(self, train_df, test_df):
        """
        Transforms both train and test dataframes with the same logic:
        List of features:
            - Podcast_Name (text)
            - Episode_Title (text)
            - Episode_Length_minutes (numeric)
            - Genre (categorical)
            - Host_Popularity_percentage (numeric)
            - Publication_Day (categorical)
            - Publication_Time (categorical)
            - Guest_Popularity_percentage (numeric)
            - Number_of_Ads (numeric)
            - Episode_Sentiment (categorical)
        Target:
            - Listening_Time_minutes (numeric)
        """
        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df["Episode_Num"] = (
            train_df["Episode_Title"].str.extract(r"(\d+)$").astype(int)
        )
        test_df["Episode_Num"] = (
            test_df["Episode_Title"].str.extract(r"(\d+)$").astype(int)
        )
        train_df = train_df.drop(columns=["Episode_Title"], errors="ignore")
        test_df = test_df.drop(columns=["Episode_Title"], errors="ignore")

        day_mapping = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }
        train_df["Publication_Day"] = train_df["Publication_Day"].map(day_mapping)
        test_df["Publication_Day"] = test_df["Publication_Day"].map(day_mapping)

        time_mapping = {
            "Morning": 0,
            "Afternoon": 1,
            "Evening": 2,
            "Night": 3,
        }
        train_df["Publication_Time"] = train_df["Publication_Time"].map(time_mapping)
        test_df["Publication_Time"] = test_df["Publication_Time"].map(time_mapping)

        sentiment_mapping = {
            "Negative": -1,
            "Neutral": 0,
            "Positive": 1,
        }
        train_df["Episode_Sentiment"] = train_df["Episode_Sentiment"].map(
            sentiment_mapping
        )
        test_df["Episode_Sentiment"] = test_df["Episode_Sentiment"].map(
            sentiment_mapping
        )

        features_to_impute = [
            "Episode_Length_minutes",
            "Host_Popularity_percentage",
            "Guest_Popularity_percentage",
            "Number_of_Ads",
        ]
        self.imputer.fit(train_df[features_to_impute])
        train_imputed_array = self.imputer.transform(train_df[features_to_impute])
        test_imputed_array = self.imputer.transform(test_df[features_to_impute])
        for feature in features_to_impute:
            train_df[feature] = train_imputed_array[
                :, features_to_impute.index(feature)
            ]
            test_df[feature] = test_imputed_array[:, features_to_impute.index(feature)]
        self._count_missing_values(train_df)
        self._count_missing_values(test_df)

        train_df = cast_columns(train_df, ["Number_of_Ads"], "int")
        test_df = cast_columns(test_df, ["Number_of_Ads"], "int")

        # Add is_weekend feature
        train_df["is_weekend"] = train_df["Publication_Day"].apply(
            lambda x: 1 if x in ["Saturday", "Sunday"] else 0
        )
        test_df["is_weekend"] = test_df["Publication_Day"].apply(
            lambda x: 1 if x in ["Saturday", "Sunday"] else 0
        )

        train_df["LP_Episode_Length_minutes"] = (
            0.7192 * train_df["Episode_Length_minutes"]
        )
        test_df["LP_Episode_Length_minutes"] = (
            0.7192 * test_df["Episode_Length_minutes"]
        )
        train_df = cast_columns(train_df, ["LP_Episode_Length_minutes"], "float")
        test_df = cast_columns(test_df, ["LP_Episode_Length_minutes"], "float")

        self.categorical_features = [
            "Podcast_Name",
            "Episode_Num",
            "Genre",
            "Publication_Day",
            "Publication_Time",
            "Episode_Sentiment",
            "is_weekend",
        ]
        train_df = cast_columns(train_df, self.categorical_features, "category")
        test_df = cast_columns(test_df, self.categorical_features, "category")

        train_df["Rounded_LP_Episode_Length_minutes"] = train_df[
            "LP_Episode_Length_minutes"
        ].round(1)
        test_df["Rounded_LP_Episode_Length_minutes"] = test_df[
            "LP_Episode_Length_minutes"
        ].round(1)
        train_df = cast_columns(
            train_df, ["Rounded_LP_Episode_Length_minutes"], "float"
        )
        test_df = cast_columns(test_df, ["Rounded_LP_Episode_Length_minutes"], "float")

        combined_features = []
        features_combinations = [
            ["Rounded_LP_Episode_Length_minutes", "Host_Popularity_percentage"],
            ["Rounded_LP_Episode_Length_minutes", "Guest_Popularity_percentage"],
            ["Rounded_LP_Episode_Length_minutes", "Number_of_Ads"],
            ["Episode_Num", "Host_Popularity_percentage"],
            ["Episode_Num", "Guest_Popularity_percentage"],
            ["Episode_Num", "Number_of_Ads"],
            ["Host_Popularity_percentage", "Guest_Popularity_percentage"],
            ["Host_Popularity_percentage", "Number_of_Ads"],
            ["Host_Popularity_percentage", "Episode_Sentiment"],
            ["Rounded_LP_Episode_Length_minutes", "Podcast_Name"],
            ["Episode_Num", "Podcast_Name"],
            ["Guest_Popularity_percentage", "Podcast_Name"],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Host_Popularity_percentage",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Guest_Popularity_percentage",
            ],
            ["Rounded_LP_Episode_Length_minutes", "Episode_Num", "Number_of_Ads"],
            ["Rounded_LP_Episode_Length_minutes", "Episode_Num", "Episode_Sentiment"],
            ["Rounded_LP_Episode_Length_minutes", "Episode_Num", "Publication_Day"],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Number_of_Ads",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Episode_Sentiment",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Publication_Day",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Publication_Time",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Guest_Popularity_percentage",
                "Number_of_Ads",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Guest_Popularity_percentage",
                "Publication_Day",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Guest_Popularity_percentage",
                "Publication_Time",
            ],
            ["Rounded_LP_Episode_Length_minutes", "Number_of_Ads", "Episode_Sentiment"],
            ["Rounded_LP_Episode_Length_minutes", "Number_of_Ads", "Publication_Day"],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Sentiment",
                "Publication_Time",
            ],
            [
                "Episode_Num",
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
            ],
            ["Episode_Num", "Host_Popularity_percentage", "Number_of_Ads"],
            ["Episode_Num", "Host_Popularity_percentage", "Episode_Sentiment"],
            ["Episode_Num", "Host_Popularity_percentage", "Publication_Day"],
            ["Episode_Num", "Host_Popularity_percentage", "Publication_Time"],
            ["Episode_Num", "Host_Popularity_percentage", "Genre"],
            ["Episode_Num", "Guest_Popularity_percentage", "Number_of_Ads"],
            ["Episode_Num", "Guest_Popularity_percentage", "Episode_Sentiment"],
            ["Episode_Num", "Guest_Popularity_percentage", "Publication_Day"],
            ["Episode_Num", "Guest_Popularity_percentage", "Publication_Time"],
            ["Episode_Num", "Guest_Popularity_percentage", "Genre"],
            ["Episode_Num", "Number_of_Ads", "Episode_Sentiment"],
            [
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
                "Number_of_Ads",
            ],
            [
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
                "Episode_Sentiment",
            ],
            [
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
                "Publication_Day",
            ],
            [
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
                "Publication_Time",
            ],
            ["Host_Popularity_percentage", "Number_of_Ads", "Publication_Day"],
            ["Guest_Popularity_percentage", "Number_of_Ads", "Episode_Sentiment"],
            ["Guest_Popularity_percentage", "Number_of_Ads", "Genre"],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Host_Popularity_percentage",
                "Number_of_Ads",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Host_Popularity_percentage",
                "Episode_Sentiment",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Host_Popularity_percentage",
                "Publication_Day",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Host_Popularity_percentage",
                "Publication_Time",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Host_Popularity_percentage",
                "Genre",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Guest_Popularity_percentage",
                "Number_of_Ads",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Guest_Popularity_percentage",
                "Episode_Sentiment",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Guest_Popularity_percentage",
                "Publication_Day",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Guest_Popularity_percentage",
                "Publication_Time",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Number_of_Ads",
                "Episode_Sentiment",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Number_of_Ads",
                "Publication_Day",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Number_of_Ads",
                "Publication_Time",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Publication_Day",
                "Publication_Time",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Publication_Day",
                "Genre",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
                "Number_of_Ads",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
                "Episode_Sentiment",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
                "Publication_Day",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
                "Publication_Time",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Number_of_Ads",
                "Episode_Sentiment",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Number_of_Ads",
                "Publication_Day",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Publication_Day",
                "Publication_Time",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Host_Popularity_percentage",
                "Publication_Day",
                "Genre",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Guest_Popularity_percentage",
                "Number_of_Ads",
                "Episode_Sentiment",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Guest_Popularity_percentage",
                "Number_of_Ads",
                "Publication_Day",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Guest_Popularity_percentage",
                "Number_of_Ads",
                "Publication_Time",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Guest_Popularity_percentage",
                "Number_of_Ads",
                "Genre",
            ],
            [
                "Rounded_LP_Episode_Length_minutes",
                "Episode_Num",
                "Publication_Time",
                "Podcast_Name",
            ],
            [
                "Episode_Num",
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
                "Number_of_Ads",
            ],
            [
                "Episode_Num",
                "Host_Popularity_percentage",
                "Guest_Popularity_percentage",
                "Episode_Sentiment",
            ],
            [
                "Episode_Num",
                "Host_Popularity_percentage",
                "Number_of_Ads",
                "Publication_Day",
            ],
            [
                "Episode_Num",
                "Host_Popularity_percentage",
                "Number_of_Ads",
                "Publication_Time",
            ],
            [
                "Episode_Num",
                "Host_Popularity_percentage",
                "Episode_Sentiment",
                "Publication_Day",
            ],
            [
                "Episode_Num",
                "Host_Popularity_percentage",
                "Episode_Sentiment",
                "Publication_Time",
            ],
            ["Episode_Num", "Host_Popularity_percentage", "Episode_Sentiment", "Genre"],
            [
                "Episode_Num",
                "Host_Popularity_percentage",
                "Publication_Day",
                "Publication_Time",
            ],
            ["Episode_Num", "Host_Popularity_percentage", "Publication_Time", "Genre"],
            [
                "Episode_Num",
                "Guest_Popularity_percentage",
                "Number_of_Ads",
                "Episode_Sentiment",
            ],
            ["Episode_Num", "Guest_Popularity_percentage", "Number_of_Ads", "Genre"],
            [
                "Episode_Num",
                "Host_Popularity_percentage",
                "Episode_Sentiment",
                "Podcast_Name",
            ],
            [
                "Host_Popularity_percentage",
                "Number_of_Ads",
                "Episode_Sentiment",
                "Podcast_Name",
            ],
            [
                "Host_Popularity_percentage",
                "Number_of_Ads",
                "Publication_Day",
                "Podcast_Name",
            ],
            [
                "Host_Popularity_percentage",
                "Number_of_Ads",
                "Publication_Time",
                "Podcast_Name",
            ],
        ]
        # for feature_group in features_combinations:
        #     feature_name = "_".join(feature_group)
        #     train_df[feature_name] = (
        #         train_df[list(feature_group)].astype(str).agg("_".join, axis=1)
        #     )
        #     test_df[feature_name] = (
        #         test_df[list(feature_group)].astype(str).agg("_".join, axis=1)
        #     )
        #     combined_features.append(feature_name)
        for feature_group in tqdm(features_combinations, desc="Combining features"):
            feature_name = "_".join(feature_group)

            # Use .astype(str) once and avoid repeated coercions
            train_df[feature_name] = (
                train_df.loc[:, feature_group].astype(str).agg("_".join, axis=1)
            )
            test_df[feature_name] = (
                test_df.loc[:, feature_group].astype(str).agg("_".join, axis=1)
            )
            combined_features.append(feature_name)

        for feature in combined_features:
            train_df = cast_columns(train_df, [feature], "category")
            test_df = cast_columns(test_df, [feature], "category")

        # Target encoding for combined categorical features
        train_df, test_df = oof_target_encode(
            train_df,
            test_df,
            features=combined_features,
            target_col="Listening_Time_minutes",
            n_splits=5,
            seed=42,
            smooth="auto",
            ordinal=False,
            prefix="TE_",
        )
        self._print(
            train_df.head(5),
            test_df.head(5),
        )

        # Double-check to see if the target is dropped
        to_be_dropped = ["Listening_Time_minutes", "Rounded_LP_Episode_Length_minutes"]
        to_be_dropped += combined_features
        train_df.drop(columns=to_be_dropped, inplace=True, errors="ignore")
        test_df.drop(columns=to_be_dropped, inplace=True, errors="ignore")
        self._print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        self.categorical_cols = list(
            train_df.select_dtypes(include=["category"]).columns
        )
        self._print(f"Number of categorical columns: {len(self.categorical_cols)}")
        self._print("Categorical columns:", list(self.categorical_cols))
        self.numerical_cols = list(train_df.select_dtypes(include=["number"]).columns)
        self._print(f"Number of numerical columns: {len(self.numerical_cols)}")
        self._print("Numerical columns:", list(self.numerical_cols))
        return train_df, test_df

    def preprocess_data(self):
        if self.preload_folds is True:
            self._print("Loading precomputed folds...")
            self.folds = []
            for fold in range(self.n_folds):
                x_train = pd.read_csv(f"data/train_fold_{fold}.csv")
                x_val = pd.read_csv(f"data/val_fold_{fold}.csv")
                y_train = x_train["Listening_Time_minutes"]
                y_val = x_val["Listening_Time_minutes"]
                self.folds.append((x_train, y_train, x_val, y_val))
        else:
            self.kfold_preparation()

    def train(self, model):
        """
        Train the model using k-fold cross-validation.
        """
        self._print("Training model...")
        scores = []
        oof_targets = []
        oof_preds = []
        oof_rmse = []
        test_id = self.test_df["id"]
        test_preds = []
        for fold_id, (x_train, y_train, x_val, y_val) in enumerate(self.folds):
            x_train_copy = x_train.copy()
            x_train, x_val = self.transform_features(x_train_copy, x_val)
            # If CatBoost is used, we pass the categorical features
            if isinstance(model, CatBoostRegressor):
                model.fit(
                    x_train,
                    y_train,
                    eval_set=(x_val, y_val),
                    cat_features=self.categorical_cols,
                    verbose=100,
                )
            else:
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
            y_pred = model.predict(x_val)
            y_pred = self._post_process_prediction(y_pred)
            oof_targets.append(y_val)
            oof_preds.append(y_pred)
            rmse = root_mean_squared_error(y_val, y_pred)
            oof_rmse.append(rmse)

            # Make predictions on the test set
            x_test = self.test_df.drop(columns=["id"])
            _, x_test = self.transform_features(x_train_copy, x_test)
            y_test_pred = model.predict(x_test)
            y_test_pred = self._post_process_prediction(y_test_pred)
            test_preds.append(y_test_pred)

        # Calculate the micro-RMSE
        oof_targets = np.concatenate(oof_targets)
        oof_preds = np.concatenate(oof_preds)
        rmse = root_mean_squared_error(oof_targets, oof_preds)
        self._print(f"Micro-RMSE: {rmse:.4f}")

        # Calculate the macro-RMSE and confidence interval
        oof_rmse = np.array(oof_rmse)
        macro_rmse = np.mean(oof_rmse)
        rmse_std = np.std(oof_rmse)
        rmse_ci = 1.96 * rmse_std / np.sqrt(self.n_folds)
        self._print(f"Macro-RMSE: {macro_rmse:.4f} ± {rmse_ci:.4f} (95% CI)")

        # Average the test predictions
        test_preds = np.mean(test_preds, axis=0)
        test_preds = self._post_process_prediction(test_preds)
        # Save the predictions with "id" column
        submission = pd.DataFrame({"id": test_id, "Listening_Time_minutes": test_preds})
        submission.to_csv("submission.csv", index=False)

    def _post_process_prediction(self, y):
        """
        Post-process the predictions to ensure they are non-negative.
        """
        y = np.clip(y, 0, None)
        return y


def main(verbose=True, seed=42, load=True):
    solver_params = {
        # "imputer": {
        #     "name": "IterativeImputer",
        #     "params": {
        #         "max_iter": 5,
        #         "n_nearest_features": 10,
        #         "random_state": seed,
        #         "verbose": 0,
        #     },
        # },
        "imputer": {
            "name": "SimpleImputer",
            "params": {
                "strategy": "median",
            },
        },
        "scaler": {
            "name": "QuantileTransformer",
            "params": {
                "output_distribution": "uniform",
                "n_quantiles": 100,
                "random_state": seed,
            },
        },
    }
    solver = Solver(
        train_df=pd.read_csv("data/train.csv"),
        test_df=pd.read_csv("data/test.csv"),
        imputer=globals()[solver_params["imputer"]["name"]](
            **solver_params["imputer"]["params"]
        ),
        outliers_detector=None,
        scaler=globals()[solver_params["scaler"]["name"]](
            **solver_params["scaler"]["params"]
        ),
        feature_selector=None,
        n_folds=5,
        verbose=verbose,
        preload_folds=load,
        seed=seed,
    )
    solver.preprocess_data()

    model_params = {
        "name": "Ridge",
        "params": {
            "alpha": 1e-3,
            "random_state": seed,
        },
    }
    model_params = {
        "name": "XGBRegressor",
        "params": {
            "booster": "gbtree",
            "enable_categorical": True,  # Enable categorical features
            "eval_metric": "rmse",  # Evaluation metric
            "tree_method": "hist",  # Fast histogram-based growth
            "colsample_bytree": 0.5,  # Feature subsample per tree
            "subsample": 0.9,  # Row subsample per tree
            "learning_rate": 0.02,  # Conservative learning rate
            "n_estimators": 50_000,  # Sufficient depth for boosting
            "max_depth": 14,  # Deep enough to capture interactions
            "min_child_weight": 10,  # Regularizes splits for stability
            # "gamma": 0.1,  # Minimum loss reduction for splits
            "random_state": seed,  # Ensure reproducibility
            "verbosity": 1,  # Silence output
            "early_stopping_rounds": 150,
            "n_jobs": -1,
        },
    }
    # model_params = {
    #     "name": "CatBoostRegressor",
    #     "params": {
    #         "iterations": 10000,  # Sufficient number of iterations
    #         "depth": 14,  # Medium depth to avoid overfitting
    #         "learning_rate": 0.02,  # Conservative learning rate
    #         "l2_leaf_reg": 1.0,  # L2 regularization
    #         "bagging_temperature": 0.5,  # Bagging temperature for randomness
    #         "rsm": 0.8,  # Random subspace method
    #         "loss_function": "RMSE",
    #         "eval_metric": "RMSE",
    #         "random_seed": seed,
    #         "allow_writing_files": False,  # Disable logging files
    #         "early_stopping_rounds": 150,
    #         "thread_count": 4,
    #     },
    # }
    solver.train(model=globals()[model_params["name"]](**model_params["params"]))

    # def objective(trial):
    #     model_params = {
    #         "name": "XGBRegressor",
    #         "params": {
    #             "booster": "gbtree",
    #             "tree_method": "hist",  # Faster histogram-based training
    #             "reg_lambda": trial.suggest_float("xgb_lambda", 1e-3, 10.0, log=True),
    #             "reg_alpha": trial.suggest_float("xgb_alpha", 1e-3, 10.0, log=True),
    #             "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
    #             "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
    #             "learning_rate": trial.suggest_float("xgb_learning_rate", 0.008, 0.05, log=True),
    #             "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000),
    #             "max_depth": trial.suggest_int("xgb_max_depth", 4, 30),
    #             "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 100),
    #             "gamma": trial.suggest_float("xgb_gamma", 0, 10.0),
    #             "random_state": seed,
    #             "verbosity": 0,
    #             "n_jobs": -1,
    #         },
    #     }
    #     score = solver.train(
    #         model=globals()[model_params["name"]](**model_params["params"])
    #     )
    #     return score

    # import optuna
    # study = optuna.create_study(
    #     direction="minimize",
    #     # sampler=optuna.samplers.TPESampler(seed=SEED),
    #     # pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    # )

    # study.optimize(objective, n_trials=1000, n_jobs=1)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
