import copy

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel,
                                       SelectKBest, VarianceThreshold, chi2,
                                       f_classif, f_regression,
                                       mutual_info_regression)
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    StackingRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import root_mean_squared_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


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
        quantiles = np.quantile(y, [0.25, 0.5, 0.75])
        bins = [-np.inf, *quantiles, np.inf]
        labels = np.arange(len(bins) - 1)
        self.train_df["target"] = pd.cut(y, bins=bins, labels=labels)
        self.train_df["kfold"] = -1
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(self.train_df, self.train_df["target"])
        ):
            self.train_df.loc[val_idx, "kfold"] = fold
        self.train_df.drop(columns=["target", "id"], inplace=True)

        # Storing the folds for later use
        self.folds = []
        for fold in range(self.n_folds):
            val_mask = self.train_df["kfold"] == fold
            x_train = self.train_df[~val_mask].drop(
                columns=["kfold"]
            )
            x_val = self.train_df[val_mask].drop(
                columns=["kfold"]
            )
            pd.DataFrame(x_train).to_csv(f"data/train_fold_{fold}.csv", index=False)
            pd.DataFrame(x_val).to_csv(f"data/val_fold_{fold}.csv", index=False)
            y_train = x_train["Listening_Time_minutes"]
            x_train.drop(columns=["Listening_Time_minutes"], inplace=True)
            y_val = x_val["Listening_Time_minutes"]
            x_val.drop(columns=["Listening_Time_minutes"], inplace=True)
            self.folds.append((x_train, y_train, x_val, y_val))

        # Double-check to see if the distriubution of the target variable is similar across folds
        for fold in range(self.n_folds):
            self._print(f"Fold {fold} distribution:")
            self._print(
                self.train_df[self.train_df["kfold"] == fold][
                    "Listening_Time_minutes"
                ].describe()
            )

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

        # Extracting the number from Episode_Title
        train_df['Episode_Title'].str.extract(r'(\d+)$').astype(int)
        test_df['Episode_Title'].str.extract(r'(\d+)$').astype(int)

        # Map Episode_Sentiment to numerical values
        sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        train_df['Episode_Sentiment'].map(sentiment_map).fillna(0).astype(int)
        test_df['Episode_Sentiment'].map(sentiment_map).fillna(0).astype(int)
        
        # We need to impute missing values for the following columns:
        # - Episode_Length_minutes (numeric)
        # - Host_Popularity_percentage (numeric)
        # - Guest_Popularity_percentage (numeric)
        # - Number_of_Ads (numeric)
        self._count_missing_values(train_df)
        self._count_missing_values(test_df)
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
            train_df[feature] = train_imputed_array[:, features_to_impute.index(feature)]
            test_df[feature] = test_imputed_array[:, features_to_impute.index(feature)]
        self._count_missing_values(train_df)
        self._count_missing_values(test_df)

        # Rescale numeric features
        numeric_features = [
            "Episode_Length_minutes",
            "Host_Popularity_percentage",
            "Guest_Popularity_percentage",
            "Number_of_Ads",
        ]
        self.scaler.fit(train_df[numeric_features])
        train_scaled_array = self.scaler.transform(train_df[numeric_features])
        test_scaled_array = self.scaler.transform(test_df[numeric_features])
        for feature in numeric_features:
            train_df[feature] = train_scaled_array[:, numeric_features.index(feature)]
            test_df[feature] = test_scaled_array[:, numeric_features.index(feature)]
        
        # Set some columns to be categorical
        categorical_features = [
            "Podcast_Name",
            "Episode_Title",
            "Genre",
            "Publication_Day",
            "Publication_Time",
            "Episode_Sentiment",
        ]
        for feature in categorical_features:
            train_df[feature] = train_df[feature].astype("category")
            test_df[feature] = test_df[feature].astype("category")
        
        # Some feature engineering
        train_df["Linear_prediction_from_Episode_Length_minutes"] = 0.7192 * train_df["Episode_Length_minutes"]
        test_df["Linear_prediction_from_Episode_Length_minutes"] = 0.7192 * test_df["Episode_Length_minutes"]
        train_df["Linear_prediction_from_Episode_Length_minutes"] = train_df["Linear_prediction_from_Episode_Length_minutes"].astype(float)
        test_df["Linear_prediction_from_Episode_Length_minutes"] = test_df["Linear_prediction_from_Episode_Length_minutes"].astype(float)

        return train_df, test_df

    def preprocess_data(self):
        if self.preload_folds is True:
            self._print("Loading precomputed folds...")
            self.folds = []
            for fold in range(self.n_folds):
                x_train = pd.read_csv(f"data/train_fold_{fold}.csv")
                x_val = pd.read_csv(f"data/val_fold_{fold}.csv")
                y_train = x_train["Listening_Time_minutes"]
                x_train.drop(columns=["Listening_Time_minutes"], inplace=True)
                y_val = x_val["Listening_Time_minutes"]
                x_val.drop(columns=["Listening_Time_minutes"], inplace=True)
                self.folds.append((x_train, y_train, x_val, y_val))
        else:
            self.kfold_preparation()

    def train(self, model):
        """
        Train the model using k-fold cross-validation.
        """
        self._print("Training model...")
        scores = []
        for fold_id, (x_train, y_train, x_val, y_val) in enumerate(self.folds):
            x_train, x_val = self.transform_features(x_train, x_val)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            y_pred = self._post_process_prediction(y_pred)
            rmse = root_mean_squared_error(y_val, y_pred)
            self._print(f"Fold {fold_id} RMSE: {rmse:.4f}")
            scores.append(rmse)

        self._print(f"Mean RMSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    def predict_test(self, model):
        """
        Fit the model on the entire training set and predict on the test set.
        """
        self._print("Predicting on test set...")
        y_train = self.train_df["Listening_Time_minutes"]
        x_train = self.train_df.drop(columns=["id", "Listening_Time_minutes"])
        test_id = self.test_df["id"]
        x_test = self.test_df.drop(columns=["id"])
        x_train, x_test = self.transform_features(x_train, x_test)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        # Post-process the predictions
        y_pred = self._post_process_prediction(y_pred)

        # Save the predictions with "id" column
        submission = pd.DataFrame({"id": test_id, "Listening_Time_minutes": y_pred})
        submission.to_csv("submission.csv", index=False)
        self._print("Predictions saved to submission.csv")
    
    def _post_process_prediction(self, y):
        """
        Post-process the predictions to ensure they are non-negative.
        """
        y = np.clip(y, 0, None)
        return y


def main(verbose=True, seed=42):
    solver_params = {
        "imputer": {
            "name": "IterativeImputer",
            "params": {
                "max_iter": 5,
                "n_nearest_features": 10,
                "random_state": seed,
                "verbose": 0,
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
        seed=seed,
    )
    solver.preprocess_data()

    # model_params = {
    #     "name": "Ridge",
    #     "params": {
    #         "alpha": 1e-3,
    #         "random_state": seed,
    #     },
    # }
    model_params = {
        "name": "XGBRegressor",
        "params": {
            "booster": "gbtree",
            "enable_categorical": True,        # Enable categorical features
            "max_cat_to_onehot": 10,            # Max categories for one-hot encoding
            "objective": "reg:squarederror",   # Regression task
            "eval_metric": "rmse",             # Evaluation metric
            "tree_method": "hist",             # Fast histogram-based growth
            "reg_lambda": 1.0,                 # L2 regularization
            "reg_alpha": 0.1,                  # L1 regularization
            "colsample_bytree": 0.8,           # Feature subsample per tree
            "subsample": 0.8,                  # Row subsample per tree
            "learning_rate": 0.015,            # Conservative learning rate
            "n_estimators": 300,               # Sufficient depth for boosting
            "max_depth": 20,                    # Deep enough to capture interactions
            "min_child_weight": 10,            # Regularizes splits for stability
            "gamma": 0.0,                      # Minimum gain for a split
            "random_state": seed,              # Ensure reproducibility
            "verbosity": 1,                    # Silence output
            "n_jobs": 4,
        },
    }
    text_features = [
        "Podcast_Name",
    ]
    cat_feautes = [
        "Episode_Title",
        "Genre",
        "Publication_Day",
        "Publication_Time",
        "Episode_Sentiment",
    ]
    # model_params = {
    #     "name": "CatBoostRegressor",
    #     "params": {
    #         "iterations": 100,            # Sufficient number of iterations
    #         "depth": 10,                  # Medium depth to avoid overfitting
    #         "learning_rate": 0.5,        # Conservative learning rate
    #         "cat_features": cat_feautes,
    #         "text_features": text_features,
    #         "loss_function": "RMSE",
    #         "random_seed": seed,
    #         "verbose": 1,                 # Suppress output
    #         "allow_writing_files": False, # Disable logging files
    #     },
    # }
    solver.train(
        model=globals()[model_params["name"]](**model_params["params"])
    )
    solver.predict_test(
        model=globals()[model_params["name"]](**model_params["params"])
    )
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
