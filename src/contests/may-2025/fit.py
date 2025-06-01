import os
import copy
import datetime
from itertools import combinations

import gc
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold, StratifiedKFold

from tqdm.auto import tqdm

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from autogluon.tabular import TabularPredictor

from src.common.utils import Timer, profile, add_k_tuple_columns, cast_columns_inplace
from src.common.preprocessing import encode_target_statistics

pd.set_option("display.max_columns", 100)


def transform_features(
    train_df: pd.DataFrame,
    target_dfs: list[pd.DataFrame],
    target_col: str = "Calories",
    seed=42,
):
    timer = Timer()
    train_df = train_df.copy()
    target_dfs = [target_df.copy() for target_df in target_dfs]
    cast_columns_inplace(train_df, ["id"], "int32")
    for target_df in target_dfs:
        cast_columns_inplace(target_df, ["id"], "int32")

    categorical_features = ["Sex"]
    int_features = []
    float_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

    train_df["Sex_Encoded"] = (
        train_df["Sex"].map({"female": -1, "male": 1}).astype("int32")
    )
    for target_df in target_dfs:
        target_df["Sex_Encoded"] = (
            target_df["Sex"].map({"female": -1, "male": 1}).astype("int32")
        )
    float_features.append("Sex_Encoded")

    def mifflin_st_jeor(row):
        """Basal Metabolic Rate (kcal / day).  Height in cm, Weight in kg."""
        if row["Sex"] == "male":
            s = 5
        else:  # female
            s = -161
        return 10 * row["Weight"] + 6.25 * row["Height"] - 5 * row["Age"] + s

    def boer_lbm(row):
        """Lean Body Mass (kg)."""
        if row["Sex"] == "male":
            return 0.407 * row["Weight"] + 0.267 * row["Height"] - 19.2
        else:
            return 0.252 * row["Weight"] + 0.473 * row["Height"] - 48.3

    def body_surface_area(row):
        """Mosteller BSA (m²)."""
        return np.sqrt(row["Height"] * row["Weight"] / 3600)

    def body_fat_pct(row):
        bmi = row["BMI"]
        adj = 10.8 if row["Sex"] == "male" else 0
        return 1.2 * bmi + 0.23 * row["Age"] - adj - 5.4

    def max_hr(age):
        """Age-based Max Heart Rate."""
        return 220 - age

    def vo2_est(hr, age):
        """Very rough HR→VO₂ regression (ml·kg⁻¹·min⁻¹)."""
        return 14.0 + 0.37 * hr - 0.006 * age

    train_df["BMI"] = train_df["Weight"] / (train_df["Height"] / 100) ** 2
    train_df["Total_Exertion"] = train_df["Duration"] * train_df["Heart_Rate"]
    train_df["Heart_Effort"] = train_df["Heart_Rate"] / train_df["Duration"]
    train_df["BMR"] = train_df.apply(mifflin_st_jeor, axis=1)
    train_df["LBM"] = train_df.apply(boer_lbm, axis=1)
    train_df["BSA"] = train_df.apply(body_surface_area, axis=1)
    train_df["Body_Fat_Pct"] = train_df.apply(body_fat_pct, axis=1)
    train_df["MHR"] = train_df["Age"].apply(max_hr)
    train_df["pct_MHR"] = train_df["Heart_Rate"] / train_df["MHR"]
    train_df["Training_Load"] = train_df["pct_MHR"] * train_df["Duration"]
    train_df["VO2"] = vo2_est(train_df["Heart_Rate"], train_df["Age"])
    train_df["Heat_Index"] = (train_df["Body_Temp"] - 37.0) * train_df["Duration"]
    for target_df in target_dfs:
        target_df["BMI"] = target_df["Weight"] / (target_df["Height"] / 100) ** 2
        target_df["Total_Exertion"] = target_df["Duration"] * target_df["Heart_Rate"]
        target_df["Heart_Effort"] = target_df["Heart_Rate"] / target_df["Duration"]
        target_df["BMR"] = target_df.apply(mifflin_st_jeor, axis=1)
        target_df["LBM"] = target_df.apply(boer_lbm, axis=1)
        target_df["BSA"] = target_df.apply(body_surface_area, axis=1)
        target_df["Body_Fat_Pct"] = target_df.apply(body_fat_pct, axis=1)
        target_df["MHR"] = target_df["Age"].apply(max_hr)
        target_df["pct_MHR"] = target_df["Heart_Rate"] / target_df["MHR"]
        target_df["Training_Load"] = target_df["pct_MHR"] * target_df["Duration"]
        target_df["VO2"] = vo2_est(target_df["Heart_Rate"], target_df["Age"])
        target_df["Heat_Index"] = (target_df["Body_Temp"] - 37.0) * target_df[
            "Duration"
        ]
    float_features.extend(
        [
            "BMI",
            "Total_Exertion",
            "Heart_Effort",
            "BMR",
            "LBM",
            "BSA",
            "Body_Fat_Pct",
            "MHR",
            "pct_MHR",
            "Training_Load",
            "VO2",
            "Heat_Index",
        ]
    )

    rounded_train = (
        train_df[float_features].round().add_prefix("Rounded_").astype("int32")
    )
    train_df = train_df.assign(**rounded_train)

    rounded_float_features = rounded_train.columns.tolist()

    for i, target_df in enumerate(target_dfs):
        rounded_target = (
            target_df[float_features].round().add_prefix("Rounded_").astype("int32")
        )
        target_dfs[i] = target_df.assign(**rounded_target)

    profile(timer, "Added new features")

    combined_features = []
    cast_columns_inplace(train_df, float_features, "float32")
    for target_df in target_dfs:
        cast_columns_inplace(target_df, float_features, "float32")

    train_df.drop(columns=combined_features + rounded_float_features, inplace=True)
    for target_df in target_dfs:
        target_df.drop(columns=combined_features, inplace=True)
        target_df.drop(columns=rounded_float_features, inplace=True)

    train_df.drop(columns=["Sex"], inplace=True)
    for target_df in target_dfs:
        target_df.drop(columns=["Sex"], inplace=True)

    cast_columns_inplace(train_df, categorical_features, "category")
    cast_columns_inplace(train_df, int_features, "int32")
    cast_columns_inplace(train_df, float_features, "float32")
    for target_df in target_dfs:
        cast_columns_inplace(target_df, categorical_features, "category")
        cast_columns_inplace(target_df, int_features, "int32")
        cast_columns_inplace(target_df, float_features, "float32")
    profile(timer, "Casting columns")

    # Assert that no columns are of type "object"
    assert len(train_df.select_dtypes(include=["object"]).columns) == 0
    for target_df in target_dfs:
        assert len(target_df.select_dtypes(include=["object"]).columns) == 0

    detector = LocalOutlierFactor(
        n_neighbors=50, algorithm="auto", contamination=0.01, n_jobs=-1
    )
    outlier_labels = detector.fit_predict(train_df.drop(columns=["id"]))
    train_df["is_outlier"] = outlier_labels
    train_df["is_outlier"] = train_df["is_outlier"].astype("int32")
    profile(timer, "Outlier detection")

    # y = np.log1p(train_df["Calories"])
    # train_df["bin"] = pd.qcut(y, q=10, labels=False, duplicates="drop")
    train_df["kfold"] = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["is_outlier"])):
        train_df.loc[val_idx, "kfold"] = fold

    # train_df.drop(columns=["bin"], inplace=True)
    cast_columns_inplace(train_df, ["kfold"], "int32")
    profile(timer, "KFold preparation")

    print(train_df.head(5))
    print(train_df.info())
    for target_df in target_dfs:
        print(target_df.info())

    return train_df, target_dfs


class Solver:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        preload_folds: bool = True,
        n_folds: int = 10,
        seed: int = 42,
        verbose: bool = True,
        log_dir: str = "logs",
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.target_col = target_col
        self.seed = seed
        self.verbose = verbose
        self.n_folds = n_folds
        self.preload_folds = preload_folds
        self.log_dir = log_dir

        # Convert "object" columns to "category"
        for col in self.train_df.select_dtypes(include=["object"]).columns:
            self.train_df[col] = self.train_df[col].astype("category")

        for col in self.test_df.select_dtypes(include=["object"]).columns:
            self.test_df[col] = self.test_df[col].astype("category")

        self._print(self.train_df.info())
        self._print(self.test_df.info())

    def _create_log_dir(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)

    def _print(self, *args):
        if self.verbose:
            print(*args)
        else:
            pass

    def _kfold_preparation(self):
        self.folds = []
        for fold in tqdm(range(self.n_folds), desc="Folds preparation"):
            val_mask = self.train_df["kfold"] == fold
            x_train = self.train_df[~val_mask].drop(columns=["kfold"])
            x_val = self.train_df[val_mask].drop(columns=["kfold"])
            y_train = x_train[self.target_col]
            y_val = x_val[self.target_col]
            self.folds.append((x_train, y_train, x_val, y_val))

    def train(self, model, metrics: dict):
        if metrics is None:
            raise ValueError("Metrics cannot be None.")

        self._kfold_preparation()

        oof_scores = dict()
        for metric_name, _ in metrics.items():
            oof_scores[metric_name] = []

        oof_ids = []
        oof_preds = []
        test_preds = []
        x_test = self.test_df.copy()
        test_ids = self.test_df["id"]
        x_test.drop(columns=["id"], inplace=True)
        for fold_id, (x_train, y_train, x_val, y_val) in enumerate(self.folds):
            val_id = x_val["id"]
            x_train.drop(columns=["id", self.target_col], inplace=True)
            x_val.drop(columns=["id", self.target_col], inplace=True)
            y_train = self._target_map(y_train)
            y_val = self._target_map(y_val)

            if isinstance(model, XGBRegressor):
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
            elif isinstance(model, CatBoostRegressor):
                model.fit(
                    x_train,
                    y_train,
                    eval_set=(x_val, y_val),
                    verbose=100,
                )
            elif isinstance(model, LGBMRegressor):
                model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    callbacks=[
                        log_evaluation(period=100),
                        early_stopping(stopping_rounds=500),
                    ],
                )
            elif isinstance(model, TabularPredictor):
                merged_train_df = pd.concat([x_train, y_train], axis=1)
                merged_val_df = pd.concat([x_val, y_val], axis=1)
                model.fit(merged_train_df, merged_val_df, time_limit=60 * 5)
            else:
                model.fit(x_train, y_train)

            y_pred = self._post_process_prediction(
                self._target_unmap(model.predict(x_val))
            )

            if (
                isinstance(model, XGBRegressor)
                or isinstance(model, CatBoostRegressor)
                or isinstance(model, LGBMRegressor)
                or isinstance(model, TabularPredictor)
            ):
                y_test_pred = model.predict(x_test)
            else:
                y_test_pred = model.predict(x_test)

            test_preds.append(y_test_pred)
            oof_ids.append(val_id)
            oof_preds.append(y_pred)

            for metric_name, metric in metrics.items():
                score = metric(
                    np.log(self._target_unmap(y_val) + 1), np.log(y_pred + 1)
                )
                oof_scores[metric_name].append(score)

                print(
                    f"Fold {fold_id + 1}/{self.n_folds}, " f"{metric_name}: {score:.6f}"
                )

        # Create log directory
        mean_rmse = np.mean(oof_scores["rmse"])
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(self.log_dir, time_stamp) + f"-rmse_{mean_rmse:.5f}"
        self._create_log_dir(self.log_dir)

        self._print("OOF scores:")
        metric_names = list(metrics.keys())
        stats_df = pd.DataFrame(columns=metric_names)
        for metric_name, scores in oof_scores.items():
            stats_df[metric_name] = scores
            mean_score, std_error = np.mean(scores), np.std(scores) / np.sqrt(
                len(scores)
            )
            self._print(f"{metric_name}: {scores} ({mean_score:.5f} ± {std_error:.5f})")
        stats_df.to_csv(os.path.join(self.log_dir, "oof_scores.csv"), index=False)

        # Sort the oof_ids and oof_preds by id
        oof_ids = np.concatenate(oof_ids)
        oof_preds = np.concatenate(oof_preds)
        sorted_indices = np.argsort(oof_ids)
        oof_ids = oof_ids[sorted_indices]
        oof_preds = oof_preds[sorted_indices]
        oof_df = pd.DataFrame({"id": oof_ids, self.target_col: oof_preds})
        oof_df.to_csv(os.path.join(self.log_dir, "oof.csv"), index=False)

        test_preds = np.mean(test_preds, axis=0)
        test_preds = self._post_process_prediction(self._target_unmap(test_preds))
        submission = pd.DataFrame({"id": test_ids, self.target_col: test_preds})
        submission.to_csv(os.path.join(self.log_dir, "submission.csv"), index=False)
        return oof_scores

    def _target_map(self, y):
        y = np.log(y + 1)
        return y

    def _target_unmap(self, y):
        y = np.exp(y) - 1
        return y

    def _post_process_prediction(self, y_pred):
        # y_pred = np.round(y_pred)
        y_pred = np.clip(y_pred, 0, None)
        return y_pred


def main(verbose=True, seed=42, log_dir="l1_logs"):
    # train_df = pd.read_parquet("data/train.parquet")
    # test_df = pd.read_parquet("data/test.parquet")
    # engineered_train_df, [engineered_test_df] = transform_features(
    #     train_df=train_df, target_dfs=[test_df], target_col="Calories"
    # )
    # engineered_train_df.to_parquet("data/train_v5.parquet", index=False)
    # engineered_test_df.to_parquet("data/test_v5.parquet", index=False)
    # return

    solver = Solver(
        train_df=pd.read_parquet("data/train_v3.parquet"),
        test_df=pd.read_parquet("data/test_v3.parquet"),
        target_col="Calories",
        n_folds=5,
        verbose=verbose,
        seed=seed,
        log_dir=log_dir,
    )

    model_params = {
        "name": "XGBRegressor",
        "params": {
            "booster": "gbtree",
            "enable_categorical": True,
            "eval_metric": "rmse",
            "tree_method": "hist",
            "learning_rate": 0.01,
            "n_estimators": 50_000,
            "max_depth": 7,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "random_state": seed,
            "verbosity": 1,
            "early_stopping_rounds": 500,
            "n_jobs": 8,
        },
    }
    model_params = {
        "name": "LGBMRegressor",
        "params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.01,
            "n_estimators": 50_000,
            "num_leaves": 128,
            "max_depth": 7,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "colsample_bytree": 0.7,
            "verbosity": -1,
            "random_state": seed,
            "n_jobs": 4,
        },
    }
    model_params = {
        "name": "CatBoostRegressor",
        "params": {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 50_000,
            "learning_rate": 0.03,  # Default: 0.03
            "depth": 6,  # Default: 6
            "l2_leaf_reg": 3.0,  # Default: 3.0
            "subsample": 1.0,  # Default: 0.8 for MVS bootstrap
            "rsm": 1.0,  # Default: 1.0
            "random_seed": seed,
            "early_stopping_rounds": 500,
            "thread_count": 4,
        },
    }

    scores = solver.train(
        model=globals()[model_params["name"]](**model_params["params"]),
        metrics={
            "rmse": root_mean_squared_error,
        },
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
