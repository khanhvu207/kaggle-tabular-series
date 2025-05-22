import gc
import numpy as np
import pandas as pd
import polars as pl
from typing import List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold


def encode_target_statistics(
    train_df: pd.DataFrame,
    target_dfs: List[pd.DataFrame],
    target_col: str,
    target_encoding_features: List[str],
    stats: List[str],
    seed: int = 42,
) -> tuple[pd.DataFrame, List[pd.DataFrame], List[str]]:
    """
    Applies RegressionTargetEncoder for a list of statistics over combined_features.

    Parameters:
        train_df (pd.DataFrame): Training data.
        target_dfs (List[pd.DataFrame]): List of test/val dataframes.
        target_col (str): Name of the target column.
        combined_features (List[str]): List of categorical features to encode.
        stats (List[str]): List of statistics to encode with.
        seed (int): Random seed for CV.

    Returns:
        train_df (pd.DataFrame): Training dataframe with TE features.
        target_dfs (List[pd.DataFrame]): List of target dataframes with TE features.
        te_features (List[str]): Names of added TE features.
    """
    te_features = []

    for stat in stats:
        encoder = RegressionTargetEncoder(statistic=stat, cv=5, random_state=seed)

        X_train = train_df[target_encoding_features]
        y_train = train_df[target_col]
        train_encoded = encoder.fit_transform(X_train, y_train)

        target_encoded_list = [
            encoder.transform(target_df[target_encoding_features])
            for target_df in target_dfs
        ]

        col_names = [f"TE_{feature}_{stat}" for feature in target_encoding_features]
        train_te_df = pd.DataFrame(train_encoded, columns=col_names, index=train_df.index)
        target_te_dfs = [
            pd.DataFrame(encoded, columns=col_names, index=target_df.index)
            for encoded, target_df in zip(target_encoded_list, target_dfs)
        ]

        train_df = pd.concat([train_df, train_te_df], axis=1)
        for i in range(len(target_dfs)):
            target_dfs[i] = pd.concat([target_dfs[i], target_te_dfs[i]], axis=1)

        te_features.extend(col_names)
        gc.collect()

    return train_df, target_dfs, te_features

class RegressionTargetEncoder(BaseEstimator, TransformerMixin):
    _pl_string_cache_enabled = False

    def __init__(
        self, 
        statistic: str = 'mean',
        fill_value: Optional[float] = None,
        smooth: float = 0.0,
        cv: Optional[int] = 5,
        random_state: Optional[int] = 42,
    ):
        """
        Leakage-free target encoding for regression tasks.
        
        Parameters:
            statistic (str): Statistic to compute ('mean', 'median', 'q25', 'q75').
            fill_value (float): Value to fill for unseen categories.
            smooth (float): Smoothing factor for mean encoding.
            cv (int): Number of folds for cross-validation.
            random_state (int): Random seed for reproducibility.    
        """
        self.statistic = statistic
        self.fill_value = fill_value
        self.smooth = smooth
        self.cv = cv
        self.random_state = random_state
        self.mapping_ = {}
        self.global_stat_ = None
        self._fitted = False
        
        # Enable string cache once for all instances
        if not RegressionTargetEncoder._pl_string_cache_enabled:
            pl.enable_string_cache()
            RegressionTargetEncoder._pl_string_cache_enabled = True

    def _compute_stat_multi(self, X_df, y, stat="mean"):
        """
        X_df : pd.DataFrame with categorical columns
        y    : pd.Series with target
        Returns {feature: {category: stat}}
        """
        df = pl.from_pandas(X_df)
        df = df.with_columns(pl.Series("y", y))

        melted = df.melt(id_vars="y",
                         variable_name="feature",
                         value_name="x")

        if stat == "median":
            agg_expr = pl.col("y").median().alias("stat")
        elif stat == "q25":
            agg_expr = pl.col("y").quantile(0.25, "nearest").alias("stat")
        elif stat == "q75":
            agg_expr = pl.col("y").quantile(0.75, "nearest").alias("stat")
        else:
            agg_expr = getattr(pl.col("y"), stat)().alias("stat")

        res = (melted
               .group_by(["feature", "x"])
               .agg(pl.count().alias("count"), agg_expr))

        if stat == "mean" and self.smooth > 0:
            res = res.with_columns(
                (
                    (pl.col("count") * pl.col("stat") +
                     self.smooth * self.global_stat_) /
                    (pl.col("count") + self.smooth)
                ).alias("smoothed")
            ).with_columns(pl.col("smoothed").alias("stat"))

        out = {}
        for f, subdf in res.group_by("feature"):
            out[f] = dict(zip(subdf["x"].to_list(), subdf["stat"].to_list()))
        return out

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)

        # global stat for unseen categories
        if self.statistic == "q25":
            self.global_stat_ = y.quantile(0.25)
        elif self.statistic == "q75":
            self.global_stat_ = y.quantile(0.75)
        elif self.statistic == "median":
            self.global_stat_ = y.median()
        else:
            self.global_stat_ = getattr(y, self.statistic)()

        self.mapping_ = self._compute_stat_multi(X, y, self.statistic)
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("You must fit the encoder first.")

        X = pd.DataFrame(X)
        default = (self.fill_value
                   if self.fill_value is not None else self.global_stat_)

        cols = []
        for col in X.columns:
            mapping = self.mapping_.get(col, {})
            enc = (X[col]
                   .map(mapping)
                   .astype(float)
                   .fillna(default)
                   .to_numpy()
                   .reshape(-1, 1))
            cols.append(enc)
        return np.hstack(cols)

    def fit_transform(self, X, y):
        if self.cv is None:
            self.fit(X, y)
            return self.transform(X)

        X = pd.DataFrame(X)
        y = pd.Series(y)
        oof = np.zeros((len(y), X.shape[1]), dtype=float)

        if self.statistic == "q25":
            self.global_stat_ = y.quantile(0.25)
        elif self.statistic == "q75":
            self.global_stat_ = y.quantile(0.75)
        elif self.statistic == "median":
            self.global_stat_ = y.median()
        else:
            self.global_stat_ = getattr(y, self.statistic)()

        bins = pd.qcut(y, self.cv, labels=False, duplicates='drop')
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                             random_state=self.random_state)

        for train_idx, val_idx in kf.split(X, bins):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            fold_map = self._compute_stat_multi(X_tr, y_tr, self.statistic)

            for j, col in enumerate(X.columns):
                mapping = fold_map.get(col, {})
                default = (self.fill_value
                           if self.fill_value is not None else self.global_stat_)
                oof[val_idx, j] = (X_val[col]
                                   .map(mapping)
                                   .astype(float)
                                   .fillna(default)
                                   .to_numpy())

        self.mapping_ = self._compute_stat_multi(X, y, self.statistic)
        self._fitted = True
        return oof