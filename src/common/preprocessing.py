import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   RobustScaler, StandardScaler, TargetEncoder)


def oof_target_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    prefix: str = "TE_",
    smooth: float = 0.0,
    n_splits: int = 5,
    seed: int =42,
    ordinal: bool = False,
    batch_size: int = 5,
):
    """
        Leakage-free out-of-fold target encoding
        Args:
            train_df (pd.DataFrame): Training DataFrame.
            test_df (pd.DataFrame): Testing DataFrame.
            features (list): List of feature names to encode.
            target_col (str): Target column name.
            smooth (float): Smoothing parameter for target encoding.
            n_splits (int): Number of splits for KFold cross-validation.
            seed (int): Random seed for reproducibility.
            ordinal (bool): If True, rank the encoded values.
            prefix (str): Prefix for the new encoded columns.
            batch_size (int): Size of feature batches for encoding.
        Returns:
            train_out (pd.DataFrame): Training DataFrame with encoded features.
            test_out (pd.DataFrame): Testing DataFrame with encoded features.
    """
    train_out = train_df.copy()
    test_out = test_df.copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def batch(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i : i + size]

    feature_batches = list(batch(features, batch_size or len(features)))

    for feature_batch in feature_batches:
        te_cols = [f"{prefix}{feature}" for feature in feature_batch]
        oof_encoded_batch = pd.DataFrame(
            index=train_df.index, columns=te_cols, dtype=float
        )

        for tr_idx, val_idx in kf.split(train_df):
            tr_part, val_part = train_df.iloc[tr_idx], train_df.iloc[val_idx]

            enc = TargetEncoder(smooth=smooth, random_state=seed)
            enc.fit(tr_part[feature_batch], tr_part[target_col])

            val_encoded = enc.transform(val_part[feature_batch])
            if not isinstance(val_encoded, pd.DataFrame):
                val_encoded = pd.DataFrame(
                    val_encoded, index=val_part.index, columns=feature_batch
                )

            val_encoded.columns = te_cols
            val_encoded.index = val_part.index

            oof_encoded_batch.iloc[val_idx] = val_encoded.values

        if ordinal:
            oof_encoded_batch = oof_encoded_batch.rank(method="dense").astype(int)

        train_out[te_cols] = oof_encoded_batch

        final_enc = TargetEncoder(smooth=smooth, random_state=seed)
        final_enc.fit(train_df[feature_batch], train_df[target_col])

        test_encoded = final_enc.transform(test_df[feature_batch])
        if not isinstance(test_encoded, pd.DataFrame):
            test_encoded = pd.DataFrame(
                test_encoded, index=test_df.index, columns=feature_batch
            )

        test_encoded.columns = te_cols
        test_encoded.index = test_df.index

        if ordinal:
            test_encoded = test_encoded.rank(method="dense").astype(int)

        test_out[te_cols] = test_encoded

    return train_out, test_out
