import os
import time
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from itertools import combinations


def add_k_tuple_columns(df, k=3, cols=None, sep="_"):
    """
    For every k-tuple of `cols` add a new column whose value is the
    row-wise concatenation (with `sep`) of those columns.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        cols (list): List of column names to use for k-tuple generation.
        sep (str): Separator to use for concatenation.

    Returns:
        pd.DataFrame: A new DataFrame (Pandas) with the k-tuple columns added.
        list[str]: List of new column names added.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    if cols is None:
        cols = list(df.columns)
    if len(cols) < k:
        raise ValueError(f"Not enough columns to form {k}-tuples")

    pl_df = pl.from_pandas(df)
    new_col_names = []

    # progress_bar = tqdm(total=len(list(combinations(cols, k))), desc=f"Adding {k}-tuple columns", unit="tuple")

    new_exprs = []
    for group in combinations(cols, k):
        colname = sep.join(group)
        exprs = [pl.col(c).cast(pl.Utf8) for c in group]
        new_exprs.append(pl.concat_str(exprs, separator=sep).alias(colname))
        new_col_names.append(colname)
        # progress_bar.update(1)

    pl_df = pl_df.with_columns(new_exprs)

    return pl_df.to_pandas(), new_col_names

def cast_columns(df: pd.DataFrame, columns: list[str], target_dtype) -> pd.DataFrame:
    """
    Casts the specified columns of a DataFrame to a target data type.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        columns (list): List of column names to cast.
        target_dtype: The target data type to cast the columns to.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns cast to the target data type.
    """
    df = df.copy()
    df[columns] = df[columns].astype(target_dtype)
    return df

def cast_columns_inplace(df: pd.DataFrame, columns: list[str], dtype) -> None:
    """
    Casts specified columns of a DataFrame in-place.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(dtype, copy=False)

def count_missing_values(self, df: pd.DataFrame) -> dict:
    """
    Counts the number of missing values in each column of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        dict: A dictionary with column names as keys and the number of missing values as values.
    """
    return {col: df[col].isnull().sum() for col in df.columns}

# ------------------------------------------------------------------------
# Modified from detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
class Timer:
    DEFAULT_TIME_FORMAT_DATE_TIME = "%Y/%m/%d %H:%M:%S"
    DEFAULT_TIME_FORMAT = ["%03dms", "%02ds", "%02dm", "%02dh"]

    def __init__(self):
        self.start = time.time() * 1000

    def get_current(self):
        return self.get_time_hhmmss(self.start)

    def reset(self):
        self.start = time.time() * 1000

    def get_time_since_start(self, format=None):
        return self.get_time_hhmmss(self.start, format)

    def unix_time_since_start(self, in_seconds=True):
        gap = time.time() * 1000 - self.start

        if in_seconds:
            gap = gap // 1000

        # Prevent 0 division errors
        if gap == 0:
            gap = 1
        return gap

    def get_time_hhmmss(self, start=None, end=None, gap=None, format=None):
        """
        Calculates time since `start` and formats as a string.
        """
        if start is None and gap is None:

            if format is None:
                format = self.DEFAULT_TIME_FORMAT_DATE_TIME

            return time.strftime(format)

        if end is None:
            end = time.time() * 1000
        if gap is None:
            gap = end - start

        s, ms = divmod(gap, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)

        if format is None:
            format = self.DEFAULT_TIME_FORMAT

        items = [ms, s, m, h]
        assert len(items) == len(format), "Format length should be same as items"

        time_str = ""
        for idx, item in enumerate(items):
            if item != 0:
                time_str = format[idx] % item + " " + time_str

        # Means no more time is left.
        if len(time_str) == 0:
            time_str = "0ms"

        return time_str.strip()
    
def profile(profiler, name):
    print(name + ": " + profiler.get_time_since_start())
    profiler.reset()