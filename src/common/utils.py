import pandas as pd


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

def count_missing_values(self, df: pd.DataFrame) -> dict:
    """
    Counts the number of missing values in each column of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        dict: A dictionary with column names as keys and the number of missing values as values.
    """
    return {col: df[col].isnull().sum() for col in df.columns}