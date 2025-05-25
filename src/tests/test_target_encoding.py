import pandas as pd
from src.common.preprocessing import RegressionTargetEncoder


def test_regression_target_encoder():
    X = pd.DataFrame({"A": [1, 1, 1, 2, 2, 2], "B": ["x", "y", "z", "x", "y", "z"]})
    y = pd.Series([1, 2, 4, 4, 5, 7])
    
    encoder = RegressionTargetEncoder(statistic="median", cv=2, random_state=42)
    X_encoded = encoder.fit_transform(X, y)
    assert X_encoded.shape == (6, 2)
    assert X_encoded.tolist() == [[4.0, 4.0], [4.0, 5.0], [1.5, 7.0], [7.0, 1.0], [7.0, 2.0], [4.5, 4.0]]