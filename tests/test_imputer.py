
import numpy as np
import pandas as pd
from src.imputer import SciPyInterpolator, QuartileClipper, FeatureImputer, DataImputer
import pytest
from src.mertices_enum import Metrics


def test_not_enough_points_returns_nan():
    value = list(Metrics)[0].value
    interp = SciPyInterpolator(kind=value)
    x = np.array([0])
    y = np.array([0])
    new_x = np.array([0, 1])

    with pytest.raises(ValueError):
        interp.interpolate(x, y, new_x)


def test_quartile_clipping():
    df = pd.DataFrame({
        'TrialType': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [10, 20, 30, 100, 200, 300]
    })
    clipper = QuartileClipper(df, quartile_feature='TrialType')
    clipper.compute_quartiles('value')

    test_df = df.copy()
    generated_values = np.array([5, 25, 35, 90, 250, 400])
    nan_mask = pd.Series([True] * 6)
    clipped = clipper.clip(test_df, generated_values, nan_mask)

    assert clipped[0] >= 10  # Q1 of A
    assert clipped[2] <= 30  # Q3 of A
    assert clipped[5] <= 300  # Q3 of B


def test_feature_imputer_with_linear():
    df = pd.DataFrame({
        'ID': ['s1', 's1', 's1', 's2', 's2', 's2'],
        'TrialType': ['A'] * 6,
        'value': [1.0, np.nan, 3.0, 4.0, np.nan, 6.0]
    })

    imputer = FeatureImputer(df, feature='value', method='linear')
    result = imputer.impute()
    assert not result.isna().any()
    assert len(result) == 6


def test_data_imputer_composite_feature():
    df = pd.DataFrame({
        'ID': ['x', 'x', 'x'],
        'TrialType': ['T1', 'T1', 'T1'],
        'Median Go RT': [400.0, np.nan, 400.0],
        'Average SSD': [200.0, 200.0, 200.0],
        'SSRT': [500.0, np.nan, 400.0],
    })

    imputer = DataImputer(df, {'SSRT': 'linear'})
    result = imputer.impute_and_build()

    assert not result['SSRT'].isna().any()
