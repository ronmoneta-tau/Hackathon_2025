import numpy as np
import pandas as pd
import pytest

from src.imputer import *
from src.metrics_enum import Metrics


def test_not_enough_points_returns_nan():
    value = list(Metrics)[0].value
    interp = SciPyInterpolator(kind=value)
    x = np.array([0])
    y = np.array([0])
    new_x = np.array([0, 1])

    with pytest.raises(ValueError):
        interp.interpolate(x, y, new_x)


def test_quartile_clipping():
    df = pd.DataFrame(
        {
            "TrialType": ["A", "A", "A", "B", "B", "B"],
            "value": [10, 20, 30, 100, 200, 300],
        }
    )
    clipper = QuartileClipper(df, quartile_feature="TrialType")
    clipper.compute_quartiles("value")

    test_df = df.copy()
    generated_values = np.array([5, 25, 35, 90, 250, 400])
    nan_mask = pd.Series([True] * 6)
    clipped = clipper.clip(test_df, generated_values, nan_mask)

    assert clipped[0] >= 10  # Q1 of A
    assert clipped[2] <= 30  # Q3 of A
    assert clipped[5] <= 300  # Q3 of B


def test_feature_imputer_with_linear():
    df = pd.DataFrame(
        {
            "ID": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "TrialType": ["A"] * 6,
            "value": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0],
        }
    )

    imputer = FeatureImputer(df, feature="value", method="linear")
    result = imputer.impute()
    assert not result.isna().any()
    assert len(result) == 6


def test_data_imputer_composite_feature():
    df = pd.DataFrame(
        {
            "ID": ["x", "x", "x"],
            "TrialType": ["T1", "T1", "T1"],
            "Median Go RT": [400.0, np.nan, 400.0],
            "Average SSD": [200.0, 200.0, 200.0],
            "SSRT": [500.0, np.nan, 400.0],
        }
    )

    imputer = DataImputer(df, {"SSRT": "linear"})
    result = imputer.impute_and_build()

    assert not result["SSRT"].isna().any()


def test_feature_imputer_invalid_method():
    df = pd.DataFrame({"ID": ["a"], "feature": [1.0]})
    # Unsupported method should raise ValueError
    with pytest.raises(ValueError):
        _ = FeatureImputer(df, feature="feature", method="unknown")


def test_mean_and_median_generation():
    # Create a series with a NaN
    values = pd.Series([1.0, np.nan, 3.0])
    idx = pd.Index([0, 1, 2])
    mask = ~values.isna()
    # Test mean_generation
    mean_arr = FeatureImputer.mean_generation(None, idx, values, mask)
    expected_mean = values.mean()
    # All positions filled with values or mean
    assert np.allclose(mean_arr, [1.0, expected_mean, 3.0], equal_nan=False)
    # Test median_generation
    median_arr = FeatureImputer.median_generation(None, idx, values, mask)
    expected_median = values.median()
    assert np.allclose(median_arr, [1.0, expected_median, 3.0], equal_nan=False)


def test_interpolation_generation_with_custom_interpolator():
    # Dummy interpolator that returns constant
    class DummyInterpolator:
        def __init__(self, kind):
            self.kind = kind

        def interpolate(self, x, y, new_x):
            return np.full(len(new_x), 42.0)

    df = pd.DataFrame({"ID": ["i", "i"], "feature": [1.0, 2.0]})
    fi = FeatureImputer(
        df, feature="feature", method="linear", interpolator_cls=DummyInterpolator
    )
    idx = pd.Index([0, 1])
    values = pd.Series([1.0, 2.0])
    mask = pd.Series([True, True])
    result = fi.interpolation_generation(idx, values, mask)
    assert np.array_equal(result, np.array([42.0, 42.0]))


def test_quartile_clipper_no_nan_mask():
    # Setup DataFrame
    df = pd.DataFrame({"TrialType": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
    clipper = QuartileClipper(df, quartile_feature="TrialType")
    clipper.compute_quartiles("value")
    # Generated values outside but no nan mask -> no clipping
    generated = np.array([0, 10, 0, 10])
    nan_mask = pd.Series([False, False, False, False])
    clipped = clipper.clip(df, generated, nan_mask)
    # Should equal original generated
    assert np.array_equal(clipped, generated)


def test_data_imputer_impute_with_mean():
    df = pd.DataFrame(
        {
            "ID": ["g1", "g1", "g2", "g2"],
            "TrialType": ["X", "X", "Y", "Y"],
            "val": [1.0, np.nan, 5.0, np.nan],
        }
    )
    imp = DataImputer(df, {"val": "mean"})
    out = imp.impute("val", "mean")
    # NaNs should be replaced by group-specific mean (only one valid -> same value)
    assert out["val"].iloc[1] == 1.0
    assert out["val"].iloc[3] == 5.0


def test_data_imputer_skip_missing_feature():
    # No error when feature not in DataFrame
    df = pd.DataFrame({"ID": ["a"], "X": [1]})
    imp = DataImputer(df, {"Y": "mean"})
    out = imp.impute_and_build()
    # Data unchanged
    assert "X" in out.columns and "Y" not in out.columns


def test_data_imputer_composite_post_go_error_efficiency():
    # Composite for Post Go Error Efficiency
    df = pd.DataFrame(
        {
            "ID": ["p", "p"],
            "TrialType": ["T", "T"],
            "Post Go Error Go % Accuracy": [80.0, 40.0],
            "Post Go Error Go RT": [20.0, 10.0],
            "Post Go Error Efficiency": [np.nan, np.nan],
        }
    )
    imp = DataImputer(df, {"Post Go Error Efficiency": "linear"})
    out = imp.impute_and_build()
    # efficiency = accuracy / RT
    assert out["Post Go Error Efficiency"].iloc[0] == 80.0 / 20.0
    assert out["Post Go Error Efficiency"].iloc[1] == 40.0 / 10.0


def test_data_imputer_composite_d_context_and_a_cue_bias():
    # Composite for D’context and A-cue bias
    key_d = "D’context"
    key_a = "A-cue bias"
    df = pd.DataFrame(
        {
            "ID": ["d", "d"],
            "TrialType": ["T", "T"],
            "Z (AX CorrectRate)": [2.0, 4.0],
            "Z(BX IncorrectRate)": [1.0, 2.0],
            "Z(AY IncorrectRate)": [3.0, 1.0],
            key_d: [np.nan, np.nan],
            key_a: [np.nan, np.nan],
        }
    )
    imp = DataImputer(df, {key_d: "linear", key_a: "linear"})
    out = imp.impute_and_build()
    # D’context = Z(AX)/Z(BX)
    assert out[key_d].tolist() == [2.0 / 1.0, 4.0 / 2.0]
    # A-cue bias = 0.5*(Z(AX)+Z(AY))
    assert out[key_a].tolist() == [0.5 * (2.0 + 3.0), 0.5 * (4.0 + 1.0)]


def test_data_imputer_composite_pbi_composite():
    # Composite for PBI_composite
    df = pd.DataFrame(
        {
            "ID": ["z", "z"],
            "TrialType": ["T", "T"],
            "AY IncorrectRate adjusted": [3.0, 1.0],
            "BX IncorrectRate adjusted": [1.0, 3.0],
            "AY Correct Mean RT": [3.0, 1.0],
            "BX Correct Mean RT": [1.0, 3.0],
            "PBI_error": [np.nan, np.nan],
            "PBI_rt": [np.nan, np.nan],
            "PBI_composite": [np.nan, np.nan],
        }
    )
    imp = DataImputer(df, {"PBI_composite": "linear"})
    out = imp.impute_and_build()
    # Compute expected PBI_error and PBI_rt
    expected_error = [(3 - 1) / (3 + 1), (1 - 3) / (1 + 3)]  # [0.5, -0.5]
    expected_rt = [(3 - 1) / (3 + 1), (1 - 3) / (1 + 3)]  # [0.5, -0.5]
    # After zscore, values become [1, -1]
    expected_composite = [1.0, -1.0]
    assert pytest.approx(out["PBI_error"].tolist()) == expected_error
    assert pytest.approx(out["PBI_rt"].tolist()) == expected_rt
    assert pytest.approx(out["PBI_composite"].tolist()) == expected_composite
