import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from metric_predictor import MetricPredictor


def test_get_best_metric_returns_valid_method():
    import pandas as pd

    df = pd.DataFrame(
        {
            "ID": ["A"] * 6 + ["B"] * 6,
            "measurement": [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1],
        }
    )
    predictor = MetricPredictor(df)
    method = predictor.get_optimal_metric("measurement", idx_to_hide=2)
    assert method in ["interpolation", "mean", "median", "knn"]


def test_get_df_with_valid_participants_filters_correctly():
    import pandas as pd

    df = pd.DataFrame(
        {"ID": ["A"] * 4 + ["B"] * 6 + ["C"] * 3, "measurement": list(range(13))}
    )
    predictor = MetricPredictor(df)
    filtered = predictor.get_df_with_valid_participants(df, "measurement")
    assert set(filtered["ID"].unique()) == {"B"}


def test_interpolation_returns_float():
    import pandas as pd

    df = pd.DataFrame({"ID": ["A"] * 6, "measurement": [10, 20, 30, 40, 50, 60]})
    predictor = MetricPredictor(df)
    group = df[df["ID"] == "A"].reset_index(drop=True)
    result = predictor.interpolate(
        group_hidden=group.drop(3),
        measurement="measurement",
        group=group,
        idx_to_hide=3,
        kind="linear",
    )
    assert isinstance(result, float)


def test_get_best_metric_returns_none_on_empty_df():
    import pandas as pd

    df = pd.DataFrame({"ID": [], "measurement": []})
    predictor = MetricPredictor(df)
    method = predictor.get_optimal_metric("measurement", idx_to_hide=0)
    assert method is None


@pytest.fixture
def sample_df():
    # 3 participants, each with 4 time points
    data = {
        "ID": [1] * 4 + [2] * 4 + [3] * 4,
        "measurement1": [1, 2, 3, 4, 2, 3, 4, 5, 1, 2, 3, 4],
        "measurement2": [10, 11, 12, 13, 11, 12, 13, 14, 10, 11, 12, 13],
    }
    return pd.DataFrame(data)


def test_get_max_num(sample_df):
    predictor = MetricPredictor(sample_df)
    max_count = predictor.get_max_num("measurement1")
    assert max_count == 4


def test_get_max_num_with_missing():
    df = pd.DataFrame(
        {"ID": [1, 1, 2, 2, 2, 3, 3], "measurement1": [1, 2, 3, 4, None, 5, 6]}
    )
    predictor = MetricPredictor(df)
    assert predictor.get_max_num("measurement1") == 2


def test_get_optimal_metric_dict_for_measurement(sample_df):
    predictor = MetricPredictor(sample_df)
    result = predictor.get_optimal_metric_dict_for_measurement("measurement1")

    assert isinstance(result, dict)
    assert all(isinstance(k, int) for k in result.keys())
    assert all(isinstance(v, str) or v is None for v in result.values())
    assert len(result) == 4


def test_get_optimal_metric_dict(sample_df):
    predictor = MetricPredictor(sample_df)
    result = predictor.get_optimal_metric_dict(["measurement1", "measurement2"])

    assert isinstance(result, dict)
    assert set(result.keys()) == {"measurement1", "measurement2"}
    for val in result.values():
        assert isinstance(val, dict)
        assert all(isinstance(k, int) for k in val.keys())
        assert all(isinstance(v, str) or v is None for v in val.values())
