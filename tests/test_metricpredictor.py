import numpy as np

from src.metric_predictor import MetricPredictor


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
