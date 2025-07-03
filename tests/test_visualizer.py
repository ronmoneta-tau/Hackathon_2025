# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pytest
# from matplotlib.collections import PathCollection, PolyCollection

# from src.visualizer import Visualizer


# @pytest.fixture(autouse=True)
# def disable_show(monkeypatch):
#     """Disable plt.show() for all tests."""
#     monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


# def make_task_df(n_participants=3, metric_name="M"):
#     """Build a toy longitudinal df for tests."""
#     rows = []
#     for i in range(1, n_participants + 1):
#         pid = f"P{i}"
#         for sess in range(6):
#             rows.append({"ID": pid, metric_name: 10 * i + sess})
#     return pd.DataFrame(rows)


# def test_plot_task_measure():
#     df = make_task_df(2, "X")
#     viz = Visualizer({"d": df})
#     viz.plot_task_measure("d", "X")
#     ax = plt.gca()
#     assert len(ax.get_lines()) == 2
#     expected = ["baseline", "home 1", "home 2", "home 3", "home 4", "home 5"]
#     assert [t.get_text() for t in ax.get_xticklabels()] == expected
#     assert "X" in ax.get_ylabel()
#     assert "Trajectories" in ax.get_title()


# def test_plot_task_measure_many_ids():
#     # Should create "Other subjects" legend if n > max_legend
#     df = make_task_df(14, "Z")
#     viz = Visualizer({"d": df})
#     viz.plot_task_measure("d", "Z", max_legend=10)
#     ax = plt.gca()
#     labels = [x.get_text() for x in ax.get_legend().get_texts()]
#     assert "Other subjects" in labels


# def test_plot_fluctuation_groups_single():
#     data = []
#     for pid, vals in [("P1", [1, 1, 1, 1, 1, 1]), ("P2", [1, 2, 3, 4, 5, 6])]:
#         for v in vals:
#             data.append({"ID": pid, "X": v})
#     df = pd.DataFrame(data)
#     comp = pd.DataFrame({"ID": ["P1", "P2"], "S": [10, 20]})
#     viz = Visualizer({"a": df, "b": comp})
#     ax = plt.gca()
#     result = viz.plot_fluctuation_groups("a", "X", "b", "S", ax=ax)
#     assert isinstance(result, dict)
#     assert "boxes" in result
#     assert len(result["boxes"]) == 2
#     assert [t.get_text() for t in ax.get_xticklabels()] == [
#         "Low fluctuation",
#         "High fluctuation",
#     ]
#     assert "S" in ax.get_ylabel()


# def test_plot_fluctuation_groups_two():
#     data = []
#     for pid, a_vals, b_vals in [
#         ("P1", [1] * 6, [2] * 6),
#         ("P2", [1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]),
#     ]:
#         for a, b in zip(a_vals, b_vals):
#             data.append({"ID": pid, "A": a, "B": b})
#     df = pd.DataFrame(data)
#     comp = pd.DataFrame({"ID": ["P1", "P2"], "Z": [7, 15]})
#     viz = Visualizer({"f": df, "c": comp})
#     ax = plt.gca()
#     result = viz.plot_fluctuation_groups("f", ["A", "B"], "c", "Z", ax=ax)
#     assert isinstance(result, dict)
#     assert len(result["boxes"]) == 2
#     assert ax.get_ylabel().startswith("Mean") or "Z" in ax.get_ylabel()


# def test_plot_task_by_fluctuation_color():
#     df = make_task_df(2, "Y")
#     df.loc[df.ID == "P1", "Y"] = 1  # Make groups distinct
#     viz = Visualizer({"t": df})
#     viz.plot_task_by_fluctuation_color("t", "Y")
#     ax = plt.gca()
#     lines = ax.get_lines()
#     assert len(lines) == 2
#     colors = {ln.get_color() for ln in lines}
#     assert {"tab:blue", "tab:red"} <= colors


# def test_plot_mean_ci():
#     df = make_task_df(2, "M")
#     viz = Visualizer({"k": df})
#     viz.plot_mean_ci("k", "M")
#     ax = plt.gca()
#     assert any(isinstance(c, PolyCollection) for c in ax.collections)
#     # Mean curve exists (black line)
#     assert any(line.get_color() in ["black", "#000000"] for line in ax.get_lines())


# def test_plot_change_vs_arousal():
#     # Build two "subjects" with defined delta and arousal
#     rows = []
#     for pid, base, end in [("P1", 4, 8), ("P2", 10, 5)]:
#         rows += (
#             [{"ID": pid, "T": base}]
#             + [{"ID": pid, "T": base}] * 4
#             + [{"ID": pid, "T": end}]
#         )
#     task_df = pd.DataFrame(rows)
#     physio_df = pd.DataFrame({"ID": ["P1", "P2"], "HR": [70, 100]})
#     viz = Visualizer({"task": task_df, "phys": physio_df})
#     viz.plot_change_vs_arousal("task", "T", "phys", "HR")
#     ax = plt.gca()
#     assert any(isinstance(c, PathCollection) for c in ax.collections)
#     assert ax.get_xlabel().startswith("HR")


# def test_plot_cross_domain_correlation():
#     df1 = pd.DataFrame({"ID": ["P1", "P2"], "A": [1.0, 2.0]})
#     df2 = pd.DataFrame({"ID": ["P1", "P2"], "B": [5.0, 3.0]})
#     viz = Visualizer({"d1": df1, "d2": df2})
#     viz.plot_cross_domain_correlation({"d1": ["A"], "d2": ["B"]})
#     ax = plt.gca()
#     assert len(ax.get_images()) == 1
#     labels = [t.get_text() for t in ax.get_xticklabels()]
#     assert "d1_A" in labels and "d2_B" in labels


# def test_handles_missing_or_empty_datasets_gracefully():
#     # No columns or empty dataframe
#     viz = Visualizer({})
#     # Should not error:
#     viz.plot_cross_domain_correlation({})
#     viz.plot_cross_domain_correlation({"foo": []})
#     # Should also not error with missing dataset
#     viz = Visualizer({"A": pd.DataFrame({"ID": ["P1"], "X": [1]})})
#     viz.plot_cross_domain_correlation({"A": ["X"], "Missing": ["Y"]})


# def test_plot_fluctuation_groups_no_ax_single():
#     # Coverage for when ax is not passed (ax=None), single metric
#     data = []
#     for pid, vals in [("A", [1, 2, 3, 4, 5, 6]), ("B", [6, 5, 4, 3, 2, 1])]:
#         for v in vals:
#             data.append({"ID": pid, "M": v})
#     df = pd.DataFrame(data)
#     comp = pd.DataFrame({"ID": ["A", "B"], "Y": [2, 4]})
#     viz = Visualizer({"fa": df, "fb": comp})
#     # Don't pass ax param
#     result = viz.plot_fluctuation_groups("fa", "M", "fb", "Y")
#     assert isinstance(result, dict)
#     assert len(result["boxes"]) == 2


# def test_plot_fluctuation_groups_no_ax_multi():
#     # Coverage for when ax is not passed (ax=None), two metrics
#     data = []
#     for pid, m1, m2 in [
#         ("A", [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]),
#         ("B", [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6]),
#     ]:
#         for a, b in zip(m1, m2):
#             data.append({"ID": pid, "A": a, "B": b})
#     df = pd.DataFrame(data)
#     comp = pd.DataFrame({"ID": ["A", "B"], "Z": [7, 15]})
#     viz = Visualizer({"fa": df, "fb": comp})
#     # Don't pass ax param
#     result = viz.plot_fluctuation_groups("fa", ["A", "B"], "fb", "Z")
#     assert isinstance(result, dict)
#     assert len(result["boxes"]) == 2

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.collections import PathCollection, PolyCollection

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from visualizer import Visualizer


@pytest.fixture(autouse=True)
def disable_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


def make_task_df(n_participants=3, metric_name="M"):
    rows = []
    for i in range(1, n_participants + 1):
        pid = f"P{i}"
        for sess in range(6):
            rows.append({"ID": pid, metric_name: 10 * i + sess})
    return pd.DataFrame(rows)


def test_plot_task_measure():
    df = make_task_df(2, "X")
    viz = Visualizer({"d": df})
    viz.plot_task_measure("d", "X")
    ax = plt.gca()
    assert len(ax.get_lines()) == 2
    expected = ["baseline", "home 1", "home 2", "home 3", "home 4", "home 5"]
    assert [t.get_text() for t in ax.get_xticklabels()] == expected
    assert "X" in ax.get_ylabel()
    assert "Trajectories" in ax.get_title()


def test_plot_task_measure_many_ids():
    df = make_task_df(14, "Z")
    viz = Visualizer({"d": df})
    viz.plot_task_measure("d", "Z", max_legend=10)
    ax = plt.gca()
    labels = [x.get_text() for x in ax.get_legend().get_texts()]
    assert "Other subjects" in labels


def test_plot_fluctuation_groups_single():
    data = []
    for pid, vals in [("P1", [1, 1, 1, 1, 1, 1]), ("P2", [1, 2, 3, 4, 5, 6])]:
        for v in vals:
            data.append({"ID": pid, "X": v})
    df = pd.DataFrame(data)
    comp = pd.DataFrame({"ID": ["P1", "P2"], "S": [10, 20]})
    viz = Visualizer({"a": df, "b": comp})
    ax = plt.gca()
    result = viz.plot_fluctuation_groups("a", "X", "b", "S", ax=ax)
    assert isinstance(result, dict)
    assert "boxes" in result
    assert len(result["boxes"]) == 2
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Low fluctuation",
        "High fluctuation",
    ]
    assert "S" in ax.get_ylabel()


def test_plot_fluctuation_groups_two():
    data = []
    for pid, a_vals, b_vals in [
        ("P1", [1] * 6, [2] * 6),
        ("P2", [1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]),
    ]:
        for a, b in zip(a_vals, b_vals):
            data.append({"ID": pid, "A": a, "B": b})
    df = pd.DataFrame(data)
    comp = pd.DataFrame({"ID": ["P1", "P2"], "Z": [7, 15]})
    viz = Visualizer({"f": df, "c": comp})
    ax = plt.gca()
    result = viz.plot_fluctuation_groups("f", ["A", "B"], "c", "Z", ax=ax)
    assert isinstance(result, dict)
    assert len(result["boxes"]) == 2
    assert ax.get_ylabel().startswith("Mean") or "Z" in ax.get_ylabel()


def test_plot_task_by_fluctuation_color():
    df = make_task_df(2, "Y")
    df.loc[df.ID == "P1", "Y"] = 1
    viz = Visualizer({"t": df})
    viz.plot_task_by_fluctuation_color("t", "Y")
    ax = plt.gca()
    lines = ax.get_lines()
    assert len(lines) == 2
    colors = {ln.get_color() for ln in lines}
    assert {"tab:blue", "tab:red"} <= colors


def test_plot_mean_ci():
    df = make_task_df(2, "M")
    viz = Visualizer({"k": df})
    viz.plot_mean_ci("k", "M")
    ax = plt.gca()
    assert any(isinstance(c, PolyCollection) for c in ax.collections)
    assert any(line.get_color() in ["black", "#000000"] for line in ax.get_lines())


def test_plot_change_vs_mean():
    # Build two "subjects" with defined delta and mean value
    rows = []
    for pid, base, end in [("P1", 4, 8), ("P2", 10, 5)]:
        rows += (
            [{"ID": pid, "T": base}]
            + [{"ID": pid, "T": base}] * 4
            + [{"ID": pid, "T": end}]
        )
    change_df = pd.DataFrame(rows)
    mean_df = pd.DataFrame({"ID": ["P1", "P2"], "Arousal": [70, 100]})
    viz = Visualizer({"change": change_df, "mean": mean_df})
    viz.plot_change_vs_mean("change", "T", "mean", "Arousal")
    ax = plt.gca()
    assert any(isinstance(c, PathCollection) for c in ax.collections)
    assert ax.get_xlabel().startswith("Arousal")
    assert "ΔT" in ax.get_ylabel()


def test_plot_cross_domain_correlation():
    df1 = pd.DataFrame({"ID": ["P1", "P2"], "A": [1.0, 2.0]})
    df2 = pd.DataFrame({"ID": ["P1", "P2"], "B": [5.0, 3.0]})
    viz = Visualizer({"d1": df1, "d2": df2})
    viz.plot_cross_domain_correlation({"d1": ["A"], "d2": ["B"]})
    ax = plt.gca()
    assert len(ax.get_images()) == 1
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "d1_A" in labels and "d2_B" in labels


def test_handles_missing_or_empty_datasets_gracefully():
    viz = Visualizer({})
    viz.plot_cross_domain_correlation({})
    viz.plot_cross_domain_correlation({"foo": []})
    viz = Visualizer({"A": pd.DataFrame({"ID": ["P1"], "X": [1]})})
    viz.plot_cross_domain_correlation({"A": ["X"], "Missing": ["Y"]})


def test_plot_fluctuation_groups_no_ax_single():
    data = []
    for pid, vals in [("A", [1, 2, 3, 4, 5, 6]), ("B", [6, 5, 4, 3, 2, 1])]:
        for v in vals:
            data.append({"ID": pid, "M": v})
    df = pd.DataFrame(data)
    comp = pd.DataFrame({"ID": ["A", "B"], "Y": [2, 4]})
    viz = Visualizer({"fa": df, "fb": comp})
    result = viz.plot_fluctuation_groups("fa", "M", "fb", "Y")
    assert isinstance(result, dict)
    assert len(result["boxes"]) == 2


def test_plot_fluctuation_groups_no_ax_multi():
    data = []
    for pid, m1, m2 in [
        ("A", [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]),
        ("B", [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6]),
    ]:
        for a, b in zip(m1, m2):
            data.append({"ID": pid, "A": a, "B": b})
    df = pd.DataFrame(data)
    comp = pd.DataFrame({"ID": ["A", "B"], "Z": [7, 15]})
    viz = Visualizer({"fa": df, "fb": comp})
    result = viz.plot_fluctuation_groups("fa", ["A", "B"], "fb", "Z")
    assert isinstance(result, dict)
    assert len(result["boxes"]) == 2
