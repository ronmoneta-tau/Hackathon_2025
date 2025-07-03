import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.metric_unit import MetricUnit
from src.visualizer import Visualizer


@pytest.fixture(autouse=True)
def disable_show(monkeypatch):
    """Prevent plt.show() from blocking tests."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


def make_task_df(n_participants=2):
    """
    Build a DataFrame with exactly 6 rows per participant and a single metric 'M'.
    Values ramp up so we can test lines and statistics.
    """
    rows = []
    for i in range(1, n_participants + 1):
        pid = f"P{i}"
        for sess in range(6):
            rows.append({"ID": pid, "M": 100 + 10 * i + sess})
    return pd.DataFrame(rows)


def test_metricunit_get():
    # Known metrics
    assert MetricUnit.get("SSRT") == "ms"
    assert MetricUnit.get("Post-Error Efficiency") == "%"
    # Unknown falls back to empty
    assert MetricUnit.get("UnknownMetric") == ""


def test_plot_task_measure():
    df = make_task_df(3)
    viz = Visualizer({"task": df})
    viz.plot_task_measure("task", metric="M")

    ax = plt.gca()
    # 3 participants → 3 lines
    assert len(ax.get_lines()) == 3
    # X tick labels
    expected = ["baseline", "home 1", "home 2", "home 3", "home 4", "home 5"]
    assert [t.get_text() for t in ax.get_xticklabels()] == expected
    # Y-label correct
    assert ax.get_ylabel() == "M"


def test_plot_fluctuation_groups_single():
    # P1 low var, P2 high var
    data = []
    for pid, vals in [("P1", [1] * 6), ("P2", [1, 2, 3, 4, 5, 6])]:
        for v in vals:
            data.append({"ID": pid, "X": v})
    df = pd.DataFrame(data)
    comp = pd.DataFrame({"ID": ["P1", "P2"], "S": [10, 20]})

    viz = Visualizer({"f": df, "c": comp})
    viz.plot_fluctuation_groups("f", "X", "c", "S")

    ax = plt.gca()
    # Two box artists
    assert len(getattr(ax, "artists", [])) == 2
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Low", "High"]
    assert ax.get_ylabel() == "S"


def test_plot_fluctuation_groups_two():
    # Two metrics A & B
    data = []
    for pid, a_vals, b_vals in [
        ("P1", [1] * 6, [2] * 6),
        ("P2", [1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]),
    ]:
        for a, b in zip(a_vals, b_vals):
            data.append({"ID": pid, "A": a, "B": b})
    df = pd.DataFrame(data)
    comp = pd.DataFrame({"ID": ["P1", "P2"], "Z": [5, 15]})

    viz = Visualizer({"f": df, "c": comp})
    viz.plot_fluctuation_groups("f", ["A", "B"], "c", "Z")

    ax = plt.gca()
    assert len(getattr(ax, "artists", [])) == 2
    assert ax.get_ylabel() == "Z"


def test_plot_task_by_fluctuation_color():
    df = make_task_df(2)
    # Force P1 constant (low), P2 varying (high)
    df.loc[df.ID == "P1", "M"] = 5
    viz = Visualizer({"t": df})
    viz.plot_task_by_fluctuation_color("t", "M")

    ax = plt.gca()
    lines = ax.get_lines()
    assert len(lines) == 2
    colors = {ln.get_color() for ln in lines}
    assert {"tab:blue", "tab:red"}.issubset(colors)


def test_plot_mean_ci():
    df = make_task_df(2)
    viz = Visualizer({"t": df})
    viz.plot_mean_ci("t", "M")

    ax = plt.gca()
    lines = ax.get_lines()
    # 2 participant lines + 1 mean line
    assert len(lines) == 3
    # CI fill_between creates a PolyCollection
    assert any(isinstance(c, plt.PolyCollection) for c in ax.collections)


def test_plot_change_vs_arousal():
    # Build task: baseline and home5 values
    rows = []
    for pid, base, end in [("P1", 5, 10), ("P2", 2, 4)]:
        rows += [{"ID": pid, "T": base}] + [{"ID": pid, "T": base}]*4 + [{"ID": pid, "T": end}]
    task_df = pd.DataFrame(rows)
    physio_df = pd.DataFrame({"ID": ["P1", "P2"], "HR": [50, 75]})

    viz = Visualizer({"task": task_df, "phys": physio_df})
    viz.plot_change_vs_arousal("task", "T", "phys", "HR")

    ax = plt.gca()
    # scatter produced as PathCollection
    assert any(isinstance(c, plt.PathCollection) for c in ax.collections)
    # trend line exists
    assert len(ax.get_lines()) >= 1


def test_plot_cross_domain_correlation():
    df1 = pd.DataFrame({"ID": ["P1", "P2"], "A": [1, 2]})
    df2 = pd.DataFrame({"ID": ["P1", "P2"], "B": [3, 4]})

    viz = Visualizer({"df1": df1, "df2": df2})
    viz.plot_cross_domain_correlation({"df1": ["A"], "df2": ["B"]})

    ax = plt.gca()
    # Heatmap image
    assert len(ax.get_images()) == 1
    # Tick labels include prefixed names
    xt = [t.get_text() for t in ax.get_xticklabels()]
    assert "df1_A" in xt and "df2_B" in xt
