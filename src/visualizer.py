# src/visualizer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union

from .metric_unit import MetricUnit


class Visualizer:
    """
    Provides Graph A–F visualizations from imputed DataFrames.

    Graph A: individual trajectories (plot_task_measure).
    Graph B: high/low fluctuation boxplot (plot_fluctuation_groups).
    Graph C: colored trajectories by fluctuation (plot_task_by_fluctuation_color).
    Graph D: trajectories with mean ±95% CI overlay (plot_mean_ci).
    Graph E: baseline arousal vs. task‐change scatter (plot_change_vs_arousal).
    Graph F: cross‐domain correlation heatmap (plot_cross_domain_correlation).
    """

    def __init__(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize with mapping of dataset names to DataFrames.

        Parameters
        ----------
        data : dict
            Keys are dataset names, values are pandas DataFrames,
            each containing an 'ID' column.
        """
        self.data = data
        self.id_col = "ID"

    def plot_task_measure(self, dataset: str, metric: str) -> None:
        """
        Graph A: Plot each participant’s trajectory over six sessions.

        Parameters
        ----------
        dataset : str
            Key in self.data for the task DataFrame.
        metric : str
            Column name of the measure to plot.
        """
        df = self.data[dataset].copy()
        df["_session"] = df.groupby(self.id_col).cumcount() + 1
        sessions = ["baseline"] + [f"home {i}" for i in range(1, 6)]

        unit = MetricUnit.get(metric)
        ylabel = f"{metric} ({unit})" if unit else metric

        fig, ax = plt.subplots(figsize=(8, 5))
        for pid, sub in df.groupby(self.id_col):
            ax.plot(
                sub["_session"],
                sub[metric],
                marker="o",
                linewidth=1.5,
                markersize=6,
                label=str(pid),
            )

        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(sessions, rotation=45, fontsize=10)
        ax.set_xlabel("Session", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{dataset}: {metric} Trajectories", fontsize=14)
        ax.legend(
            title=self.id_col,
            fontsize=8,
            title_fontsize=10,
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_fluctuation_groups(
        self,
        fluct_dataset: str,
        fluct_metrics: Union[str, List[str]],
        compare_dataset: str,
        compare_col: str,
    ) -> None:
        """
        Graph B: Boxplot of compare_col by high/low fluctuation group.

        Parameters
        ----------
        fluct_dataset : str
            Key in self.data for the DataFrame defining fluctuation.
        fluct_metrics : str or list of str
            One or two metric names to compute SD (or average SD).
        compare_dataset : str
            Key in self.data for the DataFrame with comparison variable.
        compare_col : str
            Column name in compare_dataset to boxplot.
        """
        df = self.data[fluct_dataset]
        comp = self.data[compare_dataset]

        if isinstance(fluct_metrics, str):
            stds = (
                df.groupby(self.id_col)[fluct_metrics]
                .std()
                .reset_index(name="std")
            )
            title_metrics = fluct_metrics
        else:
            m1, m2 = fluct_metrics
            s1 = df.groupby(self.id_col)[m1].std().rename("std1")
            s2 = df.groupby(self.id_col)[m2].std().rename("std2")
            merged = pd.concat([s1, s2], axis=1).reset_index()
            merged["std"] = (merged["std1"] + merged["std2"]) / 2
            stds = merged[[self.id_col, "std"]]
            title_metrics = f"{m1} & {m2}"

        median = stds["std"].median()
        stds["group"] = stds["std"].apply(lambda x: "High" if x > median else "Low")

        merged = stds.merge(
            comp[[self.id_col, compare_col]], on=self.id_col, how="left"
        )
        low = merged[merged["group"] == "Low"][compare_col].dropna()
        high = merged[merged["group"] == "High"][compare_col].dropna()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot([low, high], labels=["Low", "High"])
        ax.set_title(
            f"{compare_col} by avg-SD({title_metrics}) Fluctuation", fontsize=14
        )
        ax.set_ylabel(compare_col, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_task_by_fluctuation_color(self, dataset: str, metric: str) -> None:
        """
        Graph C: Plot trajectories colored by high/low fluctuation.

        Parameters
        ----------
        dataset : str
            Key in self.data for the task DataFrame.
        metric : str
            Column whose SD defines fluctuation groups.
        """
        df = self.data[dataset].copy()
        df["_session"] = df.groupby(self.id_col).cumcount() + 1

        stds = (
            df.groupby(self.id_col)[metric]
            .std()
            .reset_index(name="std")
        )
        med = stds["std"].median()
        stds["group"] = stds["std"].apply(lambda x: "High" if x > med else "Low")
        group_map = dict(zip(stds[self.id_col], stds["group"]))
        color_map = {"Low": "tab:blue", "High": "tab:red"}

        fig, ax = plt.subplots(figsize=(8, 5))
        for pid, sub in df.groupby(self.id_col):
            grp = group_map[pid]
            ax.plot(
                sub["_session"],
                sub[metric],
                marker="o",
                linewidth=1.5,
                markersize=6,
                color=color_map[grp],
                label=grp,
            )

        sessions = ["baseline"] + [f"home {i}" for i in range(1, 6)]
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(sessions, rotation=45, fontsize=10)
        ax.set_xlabel("Session", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{dataset}: {metric} by Fluctuation Group", fontsize=14)

        handles, labels = ax.get_legend_handles_labels()
        unique = {lbl: h for h, lbl in zip(handles, labels)}
        ax.legend(
            unique.values(),
            unique.keys(),
            title="Fluctuation",
            fontsize=10,
            title_fontsize=12,
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_mean_ci(self, dataset: str, metric: str) -> None:
        """
        Graph D: Individual trajectories with group mean ±95% CI overlay.

        Parameters
        ----------
        dataset : str
            Key in self.data for the task DataFrame.
        metric : str
            Column name of the measure to plot.
        """
        df = self.data[dataset].copy()
        df["_session"] = df.groupby(self.id_col).cumcount() + 1

        agg = (
            df.groupby("_session")[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])
        agg["ci95"] = 1.96 * agg["sem"]

        sessions = ["baseline"] + [f"home {i}" for i in range(1, 6)]

        fig, ax = plt.subplots(figsize=(8, 5))
        for _, sub in df.groupby(self.id_col):
            ax.plot(sub["_session"], sub[metric], color="lightgray", linewidth=1)
        ax.plot(agg["_session"], agg["mean"], color="black", linewidth=2, label="Mean")
        ax.fill_between(
            agg["_session"],
            agg["mean"] - agg["ci95"],
            agg["mean"] + agg["ci95"],
            color="black",
            alpha=0.2,
            label="95% CI",
        )

        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(sessions, rotation=45, fontsize=10)
        ax.set_xlabel("Session", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{dataset}: {metric} Mean ±95% CI", fontsize=14)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_change_vs_arousal(
        self,
        task_dataset: str,
        task_metric: str,
        physio_dataset: str,
        physio_metric: str,
    ) -> None:
        """
        Graph E: Scatter of baseline arousal vs. change in task metric.

        Parameters
        ----------
        task_dataset : str
            Key in self.data for the task DataFrame.
        task_metric : str
            Column name of the measure to compute change.
        physio_dataset : str
            Key in self.data for the physio baseline DataFrame.
        physio_metric : str
            Column name of the physio measure for arousal.
        """
        df = self.data[task_dataset].copy()
        df["_session"] = df.groupby(self.id_col).cumcount() + 1
        base = df[df["_session"] == 1].set_index(self.id_col)[task_metric]
        end = df[df["_session"] == 6].set_index(self.id_col)[task_metric]
        change = (end - base).rename("Δ" + task_metric)

        phys = self.data[physio_dataset].copy()
        arousal = phys.groupby(self.id_col)[physio_metric].mean()

        comp = pd.concat([change, arousal], axis=1).dropna()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(comp[physio_metric], comp["Δ" + task_metric], s=30)
        m, b = np.polyfit(comp[physio_metric], comp["Δ" + task_metric], 1)
        xs = np.array(ax.get_xlim())
        ax.plot(xs, m * xs + b, color="red", linestyle="--", linewidth=1)

        ax.set_xlabel(physio_metric, fontsize=12)
        ax.set_ylabel("Δ" + task_metric, fontsize=12)
        ax.set_title(f"{task_metric} Change vs {physio_metric}", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_cross_domain_correlation(self, columns_by_dataset: Dict[str, List[str]]) -> None:
        """
        Graph F: Correlation heatmap across specified columns from multiple datasets.

        Parameters
        ----------
        columns_by_dataset : dict
            Mapping of dataset_name → list of column names to include.
        """
        merged = None
        for name, cols in columns_by_dataset.items():
            df = self.data.get(name)
            if df is None or not cols:
                continue
            select = [self.id_col] + cols
            slice_df = df[select].copy()
            slice_df.columns = [
                self.id_col if c == self.id_col else f"{name}_{c}"
                for c in slice_df.columns
            ]
            merged = slice_df if merged is None else merged.merge(slice_df, on=self.id_col)

        data = merged.drop(columns=[self.id_col])
        corr = data.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax)
        labels = corr.columns.tolist()
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Cross-Domain Correlation", pad=20, fontsize=14)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        plt.show()
