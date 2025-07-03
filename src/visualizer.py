from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metric_unit import MetricUnit


class Visualizer:
    """
    Visualization tools for longitudinal cognitive/physiological data analysis.
    Produces publication-ready figures for individual and group trajectories,
    fluctuation grouping, and cross-domain correlations.
    """

    def __init__(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize the Visualizer.

        Parameters
        ----------
        data : dict
            Mapping of dataset names to DataFrames,
            each containing an 'ID' column.
        """
        self.data = data
        self.id_col = "ID"

    def _get_unit(self, metric: str) -> str:
        """Return a string for axis label, combining metric name and unit (if defined)."""
        unit = MetricUnit.get(metric)
        return f"{metric} ({unit})" if unit else metric

    def plot_task_measure(
        self, dataset: str, metric: str, max_legend: int = 12
    ) -> None:
        """
        Plot all individual trajectories for a measure over six sessions.
        If there are too many subjects, display a legend only for a subsample.

        Parameters
        ----------
        dataset : str
            Name of the dataset in self.data.
        metric : str
            Column to plot.
        max_legend : int
            Maximal number of subjects to show individually in the legend.
            The rest will appear as 'Other subjects'.
        """
        df = self.data[dataset].copy()
        df["_session"] = df.groupby(self.id_col).cumcount() + 1
        sessions = ["baseline"] + [f"home {i}" for i in range(1, 6)]
        ylabel = self._get_unit(metric)
        ids = df[self.id_col].unique()
        n_ids = len(ids)

        # Assign special colors to a subsample, gray to the rest
        if n_ids > max_legend:
            np.random.seed(0)
            sample_ids = np.random.choice(ids, size=max_legend, replace=False)
            color_map = {pid: f"C{i}" for i, pid in enumerate(sample_ids)}
        else:
            color_map = {pid: f"C{i}" for i, pid in enumerate(ids)}
            sample_ids = ids

        fig, ax = plt.subplots(figsize=(8, 5))
        for pid, sub in df.groupby(self.id_col):
            color = color_map.get(pid, "lightgray")
            label = str(pid) if pid in sample_ids else "Other subjects"
            lw = 2 if pid in sample_ids else 1
            zorder = 2 if pid in sample_ids else 1
            ax.plot(
                sub["_session"],
                sub[metric],
                marker="o",
                linewidth=lw,
                markersize=4 if pid in sample_ids else 2,
                color=color,
                alpha=1 if pid in sample_ids else 0.6,
                label=label,
                zorder=zorder,
            )
        # Add one dummy line for 'Other subjects' if needed
        if n_ids > max_legend:
            ax.plot([], [], color="lightgray", linewidth=1, label="Other subjects")

        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(sessions, rotation=45, fontsize=10)
        ax.set_xlabel("Session", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{dataset}: {ylabel} Trajectories", fontsize=14)
        # Only unique labels in legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(
            unique.values(),
            unique.keys(),
            title="Participant ID",
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
        ax: Optional[plt.Axes] = None,
    ) -> dict:
        """
        Plot a boxplot of the average of a comparison variable,
        grouped by high/low fluctuation in a task measure.

        Parameters
        ----------
        fluct_dataset : str
            Dataset for fluctuation grouping.
        fluct_metrics : str or list of str
            Name(s) of column(s) for SD calculation.
        compare_dataset : str
            Dataset with the comparison variable.
        compare_col : str
            Name of the comparison variable.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.

        Returns
        -------
        dict
            Result dictionary from ax.boxplot, for testability.
        """
        df = self.data[fluct_dataset]
        comp = self.data[compare_dataset]

        # Compute SD per participant (single or average of two metrics)
        if isinstance(fluct_metrics, str):
            stds = (
                df.groupby(self.id_col)[fluct_metrics].std().reset_index(name="std")
            ).copy()
            title_metrics = self._get_unit(fluct_metrics)
        else:
            m1, m2 = fluct_metrics
            s1 = df.groupby(self.id_col)[m1].std().rename("std1")
            s2 = df.groupby(self.id_col)[m2].std().rename("std2")
            merged = pd.concat([s1, s2], axis=1).reset_index()
            merged["std"] = (merged["std1"] + merged["std2"]) / 2
            stds = merged[[self.id_col, "std"]].copy()
            title_metrics = " & ".join([self._get_unit(m1), self._get_unit(m2)])

        median = stds["std"].median()
        stds["group"] = stds["std"].apply(lambda x: "High" if x > median else "Low")

        # Use mean over all available sessions of the comparison variable per participant
        mean_comp = comp.groupby(self.id_col)[compare_col].mean().reset_index()
        merged = stds.merge(mean_comp, on=self.id_col, how="left")
        low = merged[merged["group"] == "Low"][compare_col].dropna()
        high = merged[merged["group"] == "High"][compare_col].dropna()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        result = ax.boxplot([low, high], patch_artist=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Low fluctuation", "High fluctuation"])
        ylabel = self._get_unit(compare_col)
        ax.set_title(
            f"Mean {ylabel} by Fluctuation Group\n({title_metrics})", fontsize=14
        )
        ax.set_ylabel(f"Mean {ylabel}", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        if ax is None:
            plt.show()
        return result

    def plot_task_by_fluctuation_color(self, dataset: str, metric: str) -> None:
        """
        Plot individual trajectories colored by fluctuation group (high/low SD).

        Parameters
        ----------
        dataset : str
            Name of the dataset.
        metric : str
            Column for fluctuation grouping and plotting.
        """
        df = self.data[dataset].copy()
        df["_session"] = df.groupby(self.id_col).cumcount() + 1
        stds = df.groupby(self.id_col)[metric].std().reset_index(name="std")
        med = stds["std"].median()
        stds["group"] = stds["std"].apply(lambda x: "High" if x > med else "Low")
        group_map = dict(zip(stds[self.id_col], stds["group"]))
        color_map = {"Low": "tab:blue", "High": "tab:red"}
        ylabel = self._get_unit(metric)

        fig, ax = plt.subplots(figsize=(8, 5))
        for pid, sub in df.groupby(self.id_col):
            grp = group_map[pid]
            ax.plot(
                sub["_session"],
                sub[metric],
                marker="o",
                linewidth=1.5,
                markersize=4,
                color=color_map[grp],
                alpha=0.8,
            )
        sessions = ["baseline"] + [f"home {i}" for i in range(1, 6)]
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(sessions, rotation=45, fontsize=10)
        ax.set_xlabel("Session", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{dataset}: {ylabel} by Fluctuation Group", fontsize=14)
        # Add color legend only (not all participants)
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="tab:blue", lw=2, label="Low fluctuation"),
            Line2D([0], [0], color="tab:red", lw=2, label="High fluctuation"),
        ]
        ax.legend(
            handles=legend_elements,
            title="Fluctuation",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_mean_ci(self, dataset: str, metric: str) -> None:
        """
        Plot individual trajectories in gray, and the group mean with 95% CI overlay.

        Parameters
        ----------
        dataset : str
            Name of the dataset.
        metric : str
            Column to plot.
        """
        df = self.data[dataset].copy()
        df["_session"] = df.groupby(self.id_col).cumcount() + 1

        agg = df.groupby("_session")[metric].agg(["mean", "std", "count"]).reset_index()
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])
        agg["ci95"] = 1.96 * agg["sem"]
        sessions = ["baseline"] + [f"home {i}" for i in range(1, 6)]
        ylabel = self._get_unit(metric)

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
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{dataset}: {ylabel} Mean ±95% CI", fontsize=14)
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
        Scatter plot: change in task metric (session 6 - session 1) vs. baseline arousal.

        Parameters
        ----------
        task_dataset : str
            Task dataset name.
        task_metric : str
            Column for delta calculation.
        physio_dataset : str
            Physiological dataset name.
        physio_metric : str
            Column for arousal (mean per participant).
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
        pearson_r = comp[physio_metric].corr(comp["Δ" + task_metric])
        m, b = np.polyfit(comp[physio_metric], comp["Δ" + task_metric], 1)
        xs = np.array(ax.get_xlim())
        ax.plot(xs, m * xs + b, color="red", linestyle="--", linewidth=1)

        ax.set_xlabel(self._get_unit(physio_metric), fontsize=12)
        ax.set_ylabel(f"Δ{self._get_unit(task_metric)}", fontsize=12)
        ax.set_title(
            f"{task_metric} Change vs {physio_metric}\nPearson r = {pearson_r:.2f}",
            fontsize=14,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_cross_domain_correlation(
        self, columns_by_dataset: Dict[str, List[str]]
    ) -> None:
        """
        Plot a correlation heatmap across selected variables from different datasets.
        Each cell shows the Pearson r value.

        Parameters
        ----------
        columns_by_dataset : dict
            Mapping: dataset_name → list of column names to include.
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
            merged = (
                slice_df if merged is None else merged.merge(slice_df, on=self.id_col)
            )
        # --- Robustness: Handle missing or empty data
        if merged is None or merged.empty or len(merged.columns) <= 1:
            print("No data to plot correlation heatmap.")
            return
        data = merged.drop(columns=[self.id_col])
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(max(8, len(corr.columns)), 6))
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax)
        labels = corr.columns.tolist()
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Cross-Domain Correlation (Pearson r)", pad=20, fontsize=14)
        # Print Pearson r in each cell
        for (i, j), val in np.ndenumerate(corr.values):
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=7
            )
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        plt.show()
