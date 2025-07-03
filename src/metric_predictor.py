import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from src.metrics_enum import Metrics

"""
Metric Prediction Module

This module provides functionality to evaluate and select the best data imputation or interpolation
method for missing measurements in participant-based datasets. It compares various interpolation
techniques, as well as mean and median predictions using RMSE.

Classes:
    MetricPredictor: Main class for evaluating imputation methods on a pandas DataFrame.
"""


class MetricPredictor:
    """
    Finds the metric that predicts with the highest accuracy for the specified measurement.

    Attributes:
        df (pd.DataFrame): A DataFrame containing participant IDs and measurements.
    """

    def __init__(self, df):
        """
        Initializes the MetricPredictor.

        Args:
            df (pd.DataFrame): DataFrame containing 'ID' and measurement columns.
        """
        self.df = df

    def get_optimal_metric_dict(self, measurements):
        """
        Computes the best imputation method for each measurement.

        Args:
            measurements (List[str]): List of column names to evaluate.

        Returns:
            dict: Mapping from measurement name to a dict of best methods by index.
        """

        metric_dict = {}
        for measurement in measurements:
            metric_dict[measurement] = self.get_optimal_metric_dict_for_measurement(
                measurement
            )
        return metric_dict

    def get_optimal_metric_dict_for_measurement(self, measurement):
        """
        Evaluates the best method for each position/index in a given measurement column.

        Args:
            measurement (str): The measurement column name.

        Returns:
            dict: Mapping from index to optimal method name.
        """

        metric_dict = {}
        max_num = self.get_max_num(measurement)
        indices = list(range(0, max_num))
        for idx in indices:
            metric_dict[idx] = self.get_optimal_metric(measurement, idx)

        return metric_dict

    def get_optimal_metric(self, measurement, idx_to_hide):
        """
        Determines the best metric for a specific index in a measurement column.

        Args:
            measurement (str): The column to evaluate.
            idx_to_hide (int): The index to simulate as missing.

        Returns:
            str: The name of the best method ('mean', 'median', interpolation kind, etc.)
        """

        df = self.df[["ID", measurement]].copy()

        # use only participants with full data
        df = self.get_df_with_valid_participants(df, measurement)

        actual_values = []
        preds_by_kind = {kind.value: [] for kind in Metrics}
        preds_mean = []
        preds_median = []

        for participant_id, group in df.groupby("ID"):
            group = group.reset_index(drop=True)
            group_hidden = group.drop(index=idx_to_hide)

            # true value
            actual = group.loc[idx_to_hide, measurement]

            # calculate values using each metric
            for kind in (k.value for k in Metrics):
                interpolated = self.interpolate(
                    group_hidden, measurement, group, idx_to_hide, kind
                )
                preds_by_kind[kind].append(interpolated)

            mean_pred, median_pred = self.get_pred_for_id(group_hidden, measurement)

            # append values to lists
            actual_values.append(actual)
            preds_mean.append(mean_pred)
            preds_median.append(median_pred)

        all_methods = {
            **preds_by_kind,
            "mean": preds_mean,
            "median": preds_median,
        }
        best_method = self.evaluate_method_accuracy(all_methods, actual_values)
        return best_method

    def get_pred_for_id(self, group, measurement):
        """
        Returns predictions from mean and median for a participant's measurement.

        Args:
            group (pd.DataFrame): Group with one value dropped (simulating missing).
            measurement (str): Column name.

        Returns:
            Tuple[float, float]: (mean, median) predictions.
        """

        mean_pred = group[measurement].mean()
        median_pred = group[measurement].median()

        return mean_pred, median_pred

    def get_max_num(self, measurement):
        """
        Finds the maximum number of non-null measurements for any participant.

        Args:
            measurement (str): The measurement column name.

        Returns:
            int: Maximum number of values found for any single participant.
        """

        df = self.df.copy()
        df = df.dropna(subset=[measurement])
        counts = df["ID"].value_counts()
        return counts.max()

    @staticmethod
    def get_df_with_valid_participants(df, measurement):
        """
        Filters the DataFrame to include only participants with the most measurements.

        Args:
            df (pd.DataFrame): The input DataFrame.
            measurement (str): Column to check for NaNs.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """

        df = df.dropna(subset=[measurement])
        counts = df["ID"].value_counts()
        max_count = counts.max()
        valid_ids = counts[counts >= max_count].index
        return df[df["ID"].isin(valid_ids)]

    @staticmethod
    def interpolate(group_hidden, measurement, group, idx_to_hide, kind):
        """
        Performs interpolation for the hidden value.

        Args:
            group_hidden (pd.DataFrame): DataFrame without the hidden value.
            measurement (str): Column name to interpolate.
            group (pd.DataFrame): Full group.
            idx_to_hide (int): Index of the hidden value.
            kind (str): Interpolation kind (e.g., 'linear', 'cubic').

        Returns:
            float: Interpolated value, or NaN on failure.
        """

        try:
            x = np.arange(len(group_hidden))
            y = group_hidden[measurement].values

            f_interp = interp1d(x, y, kind=kind, fill_value="extrapolate")

            # determine the position of the hidden index in full group
            test_position = list(group.index).index(idx_to_hide)
            return float(f_interp(test_position))
        except:
            return np.nan

    @staticmethod
    def evaluate_method_accuracy(all_methods, actual_values):
        """
        Computes RMSE for each method and returns the one with the lowest error.

        Args:
            all_methods (dict): Dictionary of method names to predicted lists.
            actual_values (List[float]): True values for comparison.

        Returns:
            str: Method name with lowest RMSE, or None if all failed.
        """

        scores = {}
        y_true = np.array(actual_values)

        for name, preds in all_methods.items():
            y_pred = np.array(preds)
            # ignore failed predictions
            valid_mask = ~np.isnan(y_pred)
            if valid_mask.sum() > 0:
                score = mean_squared_error(y_true[valid_mask], y_pred[valid_mask])
                scores[name] = score

        if not scores:
            return None

        # return the method with the lowest RMSE
        best_method = min(scores, key=scores.get)

        return best_method
