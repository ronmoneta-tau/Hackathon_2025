import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import zscore

from src.metrics_enum import Metrics

FEATURE_BREAKDOWNS = {
    "SSRT": ["Median Go RT", "Average SSD"],
    "Post Go Error Efficiency": ["Post Go Error Go % Accuracy", "Post Go Error Go RT"],
    "D’context": ["Z (AX CorrectRate)", "Z(BX IncorrectRate)"],
    "A-cue bias": ["Z (AX CorrectRate)", "Z(AY IncorrectRate)"],
    "PBI_composite": ["PBI_error", "PBI_rt"],
    "PBI_error": ["AY IncorrectRate adjusted", "BX IncorrectRate adjusted"],
    "PBI_rt": ["AY Correct Mean RT", "BX Correct Mean RT"],
}  # TODO: move to config?


class InterpolationStrategy(ABC):
    """
    Strategy interface for interpolation.
    """

    @abstractmethod
    def interpolate(
        self, x: np.ndarray, y: np.ndarray, new_x: np.ndarray
    ) -> np.ndarray:
        pass


class SciPyInterpolator(InterpolationStrategy):
    """
    Concrete interpolation strategy using SciPy's interp1d.
    """

    def __init__(self, kind: str = "linear"):
        self.kind = kind

    def interpolate(
        self, x: np.ndarray, y: np.ndarray, new_x: np.ndarray
    ) -> np.ndarray:
        """
        Interpolates y values at new_x based on x and y using the specified interpolation kind.
        :param x: Original x values (independent variable).
        :param y: Original y values (dependent variable).
        :param new_x: New x values where interpolation is to be performed.
        :return: Interpolated y values at new_x.
        """
        if len(x) < 2:
            # Not enough points to interpolate; return NaNs
            raise ValueError("Not enough points to interpolate")

        func = interp1d(x, y, kind=self.kind, fill_value="extrapolate")
        return func(new_x)


class QuartileClipper:
    """
    Applies clipping of values to within the IQR bounds per inputted feature.
    """

    def __init__(self, df: pd.DataFrame, quartile_feature: str = "TrialType"):
        self.df = df
        self.quartile_feature = quartile_feature
        self.quartiles = None

    def compute_quartiles(self, impute_feature: str) -> None:
        """
        Computes the IQR for the specified feature grouped by the quartile feature.
        :param impute_feature: The feature for which quartiles are computed.
        """
        self.quartiles = (
            self.df.groupby(self.quartile_feature)[impute_feature]
            .quantile([0.25, 0.75])
            .unstack()
        )

    def clip(
        self, df: pd.DataFrame, generated_values: np.ndarray, nan_masks: pd.Series
    ) -> np.ndarray:
        """
        Clips the generated values to the IQR bounds for each quartile group.
        :param df: DataFrame containing the quartile feature.
        :param generated_values: The values to be clipped.
        :param nan_masks: Boolean mask indicating where the original values were NaN.
        :return: Clipped values.
        """
        clipped_values = generated_values.copy()
        data_by_quartile_feature = df[self.quartile_feature]
        lower = data_by_quartile_feature.map(lambda t: self.quartiles.loc[t, 0.25])
        upper = data_by_quartile_feature.map(lambda t: self.quartiles.loc[t, 0.75])
        clipped_values[nan_masks] = np.clip(
            generated_values[nan_masks],
            lower.values[nan_masks],
            upper.values[nan_masks],
        )
        return clipped_values


class FeatureImputer:
    """
    Imputes a single feature using an interpolation strategy and optional censor.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature: str,
        method: str,
        interpolator_cls: Any = SciPyInterpolator,
    ):
        self.df = df
        self.feature = feature
        self.method = method
        self.interpolator_cls = interpolator_cls
        if self.method in Metrics._value2member_map_:
            self.generation_function = self.interpolation_generation
        elif self.method == "median":
            self.generation_function = self.median_generation
        elif self.method == "mean":
            self.generation_function = self.mean_generation
        else:
            raise ValueError(f"Unsupported imputation method: {self.method}")

    def mean_generation(
        self, idx: pd.Index, values: pd.Series, valid_mask: pd.Series
    ) -> np.ndarray:
        """
        Generates values by filling NaNs with the mean of the valid values.
        :param idx: Index of the DataFrame.
        :param values: Series of values for the feature.
        :param valid_mask: Boolean mask indicating valid (non-NaN) values.
        :return: Numpy array of values with NaNs filled with the mean.
        """
        mean_value = values.mean()
        return np.array(values.fillna(mean_value))

    def median_generation(
        self, idx: pd.Index, values: pd.Series, valid_mask: pd.Series
    ) -> np.ndarray:
        """
        Generates values by filling NaNs with the median of the valid values.
        :param idx: Index of the DataFrame.
        :param values: Series of values for the feature.
        :param valid_mask: Boolean mask indicating valid (non-NaN) values.
        :return: Numpy array of values with NaNs filled with the median.
        """
        median_value = values.median()
        return np.array(values.fillna(median_value))

    def interpolation_generation(
        self, idx: pd.Index, values: pd.Series, valid_mask: pd.Series
    ) -> np.ndarray:
        """
        Generates values by interpolating the valid values using the specified method.
        :param idx: Index of the DataFrame.
        :param values: Series of values for the feature.
        :param valid_mask: Boolean mask indicating valid (non-NaN) values.
        :return: Numpy array of interpolated values.
        """
        x = idx[valid_mask]
        y = values[valid_mask]
        interpolator = self.interpolator_cls(kind=self.method)
        return interpolator.interpolate(x, y, idx)

    def impute(self) -> pd.Series:
        """
        Imputes the feature by generating values and clipping them to the IQR bounds.
        :return: Series of imputed values for the feature.
        """
        quartile_clipper = QuartileClipper(self.df)
        quartile_clipper.compute_quartiles(self.feature)
        feature_series = self.df[self.feature].copy().astype(float)

        for _, group in self.df.groupby("ID"):
            idx = group.index
            values = group[self.feature]
            valid_mask = ~group[self.feature].isna()
            generated_values = self.generation_function(idx, values, valid_mask)
            clipped_values = quartile_clipper.clip(group, generated_values, ~valid_mask)
            feature_series.iloc[idx] = clipped_values

        return feature_series


class DataImputer:
    """
    Orchestrates imputation for multiple features using specified methods.

    Usage:
        imputer = DataImputer(df, {'FeatureA': 'linear', 'FeatureB': 'cubic'})
        filled_df = imputer.impute()
    """

    def __init__(self, df: pd.DataFrame, features_methods: Dict[str, str]):
        self.data = df.copy()
        self.features_methods = features_methods
        self.feature_breakdowns = FEATURE_BREAKDOWNS
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def impute(self, feature: str, method: str) -> pd.DataFrame:
        """
        Imputes a single feature using the specified method.
        :param feature: The feature to impute.
        :param method: The imputation method to use (e.g., 'linear', 'cubic', 'median', 'mean').
        :return: DataFrame with the imputed feature.
        """
        self.logger.info(f"Imputing feature '{feature}' using '{method}' imputation.")
        imputer = FeatureImputer(self.data, feature, method)
        self.data[feature] = imputer.impute()

        return self.data

    def impute_and_build(self) -> pd.DataFrame:
        """
        Imputes all features specified in the features_methods dictionary and builds composite features as needed.
        :return: DataFrame with all features imputed and composite features built.
        """
        for feature, method in self.features_methods.items():
            if feature not in self.data.columns:
                self.logger.warning(f"Feature '{feature}' not in DataFrame; skipping.")
                continue

            feature_blocks = (
                self.feature_breakdowns[feature]
                if feature in self.feature_breakdowns
                else [feature]
            )
            if "PBI_error" in feature_blocks:
                feature_blocks.extend(self.feature_breakdowns["PBI_error"])
            if "PBI_rt" in feature_blocks:
                feature_blocks.extend(self.feature_breakdowns["PBI_rt"])

            feature_blocks = set(
                [fb for fb in feature_blocks if fb != "PBI_error" and fb != "PBI_rt"]
            )

            for block in feature_blocks:
                self.data = self.impute(block, method)

            if len(feature_blocks) > 1:  # incase this is a composite feature
                self.logger.info(
                    f"Creating feature '{feature}' using the imputed features."
                )
                common_feature_name = "".join(
                    c if c.isalpha() else "_" for c in feature.lower()
                )
                builder = getattr(self, f"build_{common_feature_name}")
                self.data = builder()

        return self.data

    def build_ssrt(self) -> pd.DataFrame:
        """
        Builds the missing SSRT points from its components by the formula SSRT = Median go RT - Average SSD.
        :return: DataFrame with SSRT values filled in.
        """
        nan_mask = self.data["SSRT"].isna()
        self.data.loc[nan_mask, "SSRT"] = (
            self.data.loc[nan_mask, "Median Go RT"]
            - self.data.loc[nan_mask, "Average SSD"]
        )
        return self.data

    def build_post_go_error_efficiency(self) -> pd.DataFrame:
        """
        Builds the missing Post Go Error Efficiency points from its components by the formula SSRT = Post Go Error go % Accuracy / Post Go Error GO RT.
        :return: DataFrame with Post Go Error Efficiency values filled in.
        """
        nan_mask = self.data["Post Go Error Efficiency"].isna()
        self.data.loc[nan_mask, "Post Go Error Efficiency"] = (
            self.data.loc[nan_mask, "Post Go Error Go % Accuracy"]
            / self.data.loc[nan_mask, "Post Go Error Go RT"]
        )

        return self.data

    def build_d_context(self) -> pd.DataFrame:
        """
        Builds the missing D'context points from its components by the formula D'context = Z(AX CorrectRate) - Z(BX IncorrectRate)
        :return: DataFrame with D'context values filled in.
        """
        nan_mask = self.data["D’context"].isna()
        self.data.loc[nan_mask, "D’context"] = (
            self.data.loc[nan_mask, "Z (AX CorrectRate)"]
            / self.data.loc[nan_mask, "Z(BX IncorrectRate)"]
        )

        return self.data

    def build_a_cue_bias(self) -> pd.DataFrame:
        """
        Builds the missing A-cue bias points from its components by the formula A-cue bias = 1/2 * Z(Hit RateAX) + Z(False AlarmsAY)
        :return: DataFrame with A-cue bias values filled in.
        """
        nan_mask = self.data["A-cue bias"].isna()
        self.data.loc[nan_mask, "A-cue bias"] = 0.5 * (
            self.data.loc[nan_mask, "Z (AX CorrectRate)"]
            + self.data.loc[nan_mask, "Z(AY IncorrectRate)"]
        )

        return self.data

    def build_PBI_error(self) -> pd.DataFrame:
        """
        Builds the missing PBI_error points from its components by the formula PBI_error = (Error Rate AY - Error Rate BX) / (Error Rate AY + Error Rate BX)
        :return: DataFrame with PBI_error values filled in.
        """
        nan_mask = self.data["PBI_error"].isna()
        self.data.loc[nan_mask, "PBI_error"] = (
            self.data.loc[nan_mask, "AY IncorrectRate adjusted"]
            - self.data.loc[nan_mask, "BX IncorrectRate adjusted"]
        ) / (
            self.data.loc[nan_mask, "AY IncorrectRate adjusted"]
            + self.data.loc[nan_mask, "BX IncorrectRate adjusted"]
        )

        return self.data

    def build_PBI_rt(self) -> pd.DataFrame:
        """
        Builds the missing PBI_rt points from its components by the formula PBI_rt = (Mean RT AY - Mean RT BX) / (Mean RT AY + Mean RT BX)
        :return: DataFrame with PBI_rt values filled in.
        """
        nan_mask = self.data["PBI_rt"].isna()
        self.data.loc[nan_mask, "PBI_rt"] = (
            self.data.loc[nan_mask, "AY Correct Mean RT"]
            - self.data.loc[nan_mask, "BX Correct Mean RT"]
        ) / (
            self.data.loc[nan_mask, "AY Correct Mean RT"]
            + self.data.loc[nan_mask, "BX Correct Mean RT"]
        )

        return self.data

    def build_pbi_composite(self) -> pd.DataFrame:
        """
        Builds the missing PBI_composite points from its components by the formula PBI_composite = mean(z(PBI_error), z(PBI_rt))
        :return: DataFrame with PBI_composite values filled in.
        """
        self.data = self.build_PBI_error()
        self.data = self.build_PBI_rt()

        nan_mask = self.data["PBI_composite"].isna()
        self.data.loc[nan_mask, "PBI_composite"] = np.mean(
            [
                zscore(self.data.loc[nan_mask, "PBI_error"]),
                zscore(self.data.loc[nan_mask, "PBI_rt"]),
            ],
            axis=0,
        )

        return self.data
