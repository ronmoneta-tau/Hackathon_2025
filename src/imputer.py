import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import zscore

FEATURE_BREAKDOWNS = {"SSRT": ['Median Go RT', 'Average SSD'],
                      'Post Go Error Efficiency': ['Post Go Error Go % Accuracy', 'Post Go Error Go RT'],
                      "D’context": ["Z (AX CorrectRate)", "Z(BX IncorrectRate)"],
                      "A-cue bias": ["Z (AX CorrectRate)", "Z(AY IncorrectRate)"],
                      "PBI_composite": ["PBI_error", "PBI_rt"],
                      "PBI_error": ["AY IncorrectRate adjusted", "BX IncorrectRate adjusted"],
                      "PBI_rt": ["AY Correct Mean RT", "BX Correct Mean RT"], }


class InterpolationStrategy(ABC):
    """
    Strategy interface for interpolation.
    """

    kinds = {'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous',
             'next'}  # TODO: move to const?

    @abstractmethod
    def interpolate(self, x: np.ndarray, y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
        pass


class SciPyInterpolator(InterpolationStrategy):
    """
    Concrete interpolation strategy using SciPy's interp1d.
    """

    def __init__(self, kind: str = 'linear'):
        self.kind = kind

    def interpolate(self, x: np.ndarray, y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
        if len(x) < 2:
            # Not enough points to interpolate; return NaNs
            return np.full_like(new_x, np.nan, dtype=float)

        func = interp1d(x, y, kind=self.kind, fill_value='extrapolate')
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
        self.quartiles = self.df.groupby(self.quartile_feature)[impute_feature].quantile([0.25, 0.75]).unstack()

    def clip(self, df: pd.DataFrame, generated_values: np.ndarray, nan_masks: pd.Series) -> np.ndarray:
        clipped_values = generated_values.copy()
        data_by_quartile_feature = df[self.quartile_feature]
        lower = data_by_quartile_feature.map(lambda t: self.quartiles.loc[t, 0.25])
        upper = data_by_quartile_feature.map(lambda t: self.quartiles.loc[t, 0.75])
        clipped_values[nan_masks] = np.clip(generated_values[nan_masks], lower.values[nan_masks],
                                            upper.values[nan_masks])
        return clipped_values


class FeatureImputer:
    """
    Imputes a single feature using an interpolation strategy and optional censor.
    """

    def __init__(self, df: pd.DataFrame, feature: str, method: str, interpolator_cls: Any = SciPyInterpolator):
        self.df = df
        self.feature = feature
        self.method = method
        self.interpolator_cls = interpolator_cls
        if self.method in InterpolationStrategy.kinds:
            self.generation_function = self.interpolation_generation
        elif self.method == 'median':
            self.generation_function = self.median_generation
        else:
            raise ValueError(f"Unsupported imputation method: {self.method}")

    def median_generation(self, idx: pd.Index, values: pd.Series, valid_mask: pd.Series) -> np.ndarray:
        median_value = values.median()
        return np.array(values.fillna(median_value))

    def interpolation_generation(self, idx: pd.Index, values: pd.Series, valid_mask: pd.Series) -> np.ndarray:
        x = idx[valid_mask]
        y = values[valid_mask]
        interpolator = self.interpolator_cls(kind=self.method)
        return interpolator.interpolate(x, y, idx)

    def impute(self) -> pd.Series:
        quartile_clipper = QuartileClipper(self.df)
        quartile_clipper.compute_quartiles(self.feature)
        feature_series = self.df[self.feature].copy().astype(float)

        for _, group in self.df.groupby('SDAN'):  # TODO: change to new ID name
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
        self.logger.info(f"Imputing feature '{feature}' using '{method}' interpolation.")
        imputer = FeatureImputer(self.data, feature, method)
        self.data[feature] = imputer.impute()

        return self.data

    def impute_and_build(self):
        for feature, method in self.features_methods.items():
            if feature not in self.data.columns:
                self.logger.warning(f"Feature '{feature}' not in DataFrame; skipping.")
                continue

            feature_blocks = self.feature_breakdowns[feature] if feature in self.feature_breakdowns else [feature]
            if "PBI_error" in feature_blocks:
                feature_blocks.extend(self.feature_breakdowns["PBI_error"])
            if "PBI_rt" in feature_blocks:
                feature_blocks.extend(self.feature_breakdowns["PBI_rt"])

            feature_blocks = set([fb for fb in feature_blocks if fb != "PBI_error" and fb != "PBI_rt"])

            for block in feature_blocks:
                self.data = self.impute(block, method)

            if len(feature_blocks) > 1:  # incase this is a composite feature
                self.logger.info(f"Creating feature '{feature}' using the imputed features.")
                common_feature_name = ''.join(c if c.isalpha() else '_' for c in feature.lower())
                builder = getattr(self, f'build_{common_feature_name}')
                self.data = builder()

        return self.data

    def build_ssrt(self):
        """
        Builds the missing SSRT points from its components by the formula SSRT = Median go RT - Average SSD.
        """
        nan_mask = self.data['SSRT'].isna()
        self.data.loc[nan_mask, 'SSRT'] = self.data.loc[nan_mask, 'Median Go RT'] - self.data.loc[
            nan_mask, 'Average SSD']
        return self.data

    def build_post_go_error_efficiency(self):
        """
        Builds the missing Post Go Error Efficiency points from its components by the formula SSRT = Post Go Error go % Accuracy / Post Go Error GO RT.
        """
        nan_mask = self.data['Post Go Error Efficiency'].isna()
        self.data.loc[nan_mask, 'Post Go Error Efficiency'] = self.data.loc[nan_mask, 'Post Go Error Go % Accuracy'] / \
                                                              self.data.loc[
                                                                  nan_mask, 'Post Go Error Go RT']

        return self.data

    def build_d_context(self):
        """
        Builds the missing D'context points from its components by the formula D'context = Z(AX CorrectRate) - Z(BX IncorrectRate)
        """
        nan_mask = self.data["D’context"].isna()
        self.data.loc[nan_mask, "D’context"] = self.data.loc[nan_mask, "Z (AX CorrectRate)"] / self.data.loc[
            nan_mask, "Z(BX IncorrectRate)"]

        return self.data

    def build_a_cue_bias(self):
        """
        Builds the missing A-cue bias points from its components by the formula A-cue bias = 1/2 * Z(Hit RateAX) + Z(False AlarmsAY)
        """
        nan_mask = self.data["A-cue bias"].isna()
        self.data.loc[nan_mask, "A-cue bias"] = 0.5 * (self.data.loc[nan_mask, "Z (AX CorrectRate)"] + self.data.loc[
            nan_mask, "Z(AY IncorrectRate)"])

        return self.data

    def build_PBI_error(self):
        """
        Builds the missing PBI_error points from its components by the formula PBI_error = (Error Rate AY - Error Rate BX) / (Error Rate AY + Error Rate BX)
        """
        nan_mask = self.data["PBI_error"].isna()
        self.data.loc[nan_mask, "PBI_error"] = (self.data.loc[nan_mask, "AY IncorrectRate adjusted"] - self.data.loc[
            nan_mask, "BX IncorrectRate adjusted"]) / (self.data.loc[nan_mask, "AY IncorrectRate adjusted"] +
                                                       self.data.loc[nan_mask, "BX IncorrectRate adjusted"])

        return self.data

    def build_PBI_rt(self):
        """
        Builds the missing PBI_rt points from its components by the formula PBI_rt = (Mean RT AY - Mean RT BX) / (Mean RT AY + Mean RT BX)
        """
        nan_mask = self.data["PBI_rt"].isna()
        self.data.loc[nan_mask, "PBI_rt"] = (self.data.loc[nan_mask, "AY Correct Mean RT"] - self.data.loc[
            nan_mask, "BX Correct Mean RT"]) / (self.data.loc[nan_mask, "AY Correct Mean RT"] +
                                                self.data.loc[nan_mask, "BX Correct Mean RT"])

        return self.data

    def build_pbi_composite(self):
        """
        Builds the missing PBI_composite points from its components by the formula PBI_composite = mean(z(PBI_error), z(PBI_rt))
        """
        self.data = self.build_PBI_error()
        self.data = self.build_PBI_rt()

        nan_mask = self.data["PBI_composite"].isna()
        self.data.loc[nan_mask, "PBI_composite"] = np.mean([zscore(self.data.loc[nan_mask, "PBI_error"]),
                                                            zscore(self.data.loc[nan_mask, "PBI_rt"])], axis=0)

        return self.data
