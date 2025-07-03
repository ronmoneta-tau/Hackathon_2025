import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np


class InterpolationStrategy(ABC):
    """
    Strategy interface for interpolation.
    """

    kinds = {'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'}

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
        clipped_values[nan_masks] = np.clip(generated_values[nan_masks], lower.values[nan_masks], upper.values[nan_masks])
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

        for _, group in self.df.groupby('SDAN'):
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
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def impute(self) -> pd.DataFrame:
        for feature, method in self.features_methods.items():
            if feature not in self.data.columns:
                self.logger.warning(f"Feature '{feature}' not in DataFrame; skipping.")
                continue

            self.logger.info(f"Imputing feature '{feature}' using '{method}' interpolation.")
            imputer = FeatureImputer(self.data, feature, method)
            self.data[feature] = imputer.impute()

        return self.data
