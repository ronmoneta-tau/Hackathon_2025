import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from pandas import DataFrame
from scipy.interpolate import interp1d
import numpy as np

class Imputer:
    def __init__(self, data: DataFrame, features_methods: dict):
        self.data = data
        self.features_methods = features_methods

    def impute(self) -> DataFrame:
        for feature, method in self.features_methods.items():
            quartiles = self.data.groupby('TrialType')[feature].quantile([0.25, 0.75]).unstack()

            for idx, group in self.data.groupby('SDAN'):
                # original_indices = group.index.copy()
                # group = group.reset_index(drop=True)
                original = np.array(group[feature])
                x = group.index[~group[feature].isna()]
                y = group[feature].dropna()
                if len(x) > 1:
                    f = interp1d(x, y, kind=method, fill_value='extrapolate')
                    new_values = f(group.index)
                    nan_indices = np.where(np.isnan(original))[0]
                    for nan_idx in nan_indices:
                        trial_type = quartiles.iloc[nan_idx]
                        q1, q3 = trial_type.loc[0.25], trial_type.loc[0.75]
                        if new_values[nan_idx] < q1:
                            new_values[nan_idx] = q1
                        elif new_values[nan_idx] > q3:
                            new_values[nan_idx] = q3
                    group[feature] = new_values
                    self.data.update(group)
                    # self.data.loc[original_indices, feature] = new_values

        return self.data



class InterpolationStrategy(ABC):
    """
    Strategy interface for interpolation.
    """

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


class QuartileCensor:
    """
    Applies censoring of values to within the IQR bounds per TrialType.
    """

    def __init__(self, quantiles: pd.DataFrame):
        self.quantiles = quantiles

    def censor(self, trial_types: pd.Series, values: np.ndarray) -> np.ndarray:
        lower = trial_types.map(lambda t: self.quantiles.loc[t, 0.25])
        upper = trial_types.map(lambda t: self.quantiles.loc[t, 0.75])
        return np.clip(values, lower.values, upper.values)


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
        self.interpolator = interpolator_cls(kind=method)

    def compute_quartiles(self) -> pd.DataFrame:
        return (
            self.df.groupby('TrialType')[self.feature]
            .quantile([0.25, 0.75])
            .unstack()
        )

    def impute(self) -> pd.Series:
        quartiles = self.compute_quartiles()
        result = self.df[self.feature].copy().astype(float)

        for sdan, group in self.df.groupby('SDAN'):
            idx = group.index.to_numpy()
            values = group[self.feature].to_numpy()

            valid_mask = ~group[self.feature].isna().to_numpy()
            x = idx[valid_mask]
            y = values[valid_mask]

            interpolated = self.interpolator.interpolate(x, y, idx)

            censor = QuartileCensor(quartiles)
            censored = censor.censor(group['TrialType'], interpolated)

            # Only replace NaNs in the original series
            nan_positions = np.where(np.isnan(values))[0]
            result.iloc[group.index[nan_positions]] = censored[nan_positions]

        return result


class DataImputer:
    """
    Orchestrates imputation for multiple features using specified methods.

    Usage:
        imputer = DataImputer(df, {'FeatureA': 'linear', 'FeatureB': 'cubic'})
        filled_df = imputer.impute()
    """

    def __init__(self, df: pd.DataFrame, features_methods: Dict[str, str]):
        self.df = df.copy()
        self.features_methods = features_methods
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def impute(self) -> pd.DataFrame:
        for feature, method in self.features_methods.items():
            if feature not in self.df.columns:
                self.logger.warning(f"Feature '{feature}' not in DataFrame; skipping.")
                continue

            self.logger.info(f"Imputing feature '{feature}' using '{method}' interpolation.")
            imputer = FeatureImputer(self.df, feature, method)
            self.df[feature] = imputer.impute()

        return self.df
