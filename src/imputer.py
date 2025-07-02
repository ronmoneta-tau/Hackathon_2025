from pandas import DataFrame
from scipy.interpolate import interp1d
import numpy as np


class Imputer:
    def __init__(self, data: DataFrame, features_methods: dict):
        self.data = data
        self.features_methods = features_methods

    def impute(self):
        for feature, method in self.features_methods.items():
            # check boxplots
            # interpolate by behaviour + checking boxplots boundaries
            quartiles = self.data.groupby('TrialType')[feature].quantile([0.25, 0.75]).unstack()

            for idx, group in self.data.groupby('SDAN'):
                # interpolated = group[feature].interpolate(method=method, limit_direction='both')
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
