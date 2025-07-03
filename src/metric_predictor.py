
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import interp1d


class MetricPredictor:

    def __init__(self, df):
        self.df = df

    def get_best_metric(self, measurement, idx_to_hide):
        df = self.df[["ID", measurement]].copy()

        # Step 1: Filter IDs with at least 6 measurements
        df = self.get_df_with_valid_participants(df, measurement)

        actual_values = []
        preds_interpolation = []
        preds_mean = []
        preds_median = []
        preds_knn = []

        for participant_id, group in df.groupby('ID'):
            # for idx_to_hide in group.index:
            group = group.reset_index(drop=True)
            actual = group.loc[idx_to_hide, measurement]

            interpolated, mean_pred, median_pred, knn_pred = self.get_pred_for_id(group, idx_to_hide, measurement)

            actual_values.append(actual)
            preds_interpolation.append(interpolated)
            preds_mean.append(mean_pred)
            preds_median.append(median_pred)
            preds_knn.append(knn_pred)

        best_method = self.evaluate_method_accuracy(preds_interpolation, preds_mean, preds_median, preds_knn, actual_values)
        return best_method

    def get_pred_for_id(self, group, idx_to_hide, measurement):
        # randomly hide 1 value
        actual = group.loc[idx_to_hide, measurement]

        # Remove the value to predict
        group_hidden = group.drop(index=idx_to_hide)

        interpolated = self.interpolate(group_hidden, measurement, group, idx_to_hide)
        mean_pred = group_hidden[measurement].mean()
        median_pred = group_hidden[measurement].median()
        knn_pred = self.predict_using_knn(group, idx_to_hide, group_hidden, measurement)

        return interpolated, mean_pred, median_pred, knn_pred

    def get_df_with_valid_participants(self, df, measurement):
        # Filter IDs with at least 6 measurements
        df = df.dropna(subset=[measurement])
        counts = df['ID'].value_counts()
        max_count = counts.max()
        valid_ids = counts[counts >= max_count].index
        return df[df['ID'].isin(valid_ids)]

    def predict_using_knn(self, group, idx_to_hide, group_hidden, measurement):
        # KNN Prediction
        try:
            # Use row position as "time" if no actual time feature
            X_train = np.arange(len(group_hidden)).reshape(-1, 1)
            y_train = group_hidden[measurement].values
            knn = KNeighborsRegressor(n_neighbors=min(3, len(group_hidden)))
            knn.fit(X_train, y_train)

            test_idx = list(group.index).index(idx_to_hide)
            X_test = np.array([[test_idx]])
            return knn.predict(X_test)[0]
        except:
            return np.nan

    def interpolate(self, group_hidden, measurement, group, idx_to_hide):
        # Interpolation using scipy
        try:
            # Use index positions or actual "Time" if you have it
            x = np.arange(len(group_hidden))
            y = group_hidden[measurement].values

            f_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")

            # Determine the position of the hidden index in full group
            test_position = list(group.index).index(idx_to_hide)
            return float(f_interp(test_position))
        except:
            return np.nan

    def evaluate_method_accuracy(self, preds_interpolation, preds_mean, preds_median, preds_knn, actual_values):
        methods = {
            'interpolation': preds_interpolation,
            'mean': preds_mean,
            'median': preds_median,
            'knn': preds_knn
        }

        scores = {}
        for name, preds in methods.items():
            y_true = np.array(actual_values)
            y_pred = np.array(preds)
            valid_mask = ~np.isnan(y_pred)
            if valid_mask.sum() > 0:
                score = mean_squared_error(y_true[valid_mask], y_pred[valid_mask])
                scores[name] = score

        # Step 4: Return the method with the lowest RMSE
        best_method = min(scores, key=scores.get)
        return best_method
