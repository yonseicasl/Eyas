import xgboost as xgb
import numpy as np
import time
import os


class PowerModel:
    model: xgb.Booster

    def __init__(self):
        # Create XGB model and load weights
        self.model = xgb.Booster()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, 'power_model.json')
        self.model.load_model(weights_path)


    def aggregate_features(self, per_store: np.ndarray) -> np.ndarray:
        """
        Given a per-store feature array of shape (N, 164),
        produce a fixed-length summary vector:
          - mean, std, min, max of each of the 164 features.
        Final vector length = 164 * 4 = 656.
        """
        # sanity: must be a 2D array with 164 columns
        assert per_store.ndim == 2, "per_store must be a 2D array"
        assert per_store.shape[1] == 164, f"Expected 164 features, got {per_store.shape[1]}"

        means = per_store.mean(axis=0)
        stds = per_store.std(axis=0)
        mins = per_store.min(axis=0)
        maxs = per_store.max(axis=0)
        return np.concatenate([means, stds, mins, maxs], axis=0)


    def predict(self, per_store_list: list):
        # Aggregate features
        start_time = time.time()
        M = len(per_store_list)
        X = np.zeros((M, 164 * 4), dtype=np.float32)
        for i, per_store in enumerate(per_store_list):
            # sanity: per-store block shape and label type
            # assert isinstance(per_store, np.ndarray), f"per_store_list[{i}] is not an ndarray"
            # assert per_store.ndim == 2 and per_store.shape[1] == 164, \
            #     f"per_store_list[{i}] must be shape (N,164), got {per_store.shape}"
            X[i] = self.aggregate_features(per_store)
        print(f"Time elapsed for feature aggregation, {M} samples (ms): {(time.time() - start_time) * 1000}")

        # Inference
        start_time = time.time()
        d_new_data = xgb.DMatrix(X)
        predictions = self.model.predict(d_new_data)
        print(f"Time elapsed for power prediction, {M} samples (ms): {(time.time() - start_time) * 1000}")

        return predictions




# Return a power model with pretrained weights
def get_power_model() -> PowerModel:
    return PowerModel()

