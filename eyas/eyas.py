import xgboost as xgb
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import time
import sys
import shutil

RANDOM_SEED = 123

# =============================================================================
# SECTION 1: DATA PREPARATION HELPERS
# =============================================================================

def aggregate_features(per_store: np.ndarray) -> np.ndarray:
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


def build_dataset(per_store_list, labels_list):
    """
    Transforms lists of raw features and labels into final numpy matrices.
      - per_store_list: list of np.ndarray, each (Ni, 164)
      - labels_list:    list of floats, length M
    Into:
      - X: np.ndarray of shape (M, 656)
      - y: np.ndarray of shape (M,)
    """
    # sanity: lists must align
    assert isinstance(per_store_list, list) and isinstance(labels_list, list), \
        "Inputs must be Python lists"
    assert len(per_store_list) == len(labels_list), \
        f"Number of feature arrays ({len(per_store_list)}) != number of labels ({len(labels_list)})"

    M = len(per_store_list)
    X = np.zeros((M, 164 * 4), dtype=np.float32)
    y = np.zeros(M, dtype=np.float32)

    for i, (per_store, label) in enumerate(zip(per_store_list, labels_list)):
        # sanity: per-store block shape and label type
        assert isinstance(per_store, np.ndarray), f"per_store_list[{i}] is not an ndarray"
        assert per_store.ndim == 2 and per_store.shape[1] == 164, \
            f"per_store_list[{i}] must be shape (N,164), got {per_store.shape}"
        assert np.isscalar(label), f"Label at index {i} must be a scalar float"

        X[i] = aggregate_features(per_store)
        y[i] = float(label)
    return X, y


def parse_raw_dataset(path: str, max_samples: int = 0):
    """
    Parses all .npz files in a directory to extract features and labels.
    """
    # per_store_features is a 3D array containing N candidate data
    # per_store_features[i]: Features of candidate #i (i < N)
    # per_store_features[i][j]: 164 per-store features extracted from BufferStoreNode #j of the candidate #i
    # per_store_features[i][j][k]: #k feature value among the 164 per-store features
    per_store_features = []

    # aggregate_labels is a 1D array containing N candidate data
    # aggregate_labels[i]: Cost of candidate #i (i < N)
    aggregate_labels = []

    # List all files in a given directory
    files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

    # Parse npz files containing per-store feature values and corresponding labels
    npz_files = [f for f in files if f.endswith('.npz')]
    for npz_file in npz_files:
        p = os.path.join(path, npz_file)
        npz_data = np.load(p, allow_pickle=True)
        f = npz_data['features']
        c = npz_data['costs']
        assert f.shape[0] == c.shape[0]
        n_candidates = f.shape[0]

        # Process each candidate data
        for candidate in range(n_candidates):
            features = np.asarray(f[candidate], dtype=np.float32)
            cost = np.float32(c[candidate])  # Cost is in MW
            per_store_features.append(features)
            aggregate_labels.append(cost)

    assert len(per_store_features) == len(aggregate_labels)
    print(f"\t# extracted: {len(per_store_features)}")

    # Remove erroneous values and convert units
    valid_indices = [i for i, label in enumerate(aggregate_labels) if 1 <= label * 1e6 <= 500]
    per_store_features = [per_store_features[i] for i in valid_indices]
    aggregate_labels = [aggregate_labels[i] * 1e6 for i in valid_indices]  # Convert to W
    print(f"\t# valid: {len(per_store_features)}")

    # Subsample if max_samples is specified
    if max_samples and len(per_store_features) > max_samples:
        per_store_features = per_store_features[:max_samples]
        aggregate_labels = aggregate_labels[:max_samples]
    print(f"\t# final: {len(per_store_features)}")

    return per_store_features, aggregate_labels


def load_and_prepare_data(base_path: str, model_names: list, max_samples_per_model: int):
    """
    Centralized function to load and process data for a list of ML models.
    """
    all_features = []
    all_labels = []
    for m in model_names:
        print(f"Parsing: {m}")
        model_path = os.path.join(base_path, m)
        features, labels = parse_raw_dataset(model_path, max_samples_per_model)
        all_features.extend(features)
        all_labels.extend(labels)

    return all_features, all_labels


# =============================================================================
# SECTION 2: MODEL TRAINING AND EVALUATION
# =============================================================================

def test(bst: xgb.Booster, best_iteration: int, model_names: list):
    """
    Tests the final trained model on an unseen test set for one or more ML models.
    """
    # Load and prepare the test dataset
    base_path = "./dataset/"
    per_store_features, aggregate_labels = load_and_prepare_data(base_path, model_names, 10000)

    # Build test dataset
    X_test, y_test = build_dataset(per_store_features, aggregate_labels)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Predict using the optimal number of trees found during tuning
    y_test_pred = bst.predict(dtest, iteration_range=(0, best_iteration + 1))

    # Compute and print final test metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    test_error_var = np.var(y_test - y_test_pred)

    print("\nTest set results")
    print(f"\tTest MSE (W^2): {test_mse:.2f}")
    print(f"\tTest MAE (W): {test_mae:.2f}")
    print(f"\tTest MAPE (%): {test_mape * 100:.2f}")
    print(f"\tTest Error Variance (W^2): {test_error_var:.2f}")


# =============================================================================
# SECTION 3: HYPERPARAMETER TUNING STRATEGIES
# =============================================================================

def tune_hyperparameters_and_train(model_names: list):
    """
    Main function to load data and run the hyperparameter tuning strategy.
    """
    # Load and prepare the training dataset
    base_path = "./dataset/"
    per_store_features, aggregate_labels = load_and_prepare_data(base_path, model_names, 20000)

    # Stratified split to ensure validation set represents the label distribution
    labels_series = pd.Series(aggregate_labels)
    label_bins = pd.qcut(labels_series, q=10, labels=False, duplicates='drop')
    per_store_features_train, per_store_features_val, aggregate_labels_train, aggregate_labels_val = train_test_split(
        per_store_features, aggregate_labels, test_size=0.1, random_state=RANDOM_SEED, stratify=label_bins)

    # Prepare data and build matrices
    X_train, y_train = build_dataset(per_store_features_train, aggregate_labels_train)
    X_val, y_val = build_dataset(per_store_features_val, aggregate_labels_val)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Define custom loss & metric functions
    def custom_loss(preds: np.ndarray, dmatrix: xgb.DMatrix):
        labels = dmatrix.get_label()
        grad = 2.0 * (preds - labels)
        hess = np.ones_like(labels) * 2.0
        return grad, hess

    def mape_eval_percent(preds: np.ndarray, dmatrix: xgb.DMatrix):
        labels = dmatrix.get_label().astype(np.float64)
        eps = 1e-8
        frac_mape = np.mean(np.abs((preds - labels) / (labels + eps)))
        return 'MAPE%', frac_mape * 100

    result = _train(dtrain, dval, custom_loss, mape_eval_percent)
    return result


def _train(dtrain, dval, custom_loss, mape_eval_percent):
    """
    Internal function to perform an efficient search over max_depth using early stopping.
    """
    # Hyperparameter search space
    max_depth_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_rounds_ceiling = 2000
    early_stop_rounds = 10  # Using the original successful patience level

    # Initialize trackers
    best_mape = float('inf')
    best_bst = None
    best_params = {}
    best_iteration_overall = 0
    all_candidate_results = []

    # Loop through max_depth values
    for depth in max_depth_values:
        print(f"\n{'=' * 20} TRAINING WITH max_depth={depth} {'=' * 20}")
        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': depth,
            'eta': 0.1,
            'objective': 'reg:squarederror',
            # Regularization Parameters
            'gamma': 1,  # Pruning: A split is made only if the loss reduction > gamma.
            'lambda': 1,  # L2 Regularization: Penalizes large weights, making model conservative.
            'alpha': 0,  # L1 Regularization: Can push some weights to zero.
            'subsample': 0.8,  # Row Subsampling: Uses 80% of data for each tree.
        }
        evals = [(dtrain, 'train'), (dval, 'validation')]

        start = time.time()
        bst = xgb.train(
            params, dtrain, num_boost_round=num_rounds_ceiling, evals=evals,
            early_stopping_rounds=early_stop_rounds,
            obj=custom_loss, custom_metric=mape_eval_percent,
            verbose_eval=False
        )
        print(f"Training completed in {time.time() - start:.2f}s")

        try:
            current_mape = bst.best_score
            best_iter = bst.best_iteration
        except AttributeError:
            print("Warning: Model trained to full num_boost_round. `best_score` not set.")
            # Fallback: Manually find best score if training didn't stop early
            evals_result = {}
            # This fallback is imperfect as it requires retraining, but handles the edge case.
            # A more optimized way would be to always capture evals_result.
            bst_fallback = xgb.train(params, dtrain, num_boost_round=num_rounds_ceiling, evals=evals,
                                     evals_result=evals_result, verbose_eval=False)
            validation_mape_history = evals_result['validation']['MAPE%']
            best_iter = np.argmin(validation_mape_history)
            current_mape = validation_mape_history[best_iter]

        print(f"--> Best Validation MAPE for this run: {current_mape:.2f}% at iteration {best_iter}")
        all_candidate_results.append({
            'max_depth': depth, 'best_iteration_found': best_iter, 'validation_mape_%': current_mape
        })

        if current_mape < best_mape:
            best_mape = current_mape
            best_bst = bst
            best_params = params.copy()
            best_iteration_overall = best_iter
            best_bst.set_attr(max_depth=str(depth))
            print(f"★★★ New best MAPE found ★★★")

    # Create and print a summary DataFrame
    results_df = pd.DataFrame(all_candidate_results)
    print(f"\n\n{'=' * 25} HYPERPARAMETER TUNING SUMMARY {'=' * 25}")
    print(results_df.sort_values(by='validation_mape_%').to_markdown(index=False, floatfmt=".2f"))
    print(f"\nBest Overall Validation MAPE: {best_mape:.2f}%")
    print(f"   Achieved with max_depth = {best_params['max_depth']}")
    print(f"   Optimal number of trees found = {best_iteration_overall + 1}")

    return best_bst, best_iteration_overall


# =============================================================================
# SECTION 4: MAIN EXECUTION WORKFLOW
# =============================================================================

def test_per_model(bst, best_iteration, test_models):
    """
    Iterates through a list of test models and runs the test function for each one.
    """
    print(f"\n--- Per-Model Final Test ---")
    for model_name in test_models:
        print(f"\n--- Testing on model: {model_name} ---")
        test(
            bst=bst,
            best_iteration=best_iteration,
            model_names=[model_name],  # Pass model name as a list
        )


def test_all_models(bst, best_iteration, test_models):
    """
    Tests the trained model on the aggregated dataset of all test models.
    """
    print(f"\n--- Aggregated Final Test ---")
    test(
        bst=bst,
        best_iteration=best_iteration,
        model_names=test_models,  # Pass the full list of test models
    )


def run_experiment(train: bool):
    """
    Main workflow to run the hyperparameter search and final test.
    """
    random.seed(RANDOM_SEED)

    # --- CONFIGURATION ---
    TRAINING_MODELS = [
        't5', 'distilbert', 'longformer', 'efficientnet_b1',
        'inception_v3', 'vgg16', 'segformer', 'detr'
    ]
    TEST_MODELS = [
        'opt', 'gpt2', 'bert', 'mobilenet_v3_large',
        'densenet169', 'resnet50', 'dinov2_large', 'yolov4'
    ]

    if train:
        # Tune hyperparameters and train the best model
        best_model, best_iter = tune_hyperparameters_and_train(model_names=TRAINING_MODELS)

        # Save the best model
        model_filename = "power_model.json"
        best_model.save_model(model_filename)
        print(f"\nBest model is saved to {model_filename}")

        # Integrate the final model into TVM
        shutil.copy2(src="power_model.json", dst= "../python/tvm/meta_schedule/cost_model/power_model/")
        print(f"\nBest model is integrated into TVM")


    if test:
        # Load a pre-trained model
        model_filename = "power_model.json"
        try:
            best_model = xgb.Booster()
            best_model.load_model(model_filename)
            best_iter = best_model.best_iteration
            print(f"\nSuccessfully loaded model from {model_filename}")

            # Get the max_depth attribute directly from the loaded model
            # This was an attribute set during training
            max_depth = best_model.attr('max_depth')

            # If the attribute wasn't saved (e.g., from an older version),
            # it will return None, so we add a check.
            if max_depth is None:
                max_depth = "Unknown (not saved in model file)"

            print(f"   --- Loaded Model Details ---")
            print(f"   Max Depth: {max_depth}")
            print(f"   Optimal Trees: {best_iter + 1} (using iteration 0 to {best_iter})")
        except xgb.core.XGBoostError as e:
            print(f"Error loading model: {e}")
            print(f"Please run the script with train=True first to create the model file.")

        # Test the best model on each of the unseen test models individually
        test_per_model(
            bst=best_model,
            best_iteration=best_iter,
            test_models=TEST_MODELS
        )

        # Test the best model on all test models combined
        test_all_models(
            bst=best_model,
            best_iteration=best_iter,
            test_models=TEST_MODELS
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <train?> <test?>")
        print(f"Example 1: {sys.argv[0]} 1 0")
        print(f"Example 2: {sys.argv[0]} 0 1")
        print(f"Example 2: {sys.argv[0]} 1 1")
        exit(1)
    train_enabled = bool(int(sys.argv[1]))
    run_experiment(train=train_enabled)

