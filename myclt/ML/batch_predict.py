"""
Universal batch prediction engine for all ML models.

Loads a CSV file containing only feature columns, runs predictions using
the trained model, and saves results to an output CSV.

This module is model-agnostic and works with:
    - LinearRegressionGD  (regression → continuous predictions)
    - LogisticRegressionGD (binary  → 0/1 + probability)
    - MultinomialLogisticRegression  (multiclass → class + probabilities per class)

Design:
    - Uses numpy vectorization (no Python loops over samples)
    - Efficient for large datasets (1000s of rows, 1000s of features)
    - No pandas dependency for core logic (only for optional CSV I/O convenience)
"""

import numpy as np
import os
from typing import List, Optional, Dict, Any, Tuple

from .base_models import standardize_apply


# ============================================================================
# Helper: CSV I/O (numpy-based, no pandas required)
# ============================================================================

def _detect_csv_delimiter(csv_path: str) -> str:
    """
    Auto-detect CSV delimiter by reading the first line.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Detected delimiter character (',' or ';')
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        comma_count = first_line.count(",")
        semicolon_count = first_line.count(";")
        if semicolon_count > comma_count:
            return ";"
        return ","
    except Exception:
        return ","  # safe fallback


def _load_feature_csv(csv_path: str, expected_n_features: int,
                      delimiter: str = None) -> np.ndarray:
    """
    Load a CSV that contains ONLY feature columns (no target).

    Uses numpy's genfromtxt — faster for numerical data.
    Automatically skips header row if present.

    Args:
        csv_path: Path to CSV file
        expected_n_features: Expected number of feature columns
        delimiter: CSV delimiter. If None, auto-detects.

    Returns:
        Feature matrix (n_samples, n_features)

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If column count doesn't match expected features
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if delimiter is None:
        delimiter = _detect_csv_delimiter(csv_path)

    # Try to load with header detection
    try:
        data = np.genfromtxt(csv_path, delimiter=delimiter, dtype=float, skip_header=1)
    except ValueError:
        # Fallback: try without header (maybe no header row)
        try:
            data = np.genfromtxt(csv_path, delimiter=delimiter, dtype=float)
        except ValueError as e:
            raise ValueError(f"Could not parse CSV file '{csv_path}': {e}")

    # Handle single-row CSV (1D array)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_rows, n_cols = data.shape

    if n_cols != expected_n_features:
        raise ValueError(
            f"CSV has {n_cols} columns, but model expects {expected_n_features} features. "
            f"Please provide a CSV with only the feature columns (no target column)."
        )

    return data


def _save_results_csv(results: np.ndarray, header: str, output_path: str,
                      delimiter: str = ";") -> None:
    """
    Save results array to CSV with header.

    Uses numpy's savetxt for fast I/O.

    Args:
        results: 2D array with data rows and result columns
        header: Header line (semicolon-separated column names)
        output_path: Output file path
        delimiter: CSV delimiter
    """
    np.savetxt(
        output_path,
        results,
        delimiter=delimiter,
        header=header,
        comments="",
        fmt="%.6f"  # Sufficient precision for predictions
    )


# ============================================================================
# Main batch prediction function
# ============================================================================

def batch_predict_from_csv(
    csv_path: str,
    model,
    feature_names: List[str],
    use_scaling: bool = False,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_std: Optional[np.ndarray] = None,
    delimiter: Optional[str] = None,
    output_path: Optional[str] = None,
    model_type: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    add_original_features: bool = False,
) -> Dict[str, Any]:
    """
    Batch predict from CSV: load → scale → predict → save → return results summary.

    Args:
        csv_path: Path to CSV with feature columns only (no target column)
        model: Trained model instance (must have .predict() and optionally .predict_proba())
        feature_names: List of feature column names (order must match CSV columns)
        use_scaling: Whether feature scaling was used during training
        scaler_mean: Mean from training (for standardization), required if use_scaling=True
        scaler_std: Std from training (for standardization), required if use_scaling=True
        delimiter: CSV delimiter (default ";")
        output_path: Where to save the results CSV. If None, auto-generates.
        model_type: One of "regression" / "binary" / "multinomial".
                     Auto-detected from model if None.
        class_names: Names for classes (multinomial only). Auto-generated if None.
        add_original_features: If True, include original feature columns in output CSV.

    Returns:
        Dict with keys:
            - "predictions": numpy array of predictions (n_samples,)
            - "probabilities": numpy array of probabilities (optional, shape depends on model)
            - "n_samples": number of rows processed
            - "output_path": path to saved CSV
            - "success": True if successful
            - "model_type": detected model type

    Raises:
        FileNotFoundError: If CSV not found
        ValueError: If column mismatch, model not trained, or invalid params
        RuntimeError: If model doesn't have required methods
    """
    # ========================================================================
    # 1. Validate model and auto-detect model type
    # ========================================================================
    if not hasattr(model, 'is_trained') or not model.is_trained:
        raise RuntimeError("Model is not trained! Train the model before batch prediction.")

    if not hasattr(model, 'predict'):
        raise RuntimeError("Model does not have a 'predict()' method!")

    # Auto-detect model type
    if model_type is None:
        if hasattr(model, 'model_type'):
            mt = model.model_type
            if mt == "linear_regression":
                model_type = "regression"
            elif mt == "logistic_regression":
                model_type = "binary"
            elif mt == "multinomial_logistic_regression":
                model_type = "multinomial"
            else:
                model_type = "regression"  # fallback
        else:
            # Heuristic detection
            if hasattr(model, 'predict_proba'):
                # Try a small prediction to determine shape
                test_X = np.zeros((1, len(feature_names)))
                try:
                    test_proba = model.predict_proba(test_X)
                    if test_proba.ndim == 1:
                        model_type = "binary"
                    elif test_proba.ndim == 2 and test_proba.shape[1] > 2:
                        model_type = "multinomial"
                    else:
                        model_type = "binary"
                except Exception:
                    model_type = "binary"
            else:
                model_type = "regression"

    n_features = len(feature_names)

    # ========================================================================
    # 2. Auto-detect delimiter if not provided
    # ========================================================================
    if delimiter is None:
        delimiter = _detect_csv_delimiter(csv_path)

    # ========================================================================
    # 3. Load CSV data
    # ========================================================================
    X_raw = _load_feature_csv(csv_path, n_features, delimiter)
    n_samples = X_raw.shape[0]

    if n_samples == 0:
        raise ValueError("CSV file is empty (no data rows).")

    # ========================================================================
    # 3. Apply scaling if needed
    # ========================================================================
    if use_scaling:
        if scaler_mean is None or scaler_std is None:
            raise ValueError(
                "Scaling is enabled but scaler_mean/scaler_std are not provided. "
                "Make sure you trained the model with scaling."
            )
        X_scaled = standardize_apply(X_raw, scaler_mean, scaler_std)
    else:
        X_scaled = X_raw

    # ========================================================================
    # 4. Run predictions (vectorized — single numpy call)
    # ========================================================================
    predictions = model.predict(X_scaled)

    # ========================================================================
    # 5. Get probabilities (if available)
    # ========================================================================
    probabilities = None
    probabilities_cols = []
    n_classes = 0

    if model_type == "binary" and hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)
        probabilities_cols = ["probability"]

    elif model_type == "multinomial":
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)
            n_classes = probabilities.shape[1]
            if class_names and len(class_names) == n_classes:
                probabilities_cols = [f"prob_{name}" for name in class_names]
            else:
                probabilities_cols = [f"prob_class_{k}" for k in range(n_classes)]

    # ========================================================================
    # 6. Build output array and save
    # ========================================================================
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"predictions_{base_name}.csv"

    # Build header
    header_parts = []

    if add_original_features:
        header_parts.extend(feature_names)

    # Prediction column name
    pred_col_name = "prediction"
    if model_type == "regression":
        pred_col_name = "prediction"
    elif model_type == "binary":
        pred_col_name = "predicted_class"
    elif model_type == "multinomial":
        pred_col_name = "predicted_class"

    header_parts.append(pred_col_name)
    header_parts.extend(probabilities_cols)

    # Build data array
    data_parts = []

    if add_original_features:
        data_parts.append(X_raw)

    # Reshape predictions to column vector
    pred_col = predictions.reshape(-1, 1).astype(float)
    data_parts.append(pred_col)

    # Add probability columns
    if probabilities is not None:
        if probabilities.ndim == 1:
            data_parts.append(probabilities.reshape(-1, 1))
        elif probabilities.ndim == 2:
            data_parts.append(probabilities)

    results_array = np.hstack(data_parts)

    # Save to CSV
    header_str = delimiter.join(header_parts)
    _save_results_csv(results_array, header_str, output_path, delimiter)

    # ========================================================================
    # 7. Return summary
    # ========================================================================
    result = {
        "predictions": predictions,
        "probabilities": probabilities,
        "n_samples": n_samples,
        "n_features": n_features,
        "output_path": os.path.abspath(output_path),
        "success": True,
        "model_type": model_type,
        "feature_names": feature_names,
        "prediction_column": pred_col_name,
        "probability_columns": probabilities_cols,
        "class_names": class_names,
        "n_classes": n_classes,
    }

    return result


# ============================================================================
# Convenience wrapper with pretty console output
# ============================================================================

def batch_predict_interactive(
    csv_path: str,
    model,
    feature_names: List[str],
    use_scaling: bool = False,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_std: Optional[np.ndarray] = None,
    model_type: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    output_csv_path: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run batch prediction with friendly console output.

    Designed to be called from UI menus (menu_predict).

    Args:
        Same as batch_predict_from_csv()

    Returns:
        Same as batch_predict_from_csv(), plus prints summary to console.
    """
    print("\n" + "=" * 72)
    print("  BATCH PREDICTION")
    print("=" * 72)

    print(f"\n  Loading CSV: {csv_path}")
    print(f"  Features:    {len(feature_names)}")

    try:
        result = batch_predict_from_csv(
            csv_path=csv_path,
            model=model,
            feature_names=feature_names,
            use_scaling=use_scaling,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            delimiter=delimiter,
            output_path=output_csv_path,
            model_type=model_type,
            class_names=class_names,
            add_original_features=False,
        )

        # Print summary
        print(f"\n  ✓ Processed {result['n_samples']} rows successfully!")
        print(f"  ✓ Output saved to: {result['output_path']}")

        # Show a preview of results
        predictions = result['predictions']
        print(f"\n  Prediction summary:")
        print(f"    Min:       {float(predictions.min()):.4f}")

        if result['model_type'] in ("binary", "multinomial"):
            # For classification, show class distribution
            unique, counts = np.unique(predictions, return_counts=True)
            print(f"    Max:       {float(predictions.max()):.4f}")
            print(f"    Classes:   {dict(zip(unique.astype(str), counts))}")
        else:
            # For regression, show statistics
            print(f"    Max:       {float(predictions.max()):.4f}")
            print(f"    Mean:      {float(predictions.mean()):.4f}")
            print(f"    Std:       {float(predictions.std()):.4f}")

        # Show first few rows as preview
        n_preview = min(5, result['n_samples'])
        print(f"\n  First {n_preview} rows preview:")
        print(f"    Columns: {delimiter.join(result['probability_columns'])}")
        print(f"    (Open CSV file to see full results)")

        return result

    except Exception as e:
        print(f"\n  ✗ Error during batch prediction: {e}")
        raise
