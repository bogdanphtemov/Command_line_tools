"""
Distance metrics for K-Nearest Neighbors algorithm.

Note:
    These functions operate on single point pairs (1D arrays of features).
    The KNN core uses vectorized implementations for performance; these
    functions are intended for external use (custom metrics, testing, etc.).
"""

from typing import Union, Dict, Callable
import numpy as np


# Type alias for point-like objects
ArrayLike = Union[list, tuple, np.ndarray]


def _as_arrays(point1: ArrayLike, point2: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert point-like objects to numpy arrays with float dtype."""
    return np.asarray(point1, dtype=float), np.asarray(point2, dtype=float)


def euclidean_distance(point1: ArrayLike, point2: ArrayLike) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (list, tuple, or numpy array)
        point2: Second point (list, tuple, or numpy array)
        
    Returns:
        float: Euclidean distance between point1 and point2
    """
    p1, p2 = _as_arrays(point1, point2)
    return np.linalg.norm(p1 - p2)


def manhattan_distance(point1: ArrayLike, point2: ArrayLike) -> float:
    """
    Calculate Manhattan distance between two points.
    
    Args:
        point1: First point (list, tuple, or numpy array)
        point2: Second point (list, tuple, or numpy array)
        
    Returns:
        float: Manhattan distance between point1 and point2
    """
    p1, p2 = _as_arrays(point1, point2)
    return np.sum(np.abs(p1 - p2))


def minkowski_distance(point1: ArrayLike, point2: ArrayLike, p: float = 2) -> float:
    """
    Calculate Minkowski distance between two points.
    
    Args:
        point1: First point (list, tuple, or numpy array)
        point2: Second point (list, tuple, or numpy array)
        p: Power parameter (p=1 for Manhattan, p=2 for Euclidean).
            Must be >= 1.
        
    Returns:
        float: Minkowski distance between point1 and point2
        
    Raises:
        ValueError: If p < 1 (Minkowski requires p >= 1)
    """
    if p < 1:
        raise ValueError(f"Minkowski p must be >= 1, got {p}")
    
    p1, p2 = _as_arrays(point1, point2)
    
    # For p=1 and p=2, use specialized formulas for better performance
    if p == 1:
        return np.sum(np.abs(p1 - p2))  # Manhattan distance
    elif p == 2:
        return np.linalg.norm(p1 - p2)  # Euclidean distance (faster than power)
    else:
        return np.power(np.sum(np.power(np.abs(p1 - p2), p)), 1/p)


def cosine_distance(point1: ArrayLike, point2: ArrayLike) -> float:
    """
    Calculate Cosine distance between two points.
    
    Args:
        point1: First point (list, tuple, or numpy array)
        point2: Second point (list, tuple, or numpy array)
        
    Returns:
        float: Cosine distance between point1 and point2 (1 - cosine_similarity)
        
    Note:
        If either vector is zero, returns 1.0 (maximum distance),
        since cosine similarity is undefined for zero vectors.
    """
    p1, p2 = _as_arrays(point1, point2)
    
    dot_product = np.dot(p1, p2)
    norm1 = np.linalg.norm(p1)
    norm2 = np.linalg.norm(p2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0  # Distance is maximum if one vector is zero
    
    cosine_sim = dot_product / (norm1 * norm2)
    return 1 - cosine_sim


# Define the metrics dictionary after all functions are defined
DISTANCE_METRICS: Dict[str, Callable] = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'minkowski': minkowski_distance,
    'cosine': cosine_distance
}

# Frozen set for validation
SUPPORTED_METRICS = frozenset(DISTANCE_METRICS.keys())


def get_distance_metric(metric_name: str) -> Callable:
    """
    Get distance metric function by name.
    
    Args:
        metric_name: Name of the distance metric
        
    Returns:
        Callable: Distance metric function
        
    Raises:
        ValueError: If metric_name is not supported
    """
    if metric_name not in DISTANCE_METRICS:
        supported_metrics = sorted(DISTANCE_METRICS.keys())
        supported_str = ', '.join(supported_metrics)
        raise ValueError(f"Unsupported distance metric: {metric_name}. "
                        f"Supported metrics: {supported_str}")
    
    return DISTANCE_METRICS[metric_name]


# Define public API
__all__ = [
    'euclidean_distance',
    'manhattan_distance',
    'minkowski_distance',
    'cosine_distance',
    'get_distance_metric',
    'SUPPORTED_METRICS',
    'DISTANCE_METRICS',
    'ArrayLike',
]