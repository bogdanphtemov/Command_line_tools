"""
SVM — Support Vector Machine algorithms.

Provides from-scratch implementations of:

    CLASSIFICATION:
        LinearSVM       — Binary linear SVM (hinge loss + gradient descent)
        KernelSVM       — Binary non-linear SVM (kernel trick + dual GD)
        OneVsRestSVM    — Multiclass SVM via One-vs-Rest strategy

    REGRESSION:
        LinearSVR       — Linear Support Vector Regression (ε-insensitive loss)
        KernelSVR       — Non-linear SVR with kernel trick

    KERNEL FUNCTIONS:
        linear, poly, rbf, sigmoid — available via get_kernel()

    UTILITIES:
        SessionAdapter  — Save/restore SVM sessions
        Hyperparameter tuning — Grid/Random search for C, kernel, gamma

Example:
    >>> from myclt.ML.supervised_learning.svm import LinearSVM
    >>> model = LinearSVM(C=1.0, epochs=1000)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
"""

from .core import (
    LinearSVM,
    KernelSVM,
    OneVsRestSVM,
    LinearSVR,
    KernelSVR,
    get_kernel,
)
