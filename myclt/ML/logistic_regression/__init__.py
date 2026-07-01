"""
Logistic Regression module.

Provides both:
    - Binary Logistic Regression (LogisticRegressionGD)
    - Multinomial (Softmax) Logistic Regression (MultinomialLogisticRegression)

Example usage:
    # Binary
    model = LogisticRegressionGD(learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Multinomial
    model = MultinomialLogisticRegression(learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)  # (n_samples, n_classes)
"""

from .core import LogisticRegressionGD, MultinomialLogisticRegression
from .data import Dataset, Prepareddata, load_csv_dataset
from .preprocessing import train_test_split, standardize_fit, standardize_apply
from .metrics import accuracy, precision, recall, f1_score, confusion_matrix
from .app_state import AppState, print_status, rebuild_split
from .multinomial_app_state import MultinomialAppState, print_status as m_print_status, rebuild_split as m_rebuild_split
from .multinomial_ui import (
    menu_data_multinomial, menu_train_multinomial, menu_evaluate_multinomial,
    menu_predict_multinomial, menu_visualize_multinomial, menu_save_load_multinomial
)

__all__ = [
    # Core models
    'LogisticRegressionGD',
    'MultinomialLogisticRegression',
    # Data
    'Dataset',
    'Prepareddata',
    'load_csv_dataset',
    # Preprocessing
    'train_test_split',
    'standardize_fit',
    'standardize_apply',
    # Metrics
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    # Binary state & menus
    'AppState',
    'print_status',
    'rebuild_split',
    # Multinomial state & menus
    'MultinomialAppState',
    'm_print_status',
    'm_rebuild_split',
    'menu_data_multinomial',
    'menu_train_multinomial',
    'menu_evaluate_multinomial',
    'menu_predict_multinomial',
    'menu_visualize_multinomial',
    'menu_save_load_multinomial',
]
