"""
Command-Line Interface for Multiclass SVM (One-vs-Rest).

Provides terminal-based interactive menus for the full multiclass SVM workflow:
    - Data loading and preprocessing
    - Model configuration (linear or kernel base estimator)
    - Training and evaluation
    - Visualization
    - Session save/load
    - Batch prediction

Example:
    >>> from myclt.ML.supervised_learning.svm.cli_multinomial import run_cli
    >>> run_cli()  # Start interactive CLI
"""

import sys
from typing import Optional
from .multinomial_app_state import MultinomialAppState, print_status
from .multinomial_ui import (
    menu_data, menu_train, menu_evaluate,
    menu_save_load, menu_predict, menu_visualize
)
from myclt.common.input_validation import ask_choice
from myclt.common.ui_helpers import clear_screen, print_header


def main() -> None:
    """CLI entry point for Multiclass SVM (One-vs-Rest)."""
    run_cli()


def run_cli(state: Optional[MultinomialAppState] = None) -> MultinomialAppState:
    """
    Start the interactive CLI for Multiclass SVM.

    Args:
        state: Optional existing MultinomialAppState. If None, creates new one.

    Returns:
        MultinomialAppState with the final session state
        (useful for testing / programmatic integration).
    """
    if state is None:
        state = MultinomialAppState()

    while True:
        clear_screen()
        print_header("Multiclass SVM (One-vs-Rest)")
        print_status(state)

        options = [
            "Data (load, select features, split)",
            "Train (configure model, train)",
            "Evaluate (test set, metrics)",
            "Predict (single, batch)",
            "Visualize (plots, SVs)",
            "Save / Load session",
            "Exit",
        ]
        try:
            choice = ask_choice("Choose action:", options)
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted by user.")
            return state

        try:
            if choice == 0:
                menu_data(state)
            elif choice == 1:
                menu_train(state)
            elif choice == 2:
                menu_evaluate(state)
            elif choice == 3:
                menu_predict(state)
            elif choice == 4:
                menu_visualize(state)
            elif choice == 5:
                menu_save_load(state)
            elif choice == 6:
                print("\nGoodbye!")
                return state
            else:
                print(f"\nUnexpected choice: {choice}")
                input("Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted by user.")
            return state
        except Exception as e:
            print(f"\nError: {e}")
            input("Press Enter to continue...")
