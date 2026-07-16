"""
Command-Line Interface for Binary SVM (Classification + Regression).

Provides terminal-based interactive menus for the full SVM workflow:
    - Data loading and preprocessing
    - Model configuration and training
    - Evaluation and visualization
    - Session save/load
    - Batch prediction

Example:
    >>> from myclt.ML.supervised_learning.svm.cli import run_cli
    >>> run_cli()  # Start interactive CLI
"""

import logging
from typing import Optional
from .app_state import AppState, print_status
from .ui import (
    menu_data, menu_train, menu_evaluate,
    menu_save_load, menu_predict, menu_visualize
)
from myclt.common.input_validation import ask_choice
from myclt.common.ui_helpers import clear_screen, print_header


logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for Binary SVM (Classification or Regression).

    Asks the user which mode to use, then runs the interactive CLI.
    """
    print_header("SVM — Select Mode")
    options = ["Binary Classification (LinearSVM / KernelSVM)",
               "Regression — SVR (LinearSVR / KernelSVR)"]
    choice = ask_choice("Choose mode:", options)
    mode = "classifier" if choice == 0 else "regressor"
    run_cli(mode=mode)


def run_cli(state: Optional[AppState] = None, mode: str = 'classifier') -> AppState:
    """
    Start the interactive CLI for Binary SVM.

    Args:
        state: Optional existing AppState. If None, creates new one.
        mode: 'classifier' (default) or 'regressor'

    Returns:
        AppState with the final session state
        (useful for testing / programmatic integration).
    """
    if state is None:
        state = AppState()
        state.mode = mode

    while True:
        clear_screen()
        print_header(f"SVM {'Classification' if state.mode == 'classifier' else 'Regression'}")
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
            logger.exception("Error in CLI menu action")
            print(f"\nError: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
