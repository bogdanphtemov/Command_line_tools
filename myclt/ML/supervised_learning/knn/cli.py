"""
Command-line interface for K-Nearest Neighbors algorithm.
"""

import sys
from functools import partial

from myclt.common.input_validation import ask_choice, ask_int, ask_float, ask_yes_no
from myclt.common.ui_helpers import clear_screen, print_header, pause

# Use relative imports consistently - module should be run with python -m
from .app_state import AppState, print_status, rebuild_split
from .classifier import KNNClassifier
from .regressor import KNNRegressor
from .metrics import evaluate_knn_model
from .ui import menu_data, menu_train, menu_save_load, menu_evaluate


def main(s: AppState | None = None):
    """
    Main function that launches the KNN interactive UI.
    
    Args:
        s: Optional AppState instance. If None, creates a new state.
    """
    if s is None:
        s = AppState()
    
    # Explicit menu order for consistent UI
    MENU_ORDER = ["Data", "Train", "Evaluate", "Save/Load"]
    
    # Single source of truth for menu actions
    ACTIONS = {
        "Data": partial(menu_data, s),
        "Train": partial(menu_train, s),
        "Evaluate": partial(menu_evaluate, s),
        "Save/Load": partial(menu_save_load, s),
    }
    
    # Menu options derived from explicit order plus Exit
    MENU_OPTIONS = MENU_ORDER + ["Exit"]
    
    while True:
        clear_screen()
        print_header("K-Nearest Neighbors Tool")
        print_status(s)

        choice = ask_choice("Select action: ", MENU_OPTIONS)
        # Handle both int (index) and str (option name) return types for robustness
        if isinstance(choice, int):
            action_name = MENU_OPTIONS[choice]
        else:
            action_name = choice

        # Handle exit condition explicitly
        if action_name == "Exit":
            print("Goodbye!")
            break
        
        # Execute the corresponding action
        ACTIONS[action_name]()


if __name__ == "__main__":
    # Allow Python to differentiate between user errors and unexpected errors
    import traceback
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        # Expected user errors - show clean message without traceback
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception:
        # Unexpected errors - show full traceback for debugging
        print("\n" + "=" * 60)
        print("Unexpected error (this is likely a bug):")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)