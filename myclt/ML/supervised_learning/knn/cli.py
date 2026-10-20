"""
Command-line interface for K-Nearest Neighbors algorithm.
"""

from myclt.common.input_validation import ask_choice, ask_int, ask_float, ask_yes_no
from myclt.common.ui_helpers import clear_screen, print_header, pause

# Use relative imports consistently - module should be run with python -m
from .app_state import AppState, print_status, rebuild_split
from .classifier import KNNClassifier
from .regressor import KNNRegressor
from .metrics import evaluate_knn_model
from .ui import menu_data, menu_train, menu_save_load, menu_evaluate


def main():
    """Main function that launches the KNN interactive UI."""
    s = AppState()
    
    # Single source of truth for menu actions
    ACTIONS = {
        "Data": lambda: menu_data(s),
        "Train": lambda: menu_train(s),
        "Evaluate": lambda: menu_evaluate(s),
        "Save/Load": lambda: menu_save_load(s),
    }
    
    # Menu options derived from actions plus Exit
    MENU_OPTIONS = list(ACTIONS.keys()) + ["Exit"]
    
    while True:
        clear_screen()
        print_header("K-Nearest Neighbors Tool")
        print_status(s)

        choice = ask_choice("Select action: ", MENU_OPTIONS)
        action_name = MENU_OPTIONS[choice]

        # Handle exit condition explicitly
        if action_name == "Exit":
            print("Goodbye!")
            break
        
        # Execute the corresponding action
        ACTIONS[action_name]()


if __name__ == "__main__":
    # Allow Python to show full traceback for easier debugging
    import sys
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nFor better error reporting, run with: python -m myclt.ML.supervised_learning.knn.cli")
        sys.exit(1)