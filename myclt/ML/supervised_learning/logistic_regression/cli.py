"""Small CLI entry for Logistic Regression.

This mirrors the structure used by linear_regression/cli.py: menus
are implemented in `ML/logistic_regression/ui.py` and this file
only dispatches the top-level menu.
"""

from myclt.common.input_validation import ask_choice
from myclt.common.ui_helpers import clear_screen, print_header
from .app_state import AppState, print_status
from .ui import menu_data, menu_train, menu_evaluate, menu_predict, menu_visualize, menu_save_load
from .core import LogisticRegressionGD


def main() -> None:
    state = AppState()

    while True:
        clear_screen()
        print_header("Logistic Regression Tool (MVP)")
        print_status(state)

        options = [
            "Data",
            "Train",
            "Evaluate",
            "Predict",
            "Visualize",
            "Save/Load",
            "Exit",
        ]

        choice = ask_choice("", options)

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
        else:
            return


if __name__ == "__main__":
    main()
