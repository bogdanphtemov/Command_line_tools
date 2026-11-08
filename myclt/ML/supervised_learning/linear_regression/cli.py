"""CLI entry point for Linear Regression.

Usage:
    python -m myclt.ML.supervised_learning.linear_regression.cli

Or (canonical entry point — dispatches here from the top-level menu):
    python -m myclt

Or programmatically:
    from myclt.ML.supervised_learning.linear_regression import main
    main()
"""

from myclt.common.input_validation import ask_choice
from myclt.common.ui_helpers import clear_screen, print_header
from .app_state import AppState, print_status
from .ui import menu_data, menu_train, menu_evaluate, menu_predict, menu_visualize, menu_save_load


def main() -> None:
    """
    Run the Linear Regression interactive CLI.

    Displays the top-level menu until the user selects Exit or presses Ctrl+C.
    All session state lives in memory in a single AppState instance for the
    whole session; nothing is persisted automatically (use Save/Load for that).

    A KeyboardInterrupt/EOFError exits gracefully. Unexpected exceptions are
    intentionally left to propagate: when launched via ``python -m myclt``,
    the top-level dispatcher (main.run_script) already catches them and
    returns the user to the menu.
    """
    state = AppState()

    while True:
        clear_screen()
        print_header("Linear Regression Tool")
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

        try:
            choice = ask_choice("", options)
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted by user. Exiting.")
            return

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
            else:
                return
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted by user. Exiting.")
            return


if __name__ == "__main__":
    main()