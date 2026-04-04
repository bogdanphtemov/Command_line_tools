from .ui import menu_data , menu_train , menu_evaluate , menu_predict , menu_visualize
from .app_state import AppState , print_status
from ...common.ui_helpers import clear_screen , print_header
from ...common.input_validation import ask_choice

"""
linear_regression_cli.py

A practical CLI tool for Linear / Multiple Linear Regression using
Batch Gradient Descent (from scratch), with:
- CSV (delimiter ';') or manual input
- feature/target selection
- train/test split
- standardization (z-score)
- metrics: MSE, RMSE, R^2
- plots (matplotlib): 1D regression or y_true vs y_pred
"""

# "entry point" of the entire CLI tool
def main() -> None:
    state = AppState()
    
    while True:
        clear_screen()
        print_header("Linear Regression Tool (MVP)")
        print_status(state)

        options = [
            "Data",
            "Train",
            "Evaluate",
            "Predict",
            "Visualize",
            "Exit",
        ]

        choice = ask_choice("" , options)

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
        else:
            return
        # do not pause here; main.py already pauses after subprocess returns


if __name__ == "__main__":
    main()