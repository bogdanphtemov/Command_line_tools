"""CLI entry point for Multinomial Logistic Regression."""

from myclt.common.input_validation import ask_choice
from myclt.common.ui_helpers import clear_screen, print_header
from .multinomial_app_state import MultinomialAppState, print_status
from .multinomial_ui import (
    menu_data_multinomial, menu_train_multinomial, menu_evaluate_multinomial,
    menu_predict_multinomial, menu_visualize_multinomial, menu_save_load_multinomial
)


def main() -> None:
    state = MultinomialAppState()

    while True:
        clear_screen()
        print_header("Multinomial Logistic Regression Tool")
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
            menu_data_multinomial(state)
        elif choice == 1:
            menu_train_multinomial(state)
        elif choice == 2:
            menu_evaluate_multinomial(state)
        elif choice == 3:
            menu_predict_multinomial(state)
        elif choice == 4:
            menu_visualize_multinomial(state)
        elif choice == 5:
            menu_save_load_multinomial(state)
        else:
            return


if __name__ == "__main__":
    main()
