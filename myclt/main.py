import os

from myclt.ML.supervised_learning.linear_regression.cli import main as run_linear_regression
from myclt.ML.supervised_learning.logistic_regression.cli import main as run_logistic_regression
from myclt.ML.supervised_learning.logistic_regression.cli_multinomial import main as run_multinomial_logistic_regression
from myclt.ML.supervised_learning.svm.cli import main as run_svm
from myclt.ML.supervised_learning.svm.cli_multinomial import main as run_svm_multinomial
from myclt.legacy_code.cleaner import main as run_cleaner
from myclt.legacy_code.project_creator import main as run_project_creator

MENU_STRUCTURE = {
    "Machine learning algorithms with a teacher": {
        "Linear Regression for Forecasting Continuous Values": run_linear_regression,
        "Logistic Regression for Binary Classification": run_logistic_regression,
        "Multinomial Logistic Regression for Multiclass Classification": run_multinomial_logistic_regression,
        "Support Vector Machines (SVM) for Classification and Regression": run_svm,
        "Multiclass SVM (One-vs-Rest)": run_svm_multinomial,
    },
    "Machine learning algorithms without a teacher": {},
    "Machine learning algorithms with reinforcement": {},
    "Legacy Code": {
        "Cleaning temporary files": run_cleaner,
        "Creating a minimal structure for a new project": run_project_creator,
    },
}


# clears the terminal
def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


# Prints the title
def print_header(title: str) -> None:
    """Print a centered header with separators."""
    print("=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70)


# Executes the script
def run_script(func) -> None:
    """Run a callable (function) and wait for user input to return."""
    clear_screen()
    try:
        func()
    except Exception as e:
        print(f"!ERROR!: {e}")
    input("\nPress Enter to return to the menu...")


def choose_category() -> None:
    """Main menu - operation selection"""
    while True:
        clear_screen()
        print_header("Types of operation")
        
        # category list output
        for i, category in enumerate(MENU_STRUCTURE.keys(), start=1):
            print(f"{i}. {category}")
        print("\n0. Exit")

        choice = input("\nSelect the type of operation: ")

        if choice == "0":
            break
        # handling user selection
        try:
            category = list(MENU_STRUCTURE.keys())[int(choice) - 1]
            choose_operation(category)
        except (ValueError, IndexError):
            input("!Incorrect choice!. Press Enter to try again...")


def choose_operation(category: str) -> None:
    """Submenu — selection of operations in a category"""
    while True:
        clear_screen()
        print_header(f"{category} - Operations")
        operations = MENU_STRUCTURE[category]

        # number the operations to call them
        for i, op_name in enumerate(operations.keys(), start=1):
            print(f"{i}. {op_name}")
        print("\n0. Exit")

        choice = input("\nSelect an operation: ")

        if choice == "0":
            break

        # Handling user selection
        try:
            op_name = list(operations.keys())[int(choice) - 1]
            run_script(operations[op_name])
        except (ValueError, IndexError):
            input("!Incorrect choice!. Press Enter to try again...")


# entry point to our program
if __name__ == "__main__":
    choose_category()