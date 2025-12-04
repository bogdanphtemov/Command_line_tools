import os
import subprocess


MENU_STRUCTURE = {
    "Systemic" : {
        "Cleaning temporary files" : "system/cleaner.py"
    },
    "Automation" : {

    },
    "FUN" : {

    }
}    

# clears the terminal
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

# Prints the title
def print_header(title):
    print("=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70)

# Executes the script
def run_script(script_path):
    clear_screen()
    print_header(f"Launching {script_path}")

    # error handling
    try:
        # allows you to run another process in python
        subprocess.run(["python" , script_path] , check=True)
    except FileNotFoundError:
        print(f"!ERROR!: File '{script_path}' not found") 
    except subprocess.CalledProcessError:
        print(f"!ERROR!: Execution error {script_path}")
    except Exception as e:
        print(f"!ERROR!: ?Unknown error?")
    
    input("\nPress Enter to return to the menu...")

def choose_category():
    """Main menu - operation selection"""
    while True:
        clear_screen()
        print_header("Types of operation")
        
        # category list output
        for i , category in enumerate(MENU_STRUCTURE.keys() , start=1):
            print(f"{i}. {category}")
        print("\n0. Exit")

        choice = input("\nSelect the type of operation: ")

        if choice == "0":
            break
        # handling user selection
        try:

            category = list(MENU_STRUCTURE.keys())[int(choice) - 1]
            choose_operation(category)
        except (ValueError , IndexError):
            input("!Incorrect choice!. Press Enter to try again...")


def choose_operation(category):
    """Submenu — selection of operations in a category environment"""
    while True:
        clear_screen()
        print_header(f"{category} - Operations")
        operations = MENU_STRUCTURE[category]

        # number the operations to call them
        for i , op_name in enumerate(operations.keys() , start=1):
            print(f"{i}. {op_name}")
        print("\n0. Exit")

        choice = input("\nSelect an operation: ")

        if choice == "0":
            break

        # Handling user selection
        try:

            op_name = list(operations.keys())[int(choice) - 1]
            run_script(operations[op_name])
        
        except (ValueError , IndexError):
            input("!Incorrect choice!. Press Enter to try again...")

# entry point to our program
if __name__ == "__main__":
    choose_category()