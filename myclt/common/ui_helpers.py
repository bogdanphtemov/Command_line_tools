import os

# cleaning the terminal
def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")

# print the menu
def print_header(title: str) -> None:
    
    print("=" * 72)
    print(title.center(72))
    print("=" * 72)

# pause after displaying results
def pause(msg: str = "Press Enter to continue...") -> None:
    input(f"\n{msg}")