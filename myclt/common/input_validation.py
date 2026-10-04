from typing import List, Optional

# Shows numbered choices and returns selected index (0-based)
def ask_choice(prompt: str, choices: List[str]) -> int:
    while True:
        print(prompt)
        for i, c in enumerate(choices, start=1):
            print(f"{i}) {c}")
        raw = input("Select a function: ").strip()
        try:
            val = int(raw)
            if 1 <= val <= len(choices):
                return val - 1
        except ValueError:
            pass
        print("!Invalid choice! Try again...\n")

# function for inputting integer parameters
def ask_int(prompt: str, min_val: Optional[int] = None, max_val: Optional[int] = None, default: Optional[int] = None) -> int:
    while True:
        if default is not None:
            display_prompt = f"{prompt} [default: {default}]: "
        else:
            display_prompt = f"{prompt}: "
        
        raw = input(display_prompt).strip()
        # Use default if empty input
        if raw == "" and default is not None:
            return default
        try:
            v = int(raw)
            if min_val is not None and v < min_val:
                print(f"! Must be >= {min_val}")
                continue
            if max_val is not None and v > max_val:
                print(f"! Must be <= {max_val}")
                continue
            return v
        except ValueError:
            print("!Please enter the data type: Integer")

# function for inputting float parameters
def ask_float(prompt: str, min_val: Optional[float] = None, max_val: Optional[float] = None, default: Optional[float] = None) -> float:
    while True:
        if default is not None:
            display_prompt = f"{prompt} [default: {default}]: "
        else:
            display_prompt = f"{prompt}: "
        
        raw = input(display_prompt).strip()
        # Use default if empty input
        if raw == "" and default is not None:
            return default
        try:
            v = float(raw)
            if min_val is not None and v < min_val:
                print(f"! Must be >= {min_val}")
                continue
            if max_val is not None and v > max_val:
                print(f"! Must be <= {max_val}")
                continue
            return v
        except ValueError:
            print("!Please enter the data type: Float")

# check user decisions
def ask_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    while True:
        if default is not None:
            default_str = "Y/n" if default else "y/N"
            display_prompt = f"{prompt} [{default_str}]: "
        else:
            display_prompt = f"{prompt} [y/n]: "
        
        raw = input(display_prompt).strip().lower()
        # Use default if empty input
        if raw == "" and default is not None:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False

# Special helper for asking float or auto (returns None for auto)
def ask_auto_or_float(prompt: str, min_val: float = 1e-6, max_val: float = 10.0) -> float | None:
    """
    Prompts for a float value or 'auto'. Enter empty or 'auto' returns None.
    """
    while True:
        raw = input(f"{prompt} [default: auto]: ").strip().lower()
        if raw == "" or raw == "auto":
            return None
        try:
            v = float(raw)
            if v < min_val:
                print(f"! Must be >= {min_val}")
                continue
            if v > max_val:
                print(f"! Must be <= {max_val}")
                continue
            return v
        except ValueError:
            print("!Please enter a valid float or press Enter for auto")

# Special helper for asking integer or auto (returns None for auto)
def ask_auto_or_int(prompt: str, min_val: int = 1, max_val: int = 1_000_000) -> int | None:
    """
    Prompts for an integer value or 'auto'. Enter empty or 'auto' returns None.
    """
    while True:
        raw = input(f"{prompt} [default: auto]: ").strip().lower()
        if raw == "" or raw == "auto":
            return None
        try:
            v = int(raw)
            if v < min_val:
                print(f"! Must be >= {min_val}")
                continue
            if v > max_val:
                print(f"! Must be <= {max_val}")
                continue
            return v
        except ValueError:
            print("!Please enter a valid integer or press Enter for auto")

# Special helper for asking Learning Rate with auto option
def ask_learning_rate(prompt: str, min_val: float = 1e-6, max_val: float = 10.0) -> float | None:
    """
    Prompts for learning rate. Enter empty or 'auto' for automatic selection.
    Returns float if user enters a specific value, or None for auto.
    """
    while True:
        raw = input(f"{prompt} [auto]: ").strip().lower()
        if raw == "" or raw == "auto":
            return None
        try:
            v = float(raw)
            if v < min_val:
                print(f"! Must be >= {min_val}")
                continue
            if v > max_val:
                print(f"! Must be <= {max_val}")
                continue
            return v
        except ValueError:
            print("!Please enter a valid float or press Enter for auto")

def ask_epsilon_with_auto(prompt: str, min_val: float = 0.001, max_val: float = 1.0) -> float | None:
    """
    Ask for epsilon parameter with auto optimization option.
    
    Args:
        prompt: Prompt text
        min_val: Minimum allowed value  
        max_val: Maximum allowed value
        
    Returns:
        Epsilon value or None for auto optimization
    """
    while True:
        display_prompt = f"{prompt} [Enter=Auto]: "
        raw = input(display_prompt).strip()
        if raw == "":
            return None  # Auto optimization
        try:
            v = float(raw)
            if v < min_val:
                print(f"! Must be >= {min_val}")
                continue
            if v > max_val:
                print(f"! Must be <= {max_val}")
                continue
            return v
        except ValueError:
            print("!Please enter a valid float or press Enter for auto optimization")

def ask_kernel_with_auto() -> str | None:
    """
    Ask for kernel selection with auto optimization option.
    
    Returns:
        Kernel name or None for auto optimization
    """
    kernel_options = ['rbf', 'linear', 'poly', 'sigmoid']
    
    print("\n" + "=" * 70)
    print("KERNEL CONFIGURATION [Enter=Auto]")
    print("=" * 70)
    
    while True:
        print("Available kernels:")
        for i, kernel in enumerate(kernel_options, 1):
            print(f"  {i}) {kernel}")
        print("  0) Auto optimization")
        
        raw = input("Select kernel [0 for Auto]: ").strip()
        
        if raw == "" or raw == "0" or raw.lower() == "auto":
            return None  # Auto optimization
        
        try:
            choice = int(raw) - 1
            if 0 <= choice < len(kernel_options):
                return kernel_options[choice]
            else:
                print("!Invalid choice!")
        except ValueError:
            print("!Please enter a number or 0 for Auto")

# Special helper for yes/no with recommended default
def ask_yes_no_recommended(prompt: str, recommended: bool = True) -> bool:
    """
    Like ask_yes_no but shows [Y/n] (recommended: yes/no) to guide the user.
    """
    default_str = "Y/n" if recommended else "y/N"
    recommended_str = "yes" if recommended else "no"
    display_prompt = f"{prompt} [{default_str}] (recommended: {recommended_str}): "
    
    while True:
        raw = input(display_prompt).strip().lower()
        if raw == "":
            return recommended
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("!Please enter y/n:")

# Special helper for asking yes/no with auto option
def ask_yes_no_with_auto(prompt: str, default_auto: bool = True) -> bool | None:
    """
    Ask yes/no question with auto option.
    
    Args:
        prompt: Prompt text
        default_auto: What to return for auto option
        
    Returns:
        True/False for selection, None for auto
    """
    options = ["Yes", "No", "Auto"]
    choice = ask_choice(f"{prompt} [Enter=Auto]", options)
    if choice == 2:  # Auto
        return None
    return choice == 0