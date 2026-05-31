from typing import List , Optional

# Shows numbered choices and returns selected index (0-based)
def ask_choice(prompt: str , choices: List[str]) -> int:
    while True:

        print(prompt)

        for i , c in enumerate(choices , start=1):
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
            display_prompt = f"{prompt} (default {default}): "
        else:
            display_prompt = f"{prompt}: "
        
        raw = input(display_prompt).strip()

        # Use default if empty input
        if raw == "" and default is not None:
            return default

        try :

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
            display_prompt = f"{prompt} (default {default}): "
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

        if raw in ("y" , "yes"):
            return True
        if raw in ("n" , "no"):
            return False

        print("!Please enter y/n:")
