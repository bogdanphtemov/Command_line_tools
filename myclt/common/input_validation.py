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
def ask_int(prompt: str, min_val: Optional[int] = None , max_val: Optional[int] = None) -> int:
    while True:

        raw = input(prompt).strip()

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

# function for inputting flout parameters
def ask_float(prompt: str , min_val: Optional[float] = None , max_val: Optional[float] = None) -> float:
    while True:

        raw = input(prompt).strip()

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
def ask_yes_no(prompt: str) -> bool:
    while True:
        
        raw = input(prompt).strip().lower()

        if raw in ("y" , "yes"):
            return True
        if raw in ("n" , "no"):
            return False

        print("!Please enter y/n:")
