import os
from .terminal_utils import (
    print_adaptive_header, 
    create_adaptive_separator, 
    adaptive_wrap_text,
    is_terminal_interactive
)

# cleaning the terminal
def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")

# print the menu (adaptive version)
def print_header(title: str) -> None:
    """Print adaptive header that fits terminal width."""
    print_adaptive_header(title)

# pause after displaying results
def pause(msg: str = "Press Enter to continue...") -> None:
    input(f"\n{msg}")

def print_adaptive_text_blocks(text_blocks: list, separator: str = "â”€") -> None:
    """
    Print multiple text blocks with adaptive formatting.
    
    Args:
        text_blocks: List of text strings to print
        separator: Separator character between blocks
    """
    adaptive_separator = create_adaptive_separator(separator)
    
    for i, block in enumerate(text_blocks):
        if i > 0:
            print(adaptive_separator)
        
        # Wrap and print each block
        wrapped_lines = adaptive_wrap_text(block)
        for line in wrapped_lines:
            print(line)