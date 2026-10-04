"""
Terminal utilities for dynamic terminal size detection and adaptive UI formatting.

This module provides tools to automatically adapt UI elements to the terminal size,
ensuring optimal display across different terminal dimensions.
"""

import os
import sys
import shutil
from typing import Tuple, Optional


def get_terminal_size() -> Tuple[int, int]:
    """
    Get current terminal dimensions (width, height).
    
    Returns:
        Tuple of (width, height) in characters
    """
    try:
        columns, rows = shutil.get_terminal_size()
        return columns, rows
    except (AttributeError, OSError):
        # Fallback for environments where shutil.get_terminal_size fails
        return 80, 24


def get_adaptive_width(min_width: int = 60, max_width: int = 120) -> int:
    """
    Get adaptive terminal width within reasonable bounds.
    
    Args:
        min_width: Minimum allowed width
        max_width: Maximum allowed width
        
    Returns:
        Adaptive width for UI elements
    """
    terminal_width, _ = get_terminal_size()
    
    # Clamp to reasonable bounds
    if terminal_width < min_width:
        return min_width
    elif terminal_width > max_width:
        return max_width
    else:
        return terminal_width


def create_adaptive_separator(char: str = "=") -> str:
    """
    Create a separator line that adapts to terminal width.
    
    Args:
        char: Character to use for the separator
        
    Returns:
        Adaptive separator string
    """
    width = get_adaptive_width()
    return char * width


def adaptive_wrap_text(text: str, width: Optional[int] = None) -> list:
    """
    Wrap text to fit terminal width.
    
    Args:
        text: Text to wrap
        width: Optional custom width (defaults to adaptive width)
        
    Returns:
        List of wrapped lines
    """
    if width is None:
        width = get_adaptive_width()
    
    lines = []
    current_line = ""
    
    for word in text.split():
        if len(current_line + word) + 1 <= width:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines


def adaptive_center(text: str, width: Optional[int] = None) -> str:
    """
    Center text within adaptive width.
    
    Args:
        text: Text to center
        width: Optional custom width
        
    Returns:
        Centered string
    """
    if width is None:
        width = get_adaptive_width()
    
    return text.center(width)


def adaptive_truncate(text: str, max_width: Optional[int] = None, 
                     suffix: str = "...") -> str:
    """
    Truncate text to fit terminal width with ellipsis.
    
    Args:
        text: Text to truncate
        max_width: Maximum width (defaults to adaptive width)
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated string
    """
    if max_width is None:
        max_width = get_adaptive_width()
    
    if len(text) <= max_width:
        return text
    
    return text[:max_width - len(suffix)] + suffix


def print_adaptive_header(title: str, char: str = "=") -> None:
    """
    Print an adaptive header that fits terminal width.
    
    Args:
        title: Header title
        char: Character for borders
    """
    width = get_adaptive_width()
    separator = char * width
    centered_title = adaptive_center(title)
    
    print(separator)
    print(centered_title)
    print(separator)


def is_terminal_interactive() -> bool:
    """
    Check if the terminal is interactive.
    
    Returns:
        True if running in an interactive terminal
    """
    return sys.stdout.isatty() and sys.stdin.isatty()