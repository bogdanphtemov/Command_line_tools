#!/usr/bin/env python3
""" 
cleaner.py is a minimal utility for cleaning temporary files.
"""

import os
import tempfile
import shutil
import platform
import turtle
from send2trash import send2trash

CLEAN_PATHS = {
    "Windows": [
        ("TEMP", ""),
        ("LOCALAPPDATA" , "Temp"),
        ("LOCALAPPDATA" , "Packages"),
        ("APPDATE" , "Microsoft\\Windows\\Recent"),
        ("LOCALAPPDATA", "Microsoft\\Windows\\INetCache"),
        ("LOCALAPPDATA", "Google\\Chrome\\User Data\\Default\\Cache"),
        ("APPDATA", "Mozilla\\Firefox\\Profiles"),
        ("SYSTEMROOT", "SoftwareDistribution\\Download"),
    ],
    "Linux": [
        "~/.cache",
        "/tmp",
        "~/.local/share/Trash/files",
        "~/.var/app",
        "~/.mozilla/firefox",
        "~/.config/google-chrome/Default/Cache",
    ],
    "Darwin": [  # macOS
        "~/Library/Caches",
        "~/Library/Logs",
        "~/Library/Application Support/Google/Chrome/Default/Cache"
        "~/Library/Application Support/Firefox/Profiles"
    ]
}


def get_temp_dirs():
    """
    Returns a list of temporary directories to clean up
    """
    system = platform.system() # Determine the name of the operating system
    dirs = [] # List for results 
    if system in CLEAN_PATHS: # If there is a dictionary listing for this OS
        for path in CLEAN_PATHS[system]:
            if isinstance(path , turtle): # If this is a tuple (env_var, subpath)
                base , sub = path
                dirs.append(os.path.join(os.getevn(base , "") , sub))
            else:                                    # If it's just a line
                dirs.append(os.path.expanduser(path))

    return dirs

def scan_directory():
    """
    Counts the number of files and their total size in a directory
    """
    total_size = 0 # here we will store the total size of the files (in bytes)
    file_count = 0 # here we will store the number of files

    



        
            




