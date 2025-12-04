#!/usr/bin/env python3
""" 
cleaner.py is a minimal utility for cleaning temporary files.
"""

import os
import tempfile
import shutil
import platform
from send2trash import send2trash

CLEAN_PATHS = {
    "Windows": [
        ("TEMP", ""),
        ("LOCALAPPDATA", "Temp"),
        ("LOCALAPPDATA", "Packages"),
        ("APPDATA", "Microsoft\\Windows\\Recent"),
        ("LOCALAPPDATA", "Microsoft\\Windows\\INetCache"),
        ("LOCALAPPDATA", "Google\\Chrome\\User Data\\Default\\Cache"),
        ("SYSTEMROOT", "SoftwareDistribution\\Download"),
    ],
    "Linux": [
        "~/.cache",
        "/tmp",
        "~/.local/share/Trash/files",
        "~/.var/app",
        "~/.config/google-chrome/Default/Cache",
    ],
    "Darwin": [
        "~/Library/Caches",
        "~/Library/Logs",
        "~/Library/Application Support/Google/Chrome/Default/Cache",
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
            if isinstance(path , tuple): # If this is a tuple (env_var, subpath)
                base , sub = path
                base_dir = os.getenv(base, "")
                if base_dir:
                    dirs.append(os.path.join(base_dir , sub))
            else:                                    # If it's just a line
                dirs.append(os.path.expanduser(path))

    return dirs

def scan_directory(path):
    """
    Counts the number of files and their total size in a directory
    """
    total_size = 0 # total file size
    file_count = 0 # here we will store the number of files
    if not os.path.exists(path):
        return 0 , 0

    # os.walk(path) — walks through a directory and all subdirectories
    # root — current folder
    # _ — list of subfolders (we don't need it, hence "_")
    # files — list of files in this folder

    for root , _, files in os.walk(path):
        for f in files:
            try:
            # create the full path to the file
                fp = os.path.join(root , f)

            # count the number of files and their size in bytes
                total_size += os.path.getsize(fp)

                file_count += 1

            except (FileNotFoundError , PermissionError):
                pass

    return file_count , total_size        

def clean_directory(path):

    deleted = 0 # number of deleted files
    freed = 0 # size of free space

    # checking if a file exists
    if not os.path.exists(path):
        return 0 , 0


    for root , _ , files in os.walk(path):
        for f in files:
            try:
                fp = os.path.join(root , f)

                # Determining the file size
                size = os.path.getsize(fp)

                # safe removal
                send2trash(fp)

                deleted += 1
                freed += size

            except Exception:

                pass
    return deleted , freed   

# entry point to the cleaner.py utility
# checks whether the file is run directly and not imported into another module
if __name__ == "__main__":
    dirs = get_temp_dirs()
    print("scanning temporary directories...\n")

# traversal of all folders and collection of statistics
    total_files , total_size = 0 , 0
    for d in dirs:
        files , size = scan_directory(d)
        total_files += files
        total_size += size

    print(f"Found: {total_files} number of temporary files")
    print(f"The total size of these files: {total_size / 1024**2:.2f} MB\n")

    confim = input("Do you want to delete them (they have been moved to the trash) y/n?: ").strip().lower()

# browsing folders and deleting files
    if confim == "y":
        deleted , freed = 0 , 0
        for d in dirs:
            del_files , del_size = clean_directory(d)
            deleted += del_files
            freed += del_size

        print(f"\nDeleted: {deleted} files")
        print(f"Amount of freed memory: {freed / 1024**2:.2f} MB")
    else:
        print("\nOperation is prohibited")