#!/usr/bin/env python3
"""
cleaner.py is a minimal utility for cleaning temporary files.
"""

import os
import platform
from typing import List, Tuple
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



def get_temp_dirs() -> List[str]:
    """
    Returns a list of temporary directories to clean up.
    """
    system = platform.system()
    dirs: List[str] = []
    if system in CLEAN_PATHS:
        for path in CLEAN_PATHS[system]:
            if isinstance(path, tuple):
                base, sub = path
                base_dir = os.getenv(base, "")
                if base_dir:
                    dirs.append(os.path.join(base_dir, sub))
            else:
                dirs.append(os.path.expanduser(path))
    return dirs


def scan_directory(path: str) -> Tuple[int, int]:
    """
    Counts the number of files and their total size in a directory.

    Returns:
        Tuple of (file_count, total_size_in_bytes)
    """
    total_size = 0
    file_count = 0
    if not os.path.exists(path):
        return 0, 0

    for root, _, files in os.walk(path):
        for f in files:
            try:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
                file_count += 1
            except (FileNotFoundError, PermissionError):
                pass
    return file_count, total_size


def clean_directory(path: str) -> Tuple[int, int]:
    """
    Move all files in a directory to trash.

    Returns:
        Tuple of (deleted_count, freed_size_in_bytes)
    """
    deleted = 0
    freed = 0
    if not os.path.exists(path):
        return 0, 0

    for root, _, files in os.walk(path):
        for f in files:
            try:
                fp = os.path.join(root, f)
                size = os.path.getsize(fp)
                send2trash(fp)
                deleted += 1
                freed += size
            except Exception:
                pass
    return deleted, freed


def main() -> None:
    """Entry point for the cleaner utility."""
    dirs = get_temp_dirs()
    print("scanning temporary directories...\n")

    total_files, total_size = 0, 0
    for d in dirs:
        files, size = scan_directory(d)
        total_files += files
        total_size += size

    print(f"Found: {total_files} number of temporary files")
    print(f"The total size of these files: {total_size / 1024**2:.2f} MB\n")

    confirm = input("Do you want to delete them (they have been moved to the trash) y/n?: ").strip().lower()

    if confirm == "y":
        deleted, freed = 0, 0
        for d in dirs:
            del_files, del_size = clean_directory(d)
            deleted += del_files
            freed += del_size
        print(f"\nDeleted: {deleted} files")
        print(f"Amount of freed memory: {freed / 1024**2:.2f} MB")
    else:
        print("\nOperation is prohibited")


if __name__ == "__main__":
    main()