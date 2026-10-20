"""
User Interface for K-Nearest Neighbors algorithm.
"""

import os
import shutil  # Moved from inside function for PEP 8 compliance
import traceback  # For better error reporting

from .app_state import AppState , print_status , rebuild_split
from .data import load_csv_dataset , manual_input_dataset , select_features_and_target, Dataset
from .metrics import evaluate_knn_model
from .classifier import KNNClassifier
from .regressor import KNNRegressor
from .session_adapter import KNNSessionAdapter
from myclt.common.input_validation import (
    ask_choice, ask_int, ask_float, ask_yes_no
)
from myclt.common.ui_helpers import clear_screen , print_header , pause
from myclt.ML.session_storage import SessionStorage


def menu_save_load(s: AppState) -> None:
    """Save/Load complete KNN sessions."""
    storage = SessionStorage()
    adapter = KNNSessionAdapter()
    
    while True:
        clear_screen()
        print_header("KNN Tool — Save/Load Session")
        print_status(s)
        
        options = [
            "Save session",
            "Load session",
            "List sessions",
            "Delete session",
            "Back"
        ]
        
        choice = ask_choice("", options)
        
        if choice == 0:
            # Save
            try:
                session_name = input("Session name: ").strip()
                if not session_name:
                    print("Invalid name")
                    pause()
                    continue
                
                print("Saving functionality not yet implemented")
            except Exception as e:
                print(f"Error: {e}")
            pause()
        
        elif choice == 1:
            # Load
            print("Loading functionality not yet implemented")
            pause()
        
        elif choice == 2:
            # List
            print("No sessions found")
            pause()
        
        elif choice == 3:
            # Delete
            print("Deletion functionality not yet implemented")
            pause()
        
        else:
            break

def menu_data(s: AppState) -> None:
    """Data handling menu."""
    print("Data menu not yet implemented")
    pause()

def menu_train(s: AppState) -> None:
    """Model training."""
    print("Training functionality not yet implemented")
    pause()

def menu_evaluate(s: AppState) -> None:
    """Model evaluation."""
    print("Evaluation functionality not yet implemented")
    pause()