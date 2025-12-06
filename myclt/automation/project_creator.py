import platform
import os
"""
 project_creator.py is a basic utility for creating a simple structure for new projects
"""

# A dictionary that will build a structure depending on the programming language
LANGUAGES = {
    "python" : {
        "description" : "A Python project with a basic structure, venv, and Git." , 
        "folders" : [
            "src",
            "tests"
        ],
        "files" : {
            "README.md" : "# New Python project\n",
            "requirements.txt" : "",
            ".gitignore" : (
            "venv/\n"
            "__pycache__/\n"
            "*.pyc\n"
            "build/\n"
            "dist/\n"
            "*.egg-info/\n"
            ".env\n"
            ".DS_Store\n"
            "Thumbs.db\n"
            ".vscode/\n"
            ".idea/\n"
            "*.log\n"
            "*.tmp\n"
            "*.swp\n"
            ),
            "src/__init__/py" : "",
            "tests/__init__.py" : ""
        },
        "venv_command" : (
            "python -m venv venv"
            if platform.system() == "Windows" 
            else "python3 -m venv venv"
        ),
        "main_file" : "src/main.py",
        "main_file_content" : (
            
            'def main():\n'
            '    print("Hello from your new Python project!")\n\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        
        )
    }
}

# Asks the user for a programming language and returns it for further processing
def choose_language():

   print("\nAvailable languages:")

   for lang in LANGUAGES: # Runs through all available languages
    print(f" - {lang}")
   
# Starts an infinite loop until the language is selected correctly
   while True:
    choice = input("\nSpecify the language you will use in the project: ").strip().lower()

    if choice in LANGUAGES:
        print(f"\nLanguage selected: {choice}")
        return choice
    
    print("!Unknown language. Try again!\n")

# Asks for the path and name of the project, creates a root folder, and returns its path
def create_project_folder():
    
    while True: # We create a loop where the user will enter the path to initialize the project

        initialization_path = input("\nSpecify the path where the project will be created: ").strip()

        if os.path.isdir(initialization_path):
            break
        else:
            print("!This path does not exist.Please check if the path exists and try again!\n")

    while True: # We create a loop where the user will enter a name for the project

        project_name = input("Enter project name: ").split()

        if project_name == "":
            print("!Project name cannot be empty!")
        else:
            break
    
    # form the complete project path
    project_path = os.path.join(initialization_path , project_name)

    # We check if the folder exists, if so, we warn.
    if os.path.exists(project_path):

        print("\n!This folder already exists at this path!")

        warning = input("Do you want to use this folder anyway? (y/n): ").strip().lower()

        if warning != "y":
            print("Operation cancelled.")
            return None
    else:
        os.makedirs(project_path)
        print(f"\nThe folder has been created: {project_path}")

    return project_path

# Creates all folders and files according to the selected language
def generate_project_structure(project_path , language):

    config = LANGUAGES[language]
# create folders
    for folder in config["folders"]:
        folder_path = os.path.join(project_path , folder)
        os.makedirs(folder_path, exist_ok=True)

        print(f"Created folder: {folder_path}")
# create files
    for file_path , content in config["files"].items():
        full_path = os.path.join(project_path , file_path)
        os.makedirs(os.path.dirname(full_path) , exist_ok=True)

        with open(full_path , "w" , encoding="utf-8") as f:
            f.write(content)
            f.close()
        
        print(f"Created file: {full_path}")
# create main files
    main_path = os.path.join(project_path , config["main_file"])
    os.makedirs(os.path.dirname(main_path) , exist_ok=True)

    with open(main_path , "w" , encoding="utf-8") as f:
        f.write(config["main_file_content: "])
        f.close()
    
    print(f"Created main file: {main_path}")



    print("\nproject structure has been successfully created!")

# why code dont commit?











    

    

