import platform
import os
import subprocess
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
            "src/__init__.py" : "",
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

        project_name = input("Enter project name: ").strip()

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
# 
    else:
        try:
            os.makedirs(project_path)
            print(f"\nThe folder has been created: {project_path}")
        except OSError as e:
            print(f"!Cannot create project folder: {e}!")
            return None

      
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
            
        
        print(f"Created file: {full_path}")
# create main files
    main_path = os.path.join(project_path , config["main_file"])
    os.makedirs(os.path.dirname(main_path) , exist_ok=True)

    with open(main_path , "w" , encoding="utf-8") as f:
        f.write(config["main_file_content"])
        
    
    print(f"Created main file: {main_path}")



    print("\nproject structure has been successfully created!")

# initialize Git
def initilize_git(project_path):

    print("\nInitializing Git repository...")

    # Checking if git is installed or running at all
    try:
        subprocess.run(["git" , "--version"] , stdout=subprocess.PIPE , stderr=subprocess.PIPE)
    
    except FileNotFoundError:
        print("!Git is not installed or not found in PATH!")
        return
    
    # initialize Git
    subprocess.run(["git" , "init"], cwd=project_path)

    print("Git repository initialized.")

    # Making the first commit
    subprocess.run(["git" , "add" , "."] , cwd=project_path)
    subprocess.run(["git" , "commit" , "-m" , "Initial project structure"] , cwd=project_path)

    print("First commit implemented.")

# Creating a virtual environment
def initialize_venv(project_path , language):

# venv is only needed by python
    if language != "python":
        return

    print("\nCreating virtual environment...")
# We extract the desired command
    config = LANGUAGES[language]
    venv_command = config["venv_command"]
# Trying to create a virtual environment
    try:
        subprocess.run(
            venv_command.split(),
            cwd=project_path,
            check=True
        )
    except subprocess.CalledProcessError:
        print("!Failed to create virtual environment!")
        return

    print("Successfully created virtual environment.")

# we tell the user where venv is located
    venv_path = os.path.join(project_path , "venv")
    print(f"Venv location: {venv_path}")

# utility entry point
if __name__ == "__main__":

    print("\n=== Project Creator Utility ===\n")
# user selects the language he needs
    language = choose_language()

# creating a project folder
    project_path = create_project_folder()

# We check if the function returned anything because there may be an access denial or an incorrect name
    if project_path is None:
        print("\nOperation cancelled.")
        exit(0)
# We create the initial project structure
    generate_project_structure(project_path , language)

# Initializing git
    initilize_git(project_path)

# If a virtual environment is needed, we create it
    initialize_venv(project_path , language)

    print("\nThe initial project structure has been successfully created.")

    

        





    








    













    

    

