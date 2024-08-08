import os
import shutil

# List of directories to remove
dirs_to_remove = [
    "__pycache__",
]

# List of files to remove
files_to_remove = [
    "calibration_results_example.json",
    "utils.txt.py",
]

# Function to remove directories
def remove_dirs(dir_list):
    """
    Remove directories from the given list.
    
    Args:
        dir_list (list): List of directories to remove.
    """
    for dir_path in dir_list:
        for root, dirs, _ in os.walk(".", topdown=False):
            for name in dirs:
                if name == dir_path:
                    dir_to_remove = os.path.join(root, name)
                    print(f"Removing directory: {dir_to_remove}")
                    shutil.rmtree(dir_to_remove, ignore_errors=True)

# Function to remove files
def remove_files(file_list):
    """
    Remove files from the given list.
    
    Args:
        file_list (list): List of files to remove.
    """
    for root, _, files in os.walk("."):
        for name in files:
            if name in file_list:
                file_to_remove = os.path.join(root, name)
                print(f"Removing file: {file_to_remove}")
                os.remove(file_to_remove)

# Run the cleanup functions
remove_dirs(dirs_to_remove)
remove_files(files_to_remove)

print("Cleanup complete.")
