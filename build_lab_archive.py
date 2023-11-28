import argparse
import os
import shutil
import glob
import tempfile
import subprocess

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Create a zip archive.")
parser.add_argument("--lib", required=True, type=str, help="Path to the lib directory.")
parser.add_argument("--lab", required=True, type=str, help="Path to the lab directory.")
args = parser.parse_args()

lab_dir = args.lab
last_dir = os.path.basename(lab_dir)

# # Export Jupyter notebook files in 'lab' directory to PDF
# for notebook in glob.glob(os.path.join(lab_dir, "*.ipynb")):
#     os.system(f"jupyter nbconvert --to pdf {notebook}")

# Define the files and directories to include in the zip file
files_and_dirs_to_include = [lab_dir, args.lib, "readme.md", "requirements.txt"]

# Get a list of all files in the git repo
git_files = subprocess.check_output(["git", "ls-files"]).decode("utf-8").splitlines()
print(git_files)

# Create a temporary directory
with tempfile.TemporaryDirectory() as temp_dir:
    # Copy all the files and directories to the temporary directory
    for item in files_and_dirs_to_include:
        if os.path.isdir(item):
            shutil.copytree(item, os.path.join(temp_dir, os.path.basename(item)))
        else:
            shutil.copy(item, temp_dir)

    # Create a zip file of the temporary directory
    # make archives directory if it doesnt exist
    if not os.path.exists("archives"):
        os.mkdir("archives")
    shutil.make_archive(f"archives/{last_dir}", "zip", root_dir=temp_dir)
