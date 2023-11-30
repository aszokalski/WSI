import argparse
import os
import shutil
import tempfile
import subprocess
import glob

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Create a zip archive.")
parser.add_argument("--lib", required=True, type=str, help="Path to the lib directory.")
parser.add_argument("--lab", required=True, type=str, help="Path to the lab directory.")
args = parser.parse_args()

lab_dir = args.lab
last_dir = os.path.basename(lab_dir)

# Define the files and directories to include in the zip file
files_and_dirs_to_include = [lab_dir, args.lib, "requirements.txt"]

# Export Jupyter notebook files in 'lab' directory to PDF
for notebook in glob.glob(os.path.join(lab_dir, "*.ipynb")):
    os.system(f"jupyter nbconvert --to pdf {notebook}")

# Get a list of all files in the git repo
git_files = subprocess.check_output(["git", "ls-files"]).decode("utf-8").splitlines()


# add ./ to the beginning of each file
git_files = ["./" + file for file in git_files]

# Create a temporary directory
with tempfile.TemporaryDirectory() as temp_dir:
    for item in files_and_dirs_to_include:
        if os.path.isdir(item):
            for root, dirs, files in os.walk(item):
                for file in files:
                    # check if file is in git repo
                    if os.path.join(root, file) in git_files:
                        if lab_dir in root:
                            end_dir = temp_dir + root[len(lab_dir) :]
                        else:
                            end_dir = temp_dir + root[1:]
                        # if end dir doesnt exist, create it
                        if not os.path.exists(end_dir):
                            os.makedirs(end_dir)
                        shutil.copy(os.path.join(root, file), end_dir)
        else:
            if "./" + item in git_files or item.endswith(".pdf"):
                shutil.copy(item, temp_dir)

    # Create a zip file of the temporary directory
    # make archives directory if it doesnt exist
    if not os.path.exists("archives"):
        os.mkdir("archives")
    shutil.make_archive(f"archives/{last_dir}", "zip", root_dir=temp_dir)
