import os
import shutil
import subprocess

CONDA_URL = "https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh"
if __name__ == "__main__":
    subprocess.run(f"curl -O {CONDA_URL}", shell=True)
    subprocess.run(f"bash {CONDA_URL.split('/')[-1]} -b", shell=True)
    subprocess.run(f"source ~/.bashrc")