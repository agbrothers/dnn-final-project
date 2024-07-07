#!/bin/bash

# chmod 600 ~/.kaggle/kaggle.json 
pip install --upgrade --force-reinstall --no-deps kaggle 
kaggle config set -n competition -v rsna-2024-lumbar-spine-degenerative- 
kaggle competitions leaderboard --show 
kaggle competitions download 
unzip -o *.zip 
