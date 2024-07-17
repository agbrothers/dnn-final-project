#!/bin/bash

chmod 600 ~/.kaggle/kaggle.json

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Configure kaggle and download the dataset
kaggle config set -n competition -v rsna-2024-lumbar-spine-degenerative-classification
kaggle competitions leaderboard --show 
kaggle competitions download 
mkdir -p data
unzip -o rsna-2024-lumbar-spine-degenerative-classification.zip
rm rsna-2024-lumbar-spine-degenerative-classification.zip

echo "Setup Complete!"
