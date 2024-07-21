# Set permissions for the Kaggle JSON file
$acl = Get-Acl "$env:USERPROFILE\.kaggle\kaggle.json"
$acl.SetAccessRuleProtection($true, $false)
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, "Read,Write", "Allow")
$acl.SetAccessRule($rule)
Set-Acl "$env:USERPROFILE\.kaggle\kaggle.json" $acl

# Install required packages
Write-Host "Installing required packages..."
pip install -r requirements-windows.txt

# Configure kaggle and download the dataset
kaggle config set -n competition -v rsna-2024-lumbar-spine-degenerative-classification
kaggle competitions leaderboard --show
kaggle competitions download rsna-2024-lumbar-spine-degenerative-classification

# Create data directory if it doesn't exist
New-Item -ItemType Directory -Force -Path .\data

# Extract the downloaded zip file
Expand-Archive -Path rsna-2024-lumbar-spine-degenerative-classification.zip -DestinationPath .\data -Force

# Remove the zip file
Remove-Item rsna-2024-lumbar-spine-degenerative-classification.zip

Write-Host "Setup Complete!"