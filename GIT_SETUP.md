# Git Setup Instructions

## Initial Setup

1. Initialize Git repository:
```bash
cd C:\Users\sachi\Downloads\Supply_chain_management
git init
```

2. Add all files:
```bash
git add .
```

3. Create initial commit:
```bash
git commit -m "Initial commit: Supply Chain AI Dashboard with LSTM, BiLSTM, and YOLO"
```

## Push to GitHub

1. Create a new repository on GitHub (https://github.com/new)
   - Name: supply-chain-ai-dashboard
   - Description: AI-powered supply chain management with demand forecasting, object detection, and shortage alerts
   - Keep it Public or Private as needed
   - DO NOT initialize with README (we already have one)

2. Link your local repository to GitHub:
```bash
git remote add origin https://github.com/YOUR_USERNAME/supply-chain-ai-dashboard.git
```

3. Push to GitHub:
```bash
git branch -M main
git push -u origin main
```

## Important Notes

- Model files (.h5, .pt) are excluded via .gitignore due to large size
- Dataset files are excluded via .gitignore
- Users will need to download models separately or use Git LFS

## Using Git LFS (Optional - for large model files)

If you want to include model files in git:

1. Install Git LFS:
```bash
git lfs install
```

2. Track model files:
```bash
git lfs track "*.h5"
git lfs track "*.pt"
```

3. Add .gitattributes:
```bash
git add .gitattributes
```

4. Commit and push:
```bash
git add model/
git commit -m "Add model files with Git LFS"
git push
```

## Common Git Commands

```bash
# Check status
git status

# Add specific files
git add filename.py

# Commit changes
git commit -m "Your commit message"

# Push changes
git push

# Pull latest changes
git pull

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main

# View commit history
git log --oneline
```

## Repository Structure for GitHub

```
supply-chain-ai-dashboard/
├── app/
│   └── dashboard.py
├── model/
│   └── .gitkeep (placeholder)
├── Notebook/
├── requirements.txt
├── .gitignore
├── README.md
└── run_dashboard.bat
```

Note: Add a README note that users need to download model files separately or provide a download link.
