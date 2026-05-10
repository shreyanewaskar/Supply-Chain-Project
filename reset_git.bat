@echo off
echo Resetting Git repository...

REM Remove .git folder
rmdir /s /q .git

REM Initialize new repository
git init
git add .
git commit -m "Initial commit - models excluded via gitignore"

REM Set branch and remote
git branch -M main
git remote add origin https://github.com/shreyanewaskar/Supply_chain_management.git

REM Force push
git push -u origin main --force

echo Done!
pause
