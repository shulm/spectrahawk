#!/bin/bash
# Push SpectraHawk to GitHub

echo "Publishing SpectraHawk to GitHub..."

# Ensure we are on the main branch
git branch -M main

# Add the remote if it doesn't exist
git remote add origin https://github.com/shulm/spectrahawk.git 2>/dev/null || git remote set-url origin https://github.com/shulm/spectrahawk.git

# Push
git push -u origin main

echo "Push complete!"
