#!/bin/bash
# Script to automate the release process

set -e  # Exit on error

# Get version from setup.py
VERSION=$(grep -o 'version="[^"]*"' setup.py | cut -d'"' -f2)
echo "Preparing release for version $VERSION"

# Check if the working directory is clean
if [[ -n $(git status --porcelain) ]]; then
    echo "Error: Working directory is not clean. Please commit all changes before releasing."
    exit 1
fi

# Run tests
echo "Running tests..."
python -m unittest discover -s financial_sim_library/tests || { echo "Tests failed, aborting release."; exit 1; }

# Build the distribution
echo "Building distribution..."
python setup.py sdist bdist_wheel

# Create a git tag
echo "Creating git tag v$VERSION..."
git tag -a "v$VERSION" -m "Release version $VERSION"

# Ask for confirmation before pushing
read -p "Push tag v$VERSION to remote repository? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin "v$VERSION"
    echo "Tag pushed successfully."
else
    echo "Tag not pushed. You can push it later with: git push origin v$VERSION"
fi

# Ask for confirmation before uploading to PyPI
read -p "Upload to PyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Uploading to PyPI..."
    python -m twine upload dist/*
    echo "Uploaded to PyPI successfully."
else
    echo "Not uploading to PyPI. You can upload later with: python -m twine upload dist/*"
fi

echo "Release $VERSION completed!"
echo "Don't forget to:"
echo "1. Create a GitHub release: https://github.com/your-username/financial_sim_library/releases/new"
echo "2. Update the documentation website"
echo "3. Announce the release to users" 