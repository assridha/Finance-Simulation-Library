#!/bin/bash
# Script to automate the release process

set -e  # Exit on error

# Get version from setup.py
VERSION=$(grep -o 'version="[^"]*"' setup.py | cut -d'"' -f2)
echo "Preparing release for version $VERSION"

# Ask for confirmation before proceeding
echo ""
echo "This script will perform the following actions:"
echo "1. Verify clean working directory"
echo "2. Sync with GitHub repository"
echo "3. Run tests"
echo "4. Build distribution package"
echo "5. Create git tag v$VERSION"
echo "6. Push commits to GitHub (with confirmation)"
echo "7. Push tag to GitHub"
echo "8. Create GitHub release"
echo "9. Upload to PyPI (with confirmation)"
echo ""
read -p "Do you want to proceed with the release process? (y/n) " -n 1 -r PROCEED
echo
if [[ ! $PROCEED =~ ^[Yy]$ ]]; then
    echo "Release process aborted."
    exit 0
fi

# Check if the working directory is clean
if [[ -n $(git status --porcelain) ]]; then
    echo "Error: Working directory is not clean. Please commit all changes before releasing."
    exit 1
fi

# Sync with GitHub repository
echo "Syncing with GitHub repository..."
git fetch origin
git pull origin main

# Run tests
echo "Running tests..."
python -m unittest discover -s financial_sim_library/tests || { echo "Tests failed, aborting release."; exit 1; }

# Build the distribution
echo "Building distribution..."
python setup.py sdist bdist_wheel

# Create a git tag
echo "Creating git tag v$VERSION..."
git tag -a "v$VERSION" -m "Release version $VERSION"

# Push changes to GitHub
read -p "Push commits to GitHub repository? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin main
    echo "Changes pushed to GitHub successfully."
else
    echo "Changes not pushed. You can push them later with: git push origin main"
fi

# Always push tag to GitHub
echo "Pushing tag v$VERSION to GitHub repository..."
git push origin "v$VERSION"
echo "Tag pushed successfully."

# Always create GitHub release
echo "Creating GitHub release..."
# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    # Check if authenticated with GitHub
    if gh auth status &>/dev/null; then
        echo "Creating GitHub release using GitHub CLI..."
        gh release create "v$VERSION" \
            --title "v$VERSION: Release" \
            --notes-file "RELEASE_NOTES_v$VERSION.md" \
            --discussion-category "Announcements"
        echo "GitHub release created successfully."
        echo "View the release at: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/tag/v$VERSION"
    else
        echo "GitHub CLI not authenticated. Please run 'gh auth login' first."
        echo "After authenticating, run: gh release create v$VERSION --title \"v$VERSION: Release\" --notes-file \"RELEASE_NOTES_v$VERSION.md\""
    fi
else
    echo "GitHub CLI not found. Please create the release manually at:"
    echo "https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[\/:]\(.*\)\.git/\1/')/releases/new"
    echo "Select tag: v$VERSION"
    echo "Use the contents of RELEASE_NOTES_v$VERSION.md for the description."
fi

# Ask for confirmation before uploading to PyPI
read -p "Upload to PyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Uploading to PyPI..."
    if command -v twine &> /dev/null; then
        python -m twine upload dist/*
        echo "Uploaded to PyPI successfully."
    else
        echo "Twine not found. Please install it with: pip install twine"
        echo "Then run: python -m twine upload dist/*"
    fi
else
    echo "Not uploading to PyPI. You can upload later with: python -m twine upload dist/*"
fi

echo "Release $VERSION completed!"
echo ""
echo "=============== RELEASE CHECKLIST ==============="
echo "✅ Updated version number in setup.py"
echo "✅ Updated CHANGELOG.md"
echo "✅ Created release notes in RELEASE_NOTES_v$VERSION.md"
echo "✅ Created git tag v$VERSION"
echo "✅ Pushed tag to GitHub"
echo "✅ Created GitHub release"
echo "$(if [[ $REPLY =~ ^[Yy]$ ]]; then echo "✅"; else echo "⬜"; fi) Uploaded to PyPI"
echo "⬜ Update documentation website"
echo "⬜ Announce the release to users"
echo "===============================================" 