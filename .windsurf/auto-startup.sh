#!/bin/bash

# Auto-startup script for behavioral guidelines
# Runs automatically when Windsurf starts

# Ensure pre-commit hooks are always active
if command -v pre-commit &> /dev/null; then
    pre-commit install --overwrite > /dev/null 2>&1
fi

# Set up git hooks if they don't exist
if [ ! -f .git/hooks/pre-commit ]; then
    .windsurf/scripts/setup-hooks.sh > /dev/null 2>&1
fi

# Ensure behavioral guidelines are active
echo "Behavioral guidelines: ACTIVE (foundational)"
echo "No configuration needed - always on by default"
