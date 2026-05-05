#!/bin/bash

# Setup script for behavioral guidelines integration
echo "🔧 Setting up Behavioral Guidelines Integration..."

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "⚠️  pre-commit not found. Install with: pip install pre-commit"
fi

# Create git commit message hook
cat > .git/hooks/commit-msg << 'EOF'
#!/bin/bash
# Behavioral Guidelines Commit Message Check

commit_file=$1
commit_msg=$(cat "$commit_file")

# Check for goal-driven execution
if [[ ! "$commit_msg" =~ (fix|feat|refactor|docs|test|chore) ]]; then
    echo "⚠️  Consider using conventional commit format (fix/feat/refactor/etc.)"
    echo "   This helps ensure clear success criteria"
fi

# Check for vague descriptions
if [[ ${#commit_msg} -lt 10 ]]; then
    echo "⚠️  Commit message seems vague. Add more detail about what was changed."
fi

# Check for behavioral guidelines keywords
if [[ "$commit_msg" =~ (simplify|minimal|surgical|guidelines) ]]; then
    echo "✅ Good: Following behavioral guidelines"
fi

exit 0
EOF

chmod +x .git/hooks/commit-msg

echo "✅ Behavioral guidelines integration setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Install pre-commit: pip install pre-commit"
echo "   2. Run: pre-commit install"
echo "   3. Use the behavioral-guidelines workflow in Windsurf"
echo "   4. Check settings.json for customization options"
