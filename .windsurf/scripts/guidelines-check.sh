#!/bin/bash

# Karpathy Behavioral Guidelines Pre-commit Check
# Ensures code changes follow all behavioral guidelines

echo "🔍 Running Karpathy Behavioral Guidelines Check..."

# Check for common violations
violations=0

# 1. Think Before Coding - Check for assumptions and tradeoffs
echo "📝 Checking Think Before Coding guidelines..."
commit_msg=$(git log --format=%B -n 1 2>/dev/null || echo "")
if [[ ${#commit_msg} -lt 20 ]]; then
    echo "⚠️  WARNING: Commit message is brief. Did you state assumptions clearly?"
    violations=$((violations + 1))
fi

# 2. Simplicity First - Check for overcomplicated changes
echo "🔧 Checking Simplicity First guidelines..."
files_changed=$(git diff --cached --name-only | wc -l)
if [ "$files_changed" -gt 5 ]; then
    echo "⚠️  SIMPLICITY WARNING: Changing $files_changed files. Is this surgical?"
    violations=$((violations + 1))
fi

lines_added=$(git diff --cached --numstat | awk '{sum += $1} END {print sum}')
if [ "$lines_added" -gt 200 ]; then
    echo "⚠️  SIMPLICITY WARNING: Adding $lines_added lines. Could this be 50 lines instead?"
    violations=$((violations + 1))
fi

# Check for speculative features
if git diff --cached --name-only | xargs grep -l "TODO\|FIXME\|placeholder\|future.*feature" 2>/dev/null; then
    echo "⚠️  SIMPLICITY WARNING: Found speculative features. Are these beyond what was asked?"
    violations=$((violations + 1))
fi

# Check for unnecessary flexibility/configurability
if git diff --cached --name-only | xargs grep -c "config.*option\|parameter.*default\|setting.*value" 2>/dev/null | awk -F: '$2 > 2 {print}' | grep .; then
    echo "⚠️  SIMPLICITY WARNING: Found configuration options. Was this flexibility requested?"
    violations=$((violations + 1))
fi

# 3. Surgical Changes - Check for overreach
echo "🎯 Checking Surgical Changes guidelines..."
# Check if changes touch unrelated areas
changed_dirs=$(git diff --cached --name-only | xargs dirname | sort -u | wc -l)
if [ "$changed_dirs" -gt 3 ]; then
    echo "⚠️  SURGICAL WARNING: Changing $changed_dirs different directories. Are you touching only what's necessary?"
    violations=$((violations + 1))
fi

# Check for formatting/style changes mixed with functional changes
if git diff --cached --name-only | xargs git diff --cached | grep -E "^\+\s*(//|#)\s*" | grep -E "(format|style|clean|tidy)" > /dev/null; then
    echo "⚠️  SURGICAL WARNING: Found style/formatting comments. Are you improving unrelated code?"
    violations=$((violations + 1))
fi

# 4. Goal-Driven Execution - Check for verifiable criteria
echo "🎯 Checking Goal-Driven Execution guidelines..."
# Check if commit message indicates clear success criteria
if [[ ! "$commit_msg" =~ (test|fix|add|remove|update|refactor) ]]; then
    echo "⚠️  GOAL WARNING: Commit doesn't clearly indicate what was accomplished. Define success criteria."
    violations=$((violations + 1))
fi

# Check for vague goal indicators
if [[ "$commit_msg" =~ (work|stuff|things|update|fix|improve) && ${#commit_msg} -lt 30 ]]; then
    echo "⚠️  GOAL WARNING: Vague goal description. Transform into verifiable goal."
    violations=$((violations + 1))
fi

# Summary and guidance
if [ $violations -gt 0 ]; then
    echo ""
    echo "🚨 Found $violations potential behavioral guideline violations:"
    echo ""
    echo "📋 Think Before Coding:"
    echo "   • State assumptions explicitly"
    echo "   • Surface tradeoffs and alternatives"
    echo "   • Ask when uncertain"
    echo ""
    echo "🔧 Simplicity First:"
    echo "   • Minimum code to solve the problem"
    echo "   • No speculative features"
    echo "   • Would a senior engineer say this is overcomplicated?"
    echo ""
    echo "🎯 Surgical Changes:"
    echo "   • Touch only what you must"
    echo "   • Every line should trace to user request"
    echo "   • Clean up only your own mess"
    echo ""
    echo "🎯 Goal-Driven Execution:"
    echo "   • Define verifiable success criteria"
    echo "   • Transform vague goals into specific ones"
    echo "   • Loop until verified"
    echo ""
    echo "Continue commit despite warnings? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Commit aborted. Review changes against behavioral guidelines."
        exit 1
    fi
else
    echo "✅ All behavioral guidelines check passed!"
    echo "📈 Success indicators: surgical changes, minimal code, clear goals"
fi

exit 0
