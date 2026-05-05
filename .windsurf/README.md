# Foundational Behavioral Guidelines

This directory contains the **foundational** behavioral guidelines setup. These guidelines are **always active by default** - no configuration needed.

## 🎯 Purpose

These behavioral guidelines are built into the workspace and automatically enforce:

1. **Think Before Coding** - Don't assume. Don't hide confusion. Surface tradeoffs.
2. **Simplicity First** - Minimum code that solves the problem. Nothing speculative.
3. **Surgical Changes** - Touch only what you must. Clean up only your own mess.
4. **Goal-Driven Execution** - Define success criteria. Loop until verified.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 🔄 Always Active - No Setup Required

**This is how Windsurf behaves by default. Every time you login:**

- ✅ Real-time linting enforces guidelines during coding
- ✅ Pre-commit hooks validate before every commit
- ✅ Git message checks ensure goal-driven execution
- ✅ Continuous monitoring prevents guideline violations

**No flags to set. No configuration needed. No manual activation.**

The behavioral guidelines are foundational to this workspace - they're always on.

## 📁 Structure

```
.windsurf/
├── workflows/
│   └── behavioral-guidelines.md    # Interactive checklist workflow
├── scripts/
│   ├── guidelines-check.sh          # Pre-commit validation script
│   └── setup-hooks.sh              # Setup automation
├── linters/
│   └── guidelines.py               # Custom linter rules
├── settings.json                   # Windsurf configuration
└── README.md                       # This file
```

## 🚀 Quick Setup

1. **Install dependencies:**
   ```bash
   pip install pre-commit
   ```

2. **Run setup script:**
   ```bash
   .windsurf/scripts/setup-hooks.sh
   ```

3. **Install git hooks:**
   ```bash
   pre-commit install
   ```

## 🔄 How It Works

### 1. Windsurf Workflows
- Use `/behavioral-guidelines` command before coding
- Interactive checklist for each guideline principle
- Ensures you think before implementing

### 2. Pre-commit Hooks
- Automatically runs before each commit
- Checks for common violations:
  - Too many files changed (>5)
  - Too many lines added (>200)
  - TODO/FIXME markers (speculative features)
  - Unnecessary abstractions

### 3. Custom Linter
- Real-time feedback in Windsurf
- Detects over-engineering patterns
- Warns about complexity and abstractions

### 4. Git Hooks
- Commit message validation
- Ensures clear, goal-oriented descriptions
- Tracks guideline adherence

## ⚙️ Configuration

Edit `.windsurf/settings.json` to customize:

```json
{
  "guidelines": {
    "simplicityFirst": {
      "maxLinesPerFile": 200,    // Adjust as needed
      "maxFilesPerChange": 5
    }
  }
}
```

## 📋 Daily Usage

### Before Coding:
1. Run `/behavioral-guidelines` workflow
2. Complete the checklist
3. State assumptions and success criteria

### During Coding:
1. Linter provides real-time feedback
2. Warnings for guideline violations
3. Suggestions for simplification

### Before Commit:
1. Pre-commit hook validates changes
2. Checks for surgical changes
3. Ensures minimal, focused commits

## 🎨 Integration Points

### VS Code/Windsurf
- Settings in `.vscode/settings.json`
- Real-time linting and hints
- Workflow commands available

### Git Integration
- Pre-commit validation
- Commit message checks
- Automated enforcement

### Development Workflow
- Checklist before starting
- Validation during development
- Enforcement before commit

## 🔧 Troubleshooting

### Pre-commit not running:
```bash
pre-commit uninstall
pre-commit install
```

### Linter not working:
1. Check Windsurf extensions
2. Verify Python linter dependencies
3. Restart Windsurf

### Workflow not found:
1. Verify `.windsurf/workflows/` directory
2. Check file permissions
3. Restart Windsurf

## 📈 Success Metrics

The integration is working if you see:
- ✅ Fewer unnecessary changes in diffs
- ✅ Shorter, more focused commits
- ✅ Less over-engineering
- ✅ Clearer assumptions stated upfront
- ✅ Verifiable success criteria defined

## 🔄 Maintenance

- Review settings quarterly
- Update linter rules for new patterns
- Adjust thresholds based on project needs
- Gather feedback from team members
