"""
Karpathy Behavioral Guidelines Linter
Custom linter rules to enforce all behavioral guidelines during development
"""

import ast
import re
from typing import List, Dict, Any

class KarpathyBehavioralGuidelinesLinter:
    """Enforces all Karpathy behavioral guidelines in code"""

    def __init__(self):
        self.violations = []

    def check_file(self, filepath: str, content: str) -> List[Dict[str, Any]]:
        """Check a file for guideline violations"""
        self.violations = []

        # Parse AST if it's a Python file
        if filepath.endswith('.py'):
            try:
                tree = ast.parse(content)
                self.check_python_file(tree, filepath)
            except SyntaxError:
                pass

        # Check all files for common violations
        self.check_think_before_coding_violations(content, filepath)
        self.check_simplicity_first_violations(content, filepath)
        self.check_surgical_changes_violations(content, filepath)
        self.check_goal_driven_violations(content, filepath)

        return self.violations

    def check_python_file(self, tree: ast.AST, filepath: str):
        """Check Python-specific violations"""
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        # Simplicity First: Check for over-abstraction
        if len(classes) > 3:
            self.add_violation(
                "simplicity_first",
                f"File has {len(classes)} classes. Would a senior engineer say this is overcomplicated?",
                filepath
            )

        # Simplicity First: Check for complex methods
        for node in functions:
            if len(node.body) > 20:  # Too many lines
                self.add_violation(
                    "simplicity_first",
                    f"Method '{node.name}' is too long ({len(node.body)} lines). Could this be 50 lines instead?",
                    filepath
                )

        # Simplicity First: Check for unnecessary abstractions
        for node in classes:
            # Check for single-use abstractions
            class_methods = [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
            if len(class_methods) <= 2 and len(node.bases) > 0:
                self.add_violation(
                    "simplicity_first",
                    f"Class '{node.name}' appears to be a single-use abstraction. Is this necessary?",
                    filepath
                )

    def check_think_before_coding_violations(self, content: str, filepath: str):
        """Check Think Before Coding violations"""
        lines = content.split('\n')

        # Look for assumptions that should be stated
        for i, line in enumerate(lines, 1):
            # Check for magic numbers without explanation
            if re.search(r'\b(?!1|0|2|10|100)\d{2,}\b', line) and not re.search(r'#.*\d', line):
                self.add_violation(
                    "think_before_coding",
                    f"Line {i}: Found magic number. Have you stated your assumptions?",
                    filepath
                )

            # Check for complex logic without comments
            if re.search(r'(if.*and.*if|for.*if.*while)', line) and not re.search(r'#.*', line):
                self.add_violation(
                    "think_before_coding",
                    f"Line {i}: Complex logic without explanation. Have you surfaced tradeoffs?",
                    filepath
                )

    def check_simplicity_first_violations(self, content: str, filepath: str):
        """Check Simplicity First violations"""
        lines = content.split('\n')

        # Check for speculative features
        for i, line in enumerate(lines, 1):
            if re.search(r'\b(TODO|FIXME|XXX|HACK|FUTURE|PLACEHOLDER)\b', line):
                self.add_violation(
                    "simplicity_first",
                    f"Line {i}: Found speculative feature marker. Are you adding beyond what was asked?",
                    filepath
                )

        # Check for over-engineering patterns
        over_engineering_patterns = [
            (r'Factory', "Factory pattern - is this necessary or over-engineered?"),
            (r'Builder', "Builder pattern - could this be simpler?"),
            (r'Abstract.*Interface', "Abstract interface - is this over-engineering?"),
            (r'Singleton', "Singleton pattern - is this flexibility requested?"),
            (r'Observer|Publisher|Subscriber', "Observer pattern - is this complexity needed?"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, message in over_engineering_patterns:
                if re.search(pattern, line):
                    self.add_violation(
                        "simplicity_first",
                        f"Line {i}: {message}",
                        filepath
                    )

        # Check for unnecessary flexibility
        flexibility_patterns = [
            (r'config.*option', "Configuration option - was this flexibility requested?"),
            (r'parameter.*default', "Parameter with default - is this configurability needed?"),
            (r'setting.*value', "Setting value - is this beyond requirements?"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, message in flexibility_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.add_violation(
                        "simplicity_first",
                        f"Line {i}: {message}",
                        filepath
                    )

        # Check for error handling for impossible scenarios
        if re.search(r'try.*except.*pass', content):
            self.add_violation(
                "simplicity_first",
                "Found try/except/pass. Are you handling impossible scenarios?",
                filepath
            )

    def check_surgical_changes_violations(self, content: str, filepath: str):
        """Check Surgical Changes violations"""
        lines = content.split('\n')

        # Check for style/formatting changes (would need git diff for full check)
        for i, line in enumerate(lines, 1):
            # Look for comments about formatting
            if re.search(r'#.*(format|style|clean|tidy|pretty)', line, re.IGNORECASE):
                self.add_violation(
                    "surgical_changes",
                    f"Line {i}: Found formatting comment. Are you improving unrelated code?",
                    filepath
                )

            # Look for refactoring of working code
            if re.search(r'#.*(refactor|improve|optimize)', line, re.IGNORECASE):
                self.add_violation(
                    "surgical_changes",
                    f"Line {i}: Found refactoring comment. Are you refactoring things that aren't broken?",
                    filepath
                )

    def check_goal_driven_violations(self, content: str, filepath: str):
        """Check Goal-Driven Execution violations"""
        lines = content.split('\n')

        # Check for vague function/variable names
        vague_patterns = [
            (r'\b(data|info|stuff|things|items|objects)\b', "Vague naming - be more specific about the goal"),
            (r'\b(process|handle|manage|do)\b', "Vague verb name - what specific action are you performing?"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, message in vague_patterns:
                if re.search(pattern, line):
                    self.add_violation(
                        "goal_driven",
                        f"Line {i}: {message}",
                        filepath
                    )

        # Check for functions without clear purpose
        function_matches = re.findall(r'def\s+(\w+)\s*\(', content)
        for func_name in function_matches:
            if len(func_name) < 4 or func_name in ['temp', 'test', 'demo', 'dummy']:
                self.add_violation(
                    "goal_driven",
                    f"Function '{func_name}' has unclear purpose. Define verifiable success criteria.",
                    filepath
                )

    def add_violation(self, guideline: str, message: str, filepath: str):
        """Add a violation record"""
        self.violations.append({
            'guideline': guideline,
            'message': message,
            'file': filepath,
            'severity': 'warning'
        })

# Example usage in Windsurf extension
def run_karpathy_linter(filepath: str, content: str) -> List[Dict[str, Any]]:
    """Main linter entry point for Karpathy behavioral guidelines"""
    linter = KarpathyBehavioralGuidelinesLinter()
    return linter.check_file(filepath, content)
