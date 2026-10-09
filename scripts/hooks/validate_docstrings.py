#!/usr/bin/env python3
"""
HyperMRI Docstring Standards Validator.

Validates that docstrings follow Google/NumPy style as specified in
MIGRATION_TO_REGISTRY_SYSTEM.md.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


class DocstringValidator(ast.NodeVisitor):
    """AST visitor to validate docstring standards."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.errors: List[Tuple[int, str]] = []

        # Patterns for Google style docstrings
        self.google_sections = {
            "Args:",
            "Arguments:",
            "Parameters:",
            "Param:",
            "Params:",
            "Returns:",
            "Return:",
            "Yields:",
            "Yield:",
            "Raises:",
            "Raise:",
            "Except:",
            "Exceptions:",
            "Example:",
            "Examples:",
            "Note:",
            "Notes:",
            "See Also:",
            "References:",
            "Todo:",
            "Warning:",
            "Warnings:",
        }

        # Patterns for NumPy style docstrings
        self.numpy_sections = {
            "Parameters",
            "Returns",
            "Yields",
            "Raises",
            "See Also",
            "Notes",
            "References",
            "Examples",
        }

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Validate class docstrings."""
        if not self._is_private(node.name):
            self._validate_class_docstring(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Validate function/method docstrings."""
        if not self._is_private(node.name) and not self._is_test_function(node.name):
            self._validate_function_docstring(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Validate async function docstrings."""
        if not self._is_private(node.name):
            self._validate_function_docstring(node)
        self.generic_visit(node)

    def _validate_class_docstring(self, node: ast.ClassDef) -> None:
        """Validate class docstring follows standards."""
        docstring = ast.get_docstring(node)

        if not docstring:
            self.errors.append(
                (
                    node.lineno,
                    f"Class '{node.name}' missing docstring",
                )
            )
            return

        # Check for required sections in class docstrings
        self._validate_class_docstring_content(node, docstring)

    def _validate_function_docstring(self, node: ast.FunctionDef) -> None:
        """Validate function docstring follows standards."""
        docstring = ast.get_docstring(node)

        # Check if function needs docstring
        if self._needs_docstring(node) and not docstring:
            self.errors.append(
                (node.lineno, f"Function '{node.name}' missing docstring")
            )
            return

        if docstring:
            self._validate_function_docstring_content(node, docstring)

    def _validate_class_docstring_content(
        self, node: ast.ClassDef, docstring: str
    ) -> None:
        """Validate class docstring content and structure."""
        lines = docstring.split("\n")

        # Check for one-line summary
        if not lines[0].strip():
            self.errors.append(
                (
                    node.lineno,
                    f"Class '{node.name}' docstring should start with one - line summary",
                )
            )

        # Check for proper Google style formatting
        if self._has_parameters(node) and not self._has_attributes_section(docstring):
            self.errors.append(
                (
                    node.lineno,
                    f"Class '{node.name}' with attributes should have 'Attributes: ' section",
                )
            )

        # Check for Example section in main classes
        if (
            self._is_main_class(node.name)
            and "Example:" not in docstring
            and "Examples:" not in docstring
        ):
            self.errors.append(
                (
                    node.lineno,
                    f"Main class '{node.name}' should include Example section",
                )
            )

    def _validate_function_docstring_content(
        self, node: ast.FunctionDef, docstring: str
    ) -> None:
        """Validate function docstring content and structure."""
        lines = docstring.split("\n")

        # Check for one-line summary
        if not lines[0].strip():
            self.errors.append(
                (
                    node.lineno,
                    f"Function '{node.name}' docstring should start with one - line summary",
                )
            )

        # Check for Args/Parameters section if function has arguments
        args = [arg.arg for arg in node.args.args if arg.arg != "self"]
        if args and not self._has_args_section(docstring):
            self.errors.append(
                (
                    node.lineno,
                    f"Function '{node.name}' with parameters should have Parameters section",
                )
            )

        # Check for Returns section if function has return annotation
        if node.returns and not self._has_returns_section(docstring):
            self.errors.append(
                (
                    node.lineno,
                    f"Function '{node.name}' with return annotation should have Returns section",
                )
            )

        # Check for proper style (Google vs NumPy)
        style = self._detect_docstring_style(docstring)
        if style == "mixed":
            self.errors.append(
                (
                    node.lineno,
                    f"Function '{node.name}' docstring mixes Google and NumPy styles",
                )
            )

        # Validate all functions use NumPy style
        if style != "numpy" and style != "none":
            self.errors.append(
                (
                    node.lineno,
                    f"Function '{node.name}' should use NumPy style docstring, not {style}",
                )
            )

    def _needs_docstring(self, node: ast.FunctionDef) -> bool:
        """Determine if function needs a docstring."""
        # Public functions always need docstrings
        if not self._is_private(node.name):
            return True

        # Special methods (except __init__) don't require docstrings
        if (
            node.name.startswith("__")
            and node.name.endswith("__")
            and node.name != "__init__"
        ):
            return False

        return False

    def _has_parameters(self, node: ast.ClassDef) -> bool:
        """Check if class has __init__ with parameters."""
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                args = [arg.arg for arg in child.args.args if arg.arg != "self"]
                return len(args) > 0
        return False

    def _has_attributes_section(self, docstring: str) -> bool:
        """Check if docstring has Attributes section."""
        return "Attributes\n" in docstring or "Attributes\r\n" in docstring

    def _has_args_section(self, docstring: str) -> bool:
        """Check if docstring has Parameters section."""
        numpy_patterns = ["Parameters\n", "Parameters\r\n"]
        return any(pattern in docstring for pattern in numpy_patterns)

    def _has_returns_section(self, docstring: str) -> bool:
        """Check if docstring has Returns section."""
        numpy_patterns = ["Returns\n", "Returns\r\n"]
        return any(pattern in docstring for pattern in numpy_patterns)

    def _detect_docstring_style(self, docstring: str) -> str:
        """Detect if docstring uses Google, NumPy, or mixed style."""
        has_google = any(section in docstring for section in self.google_sections)
        has_numpy = any(
            f"{section}\n" in docstring or f"{section}\r\n" in docstring
            for section in self.numpy_sections
        )

        if has_google and has_numpy:
            return "mixed"
        elif has_google:
            return "google"
        elif has_numpy:
            return "numpy"
        else:
            return "none"

    def _is_data_processing_function(self, node: ast.FunctionDef) -> bool:
        """Check if function processes data arrays."""
        data_processing_indicators = [
            "reconstruct",
            "process",
            "transform",
            "filter",
            "denoise",
            "interpolate",
            "resample",
            "normalize",
            "correct",
            "fit",
        ]

        func_name_lower = node.name.lower()
        return any(
            indicator in func_name_lower for indicator in data_processing_indicators
        )

    def _is_main_class(self, class_name: str) -> bool:
        """Check if this is a main user-facing class."""
        main_classes = [
            "MRIExperiment",
            "Study",
            "VendorRegistry",
            "ReconstructionRegistry",
            "SequenceRegistry",
            "BrukerHandler",
            "SiemensHandler",
        ]
        return class_name in main_classes

    def _is_private(self, name: str) -> bool:
        """Check if name indicates private method/class."""
        return name.startswith("_")

    def _is_test_function(self, name: str) -> bool:
        """Check if this is a test function."""
        return name.startswith("test_")


def validate_file(filepath: Path) -> List[Tuple[int, str]]:
    """Validate docstring standards in a single file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        validator = DocstringValidator(filepath)
        validator.visit(tree)

        return validator.errors

    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e.msg}")]
    except Exception as e:
        return [(0, f"Error processing file: {str(e)}")]


def main() -> int:
    """Main entry point for pre-commit hook."""
    if len(sys.argv) < 2:
        print("Usage: validate_docstrings.py <file1.py> [file2.py ...]")
        return 1

    exit_code = 0

    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)

        if not filepath.suffix == ".py":
            continue

        errors = validate_file(filepath)

        if errors:
            print(f"\n❌ Docstring violations in {filepath}:")
            for line_no, message in errors:
                print(f"  Line {line_no}: {message}")
            exit_code = 1

    if exit_code == 0:
        print("✅ All files pass docstring standards checks")
    else:
        print(f"\n💡 See MIGRATION_TO_REGISTRY_SYSTEM.md for docstring standards")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
