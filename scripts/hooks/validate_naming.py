#!/usr/bin/env python3
"""
HyperMRI Naming Convention Validator.

Validates that code follows the naming conventions specified in
MIGRATION_TO_REGISTRY_SYSTEM.md.
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple


class NamingValidator(ast.NodeVisitor):
    """AST visitor to validate naming conventions."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.errors: List[Tuple[int, str]] = []

        # Expected naming patterns based on migration plan
        self.class_patterns = {
            "handlers": re.compile(r"^[A-Z][a-zA-Z]*Handler$"),
            "strategies": re.compile(r"^[A-Z][a-zA-Z]*RecoStrategy$"),
            "registries": re.compile(r"^[A-Z][a-zA-Z]*Registry$"),
            "sequences": re.compile(r"^[A-Z][a-zA-Z]*Sequence$"),
            "experiments": re.compile(r"^MRIExperiment$|^[A-Z][a-zA-Z]*Experiment$"),
        }

        # Forbidden class names
        self.forbidden_names = {
            "MRIExp",  # Use MRIExperiment instead
        }

        # Variable naming patterns
        self.good_var_names = {
            "data_array",
            "reconstruction_options",
            "sequence_parameters",
            "vendor_handler",
        }

        self.bad_var_names = {
            "reco_array",
            "reco_options",
            "seq_params",
            "handler",
            "seq2d_data",
        }

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Validate class names."""
        class_name = node.name
        line_no = node.lineno

        # Check forbidden names
        if class_name in self.forbidden_names:
            self.errors.append(
                (
                    line_no,
                    f"Class name '{class_name}' is deprecated. Use standardized naming.",
                )
            )

        # Check naming patterns for specific types
        self._validate_class_naming(class_name, line_no)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Validate function names."""
        func_name = node.name
        line_no = node.lineno

        # Check snake_case (but skip AST visitor methods which are required to
        # have specific names)
        if (
            not self._is_snake_case(func_name)
            and not func_name.startswith("__")
            and not func_name.startswith("visit_")  # AST visitor methods
        ):
            self.errors.append(
                (line_no, f"Function '{func_name}' should use snake_case naming")
            )

        # Check boolean function naming (but skip private helper methods)
        if (
            self._returns_bool(node)
            and not self._is_boolean_function_name(func_name)
            and not func_name.startswith("_")  # Skip private methods
        ):
            self.errors.append(
                (
                    line_no,
                    f"Boolean function '{func_name}' should start with is_/has_/can_",
                )
            )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Validate variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                line_no = node.lineno

                # Check for deprecated variable names
                if var_name in self.bad_var_names:
                    self.errors.append(
                        (
                            line_no,
                            f"Variable '{var_name}' uses deprecated naming. "
                            f"Consider using standardized alternatives.",
                        )
                    )

        self.generic_visit(node)

    def _validate_class_naming(self, class_name: str, line_no: int) -> None:
        """Validate class name follows expected patterns."""
        # Determine class type and validate pattern
        if class_name.endswith("Handler"):
            if not self.class_patterns["handlers"].match(class_name):
                self.errors.append(
                    (
                        line_no,
                        f"Handler class '{class_name}' should follow pattern: VendorHandler",
                    )
                )

        elif class_name.endswith("RecoStrategy") or class_name.endswith("Strategy"):
            if not self.class_patterns["strategies"].match(class_name):
                self.errors.append(
                    (
                        line_no,
                        f"Strategy class '{class_name}' should follow pattern: VendorSequenceRecoStrategy",
                    )
                )

        elif class_name.endswith("Registry"):
            if not self.class_patterns["registries"].match(class_name):
                self.errors.append(
                    (
                        line_no,
                        f"Registry class '{class_name}' should follow pattern: ComponentRegistry",
                    )
                )

    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention."""
        return re.match(r"^[a-z_][a-z0-9_]*$", name) is not None

    def _is_boolean_function_name(self, name: str) -> bool:
        """Check if function name follows boolean naming convention."""
        boolean_prefixes = ["is_", "has_", "can_", "should_", "will_"]
        return any(name.startswith(prefix) for prefix in boolean_prefixes)

    def _returns_bool(self, node: ast.FunctionDef) -> bool:
        """Check if function has boolean return annotation."""
        if node.returns and isinstance(node.returns, ast.Name):
            return node.returns.id == "bool"
        return False


def validate_file(filepath: Path) -> List[Tuple[int, str]]:
    """Validate naming conventions in a single file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        validator = NamingValidator(filepath)
        validator.visit(tree)

        return validator.errors

    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e.msg}")]
    except Exception as e:
        return [(0, f"Error processing file: {str(e)}")]


def main() -> int:
    """Main entry point for pre-commit hook."""
    if len(sys.argv) < 2:
        print("Usage: validate_naming.py <file1.py> [file2.py ...]")
        return 1

    exit_code = 0

    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)

        if not filepath.suffix == ".py":
            continue

        errors = validate_file(filepath)

        if errors:
            print(f"\n❌ Naming violations in {filepath}:")
            for line_no, message in errors:
                print(f"  Line {line_no}: {message}")
            exit_code = 1

    if exit_code == 0:
        print("✅ All files pass naming convention checks")
    else:
        print(f"\n💡 See MIGRATION_TO_REGISTRY_SYSTEM.md for naming conventions")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
