#!/usr/bin/env python3
"""
HyperMRI Type Hints Validator.

Validates that functions have proper type hints as specified in
MIGRATION_TO_REGISTRY_SYSTEM.md.
"""

import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple


class TypeHintValidator(ast.NodeVisitor):
    """AST visitor to validate type hints."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.errors: List[Tuple[int, str]] = []
        self.has_typing_imports = False
        self.typing_imports: Set[str] = set()

        # Required type annotations for different function types
        self.data_processing_functions = {
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
        }

    def visit_Import(self, node: ast.Import) -> None:
        """Track typing imports."""
        for alias in node.names:
            if alias.name == "typing":
                self.has_typing_imports = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track typing imports."""
        if node.module == "typing":
            self.has_typing_imports = True
            for alias in node.names:
                self.typing_imports.add(alias.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Validate class-level type hints."""
        # Check for class attributes with type hints
        self._validate_class_attributes(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Validate function type hints."""
        if not self._is_private(node.name) and not self._is_test_function(node.name):
            self._validate_function_type_hints(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Validate async function type hints."""
        if not self._is_private(node.name):
            self._validate_function_type_hints(node)
        self.generic_visit(node)

    def _validate_class_attributes(self, node: ast.ClassDef) -> None:
        """Validate that class attributes have type hints where appropriate."""
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                # This is a typed attribute - good!
                continue
            elif isinstance(child, ast.Assign):
                # Check if this should have type hints
                for target in child.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        attr_name = target.id
                        if self._should_have_type_hint(attr_name):
                            self.errors.append(
                                (
                                    child.lineno,
                                    f"Class attribute '{attr_name}' should have type hint",
                                )
                            )

    def _validate_function_type_hints(self, node: ast.FunctionDef) -> None:
        """Validate function parameter and return type hints."""
        func_name = node.name
        line_no = node.lineno

        # Check parameter type hints
        missing_param_types = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            if not arg.annotation:
                missing_param_types.append(arg.arg)

        if missing_param_types:
            self.errors.append(
                (
                    line_no,
                    f"Function '{func_name}' missing type hints for parameters: {', '.join(missing_param_types)}",
                )
            )

        # Check return type annotation
        if not node.returns and not self._is_init_method(func_name):
            # Special handling for different function types
            if self._is_data_processing_function(func_name):
                self.errors.append(
                    (
                        line_no,
                        f"Data processing function '{func_name}' should have return type annotation",
                    )
                )
            elif self._is_public_method(func_name):
                self.errors.append(
                    (
                        line_no,
                        f"Public function '{func_name}' should have return type annotation",
                    )
                )

        # Validate typing imports are present if needed
        if (missing_param_types or not node.returns) and not self.has_typing_imports:
            self.errors.append(
                (line_no, f"File should import from typing module for type hints")
            )

        # Check for proper use of Optional, Union, etc.
        self._validate_advanced_type_usage(node)

    def _validate_advanced_type_usage(self, node: ast.FunctionDef) -> None:
        """Validate proper use of advanced typing constructs."""
        func_name = node.name
        line_no = node.lineno

        # Check for proper Optional usage (instead of Union[X, None])
        for arg in node.args.args:
            if arg.annotation:
                if self._is_union_with_none(arg.annotation):
                    self.errors.append(
                        (
                            line_no,
                            f"Parameter '{arg.arg}' should use Optional[T] instead of Union[T, None]",
                        )
                    )

        # Check return type for Union with None
        if node.returns and self._is_union_with_none(node.returns):
            self.errors.append(
                (
                    line_no,
                    f"Function '{func_name}' should use Optional[T] instead of Union[T, None] for return type",
                )
            )

    def _should_have_type_hint(self, attr_name: str) -> bool:
        """Check if attribute should have type hint."""
        # Common patterns that should be typed
        typed_patterns = [
            "config",
            "registry",
            "handler",
            "strategy",
            "options",
            "parameters",
            "metadata",
            "data",
            "array",
            "path",
        ]
        return any(pattern in attr_name.lower() for pattern in typed_patterns)

    def _is_data_processing_function(self, func_name: str) -> bool:
        """Check if function processes data arrays."""
        func_name_lower = func_name.lower()
        return any(
            pattern in func_name_lower for pattern in self.data_processing_functions
        )

    def _is_public_method(self, func_name: str) -> bool:
        """Check if this is a public method."""
        return not func_name.startswith("_") and not self._is_test_function(func_name)

    def _is_init_method(self, func_name: str) -> bool:
        """Check if this is an __init__ method."""
        return func_name == "__init__"

    def _is_private(self, name: str) -> bool:
        """Check if name indicates private method/class."""
        return name.startswith("_")

    def _is_test_function(self, name: str) -> bool:
        """Check if this is a test function."""
        return name.startswith("test_")

    def _is_union_with_none(self, annotation: ast.AST) -> bool:
        """Check if annotation is Union[T, None] pattern."""
        if isinstance(annotation, ast.Subscript):
            if (
                isinstance(annotation.value, ast.Name)
                and annotation.value.id == "Union"
            ):
                # Check if one of the subscript elements is None
                if isinstance(annotation.slice, ast.Tuple):
                    return any(
                        isinstance(elt, ast.Constant) and elt.value is None
                        for elt in annotation.slice.elts
                    )
        return False


def validate_file(filepath: Path) -> List[Tuple[int, str]]:
    """Validate type hints in a single file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        validator = TypeHintValidator(filepath)
        validator.visit(tree)

        return validator.errors

    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e.msg}")]
    except Exception as e:
        return [(0, f"Error processing file: {str(e)}")]


def main() -> int:
    """Main entry point for pre-commit hook."""
    if len(sys.argv) < 2:
        print("Usage: validate_type_hints.py <file1.py> [file2.py ...]")
        return 1

    exit_code = 0

    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)

        if not filepath.suffix == ".py":
            continue

        errors = validate_file(filepath)

        if errors:
            print(f"\n❌ Type hint violations in {filepath}:")
            for line_no, message in errors:
                print(f"  Line {line_no}: {message}")
            exit_code = 1

    if exit_code == 0:
        print("✅ All files pass type hint checks")
    else:
        print(f"\n💡 See MIGRATION_TO_REGISTRY_SYSTEM.md for type hinting conventions")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
