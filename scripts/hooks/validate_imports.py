#!/usr/bin/env python3
"""
HyperMRI Import Standards Validator.

Validates that imports follow the standards specified in
MIGRATION_TO_REGISTRY_SYSTEM.md.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


class ImportValidator(ast.NodeVisitor):
    """AST visitor to validate import standards."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.errors: List[Tuple[int, str]] = []
        self.imports: List[Tuple[int, str, str]] = []  # (line, module, type)

        # Expected import order
        self.import_groups = {"stdlib": [], "third_party": [], "local": []}

        # Standard library modules (partial list)
        self.stdlib_modules = {
            "os",
            "sys",
            "json",
            "yaml",
            "pathlib",
            "typing",
            "abc",
            "time",
            "datetime",
            "collections",
            "functools",
            "itertools",
            "logging",
            "warnings",
            "traceback",
            "importlib",
            "contextlib",
            "copy",
        }

        # Third-party modules
        self.third_party_modules = {
            "numpy",
            "pandas",
            "xarray",
            "matplotlib",
            "scipy",
            "sklearn",
            "tqdm",
            "IPython",
            "git",
            "pytest",
        }

    def visit_Import(self, node: ast.Import) -> None:
        """Validate regular imports."""
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            self.imports.append((node.lineno, alias.name, "import"))
            self._categorize_import(node.lineno, module_name, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Validate from imports."""
        if node.module:
            module_name = node.module.split(".")[0]
            self.imports.append((node.lineno, node.module, "from"))
            self._categorize_import(node.lineno, module_name, node.module)

            # Check for specific import patterns
            self._validate_from_import_patterns(node)
        self.generic_visit(node)

    def _categorize_import(
        self, line_no: int, module_name: str, full_module: str
    ) -> None:
        """Categorize import into stdlib, third-party, or local."""
        if module_name in self.stdlib_modules:
            self.import_groups["stdlib"].append((line_no, full_module))
        elif module_name in self.third_party_modules:
            self.import_groups["third_party"].append((line_no, full_module))
        elif module_name.startswith(".") or "hypermri" in full_module:
            self.import_groups["local"].append((line_no, full_module))
        else:
            # Assume third-party if not recognized
            self.import_groups["third_party"].append((line_no, full_module))

    def _validate_from_import_patterns(self, node: ast.ImportFrom) -> None:
        """Validate specific from import patterns."""
        if node.module == "typing":
            # Check for proper typing imports
            imported_names = [alias.name for alias in node.names]
            self._validate_typing_imports(node.lineno, imported_names)

        elif node.module and node.module.startswith("hypermri"):
            # Check for proper local imports
            self._validate_local_imports(node)

    def _validate_typing_imports(self, line_no: int, imported_names: List[str]) -> None:
        """Validate typing imports follow standards."""
        # Recommended typing imports order
        recommended_order = [
            "Dict",
            "List",
            "Tuple",
            "Union",
            "Optional",
            "Any",
            "Protocol",
            "TypeVar",
            "Generic",
            "Callable",
            "Literal",
            "ClassVar",
            "Final",
        ]

        # Check if imports are in recommended order
        present_imports = [name for name in recommended_order if name in imported_names]
        if present_imports != sorted(
            present_imports, key=lambda x: recommended_order.index(x)
        ):
            self.errors.append(
                (
                    line_no,
                    f"Typing imports should be ordered: {', '.join(recommended_order)}",
                )
            )

    def _validate_local_imports(self, node: ast.ImportFrom) -> None:
        """Validate local hypermri imports."""
        if node.module:
            # Check for deprecated import patterns
            deprecated_patterns = {
                "hypermri.brukerexp": "Use hypermri.vendors.bruker instead",
                "hypermri.recon_strategy": "Use hypermri.reconstruction.strategies instead",
            }

            for deprecated, suggestion in deprecated_patterns.items():
                if deprecated in node.module:
                    self.errors.append(
                        (
                            node.lineno,
                            f"Import from '{deprecated}' is deprecated. {suggestion}",
                        )
                    )

    def validate_import_order(self) -> None:
        """Validate that imports are grouped and ordered correctly."""
        if not self.imports:
            return

        # Check for proper grouping
        last_group = None
        last_line = 0

        for line_no, module, import_type in self.imports:
            current_group = self._get_import_group(module)

            # Check if we're going backwards in groups
            if last_group and current_group:
                group_order = ["stdlib", "third_party", "local"]
                if (
                    group_order.index(current_group) < group_order.index(last_group)
                    and line_no > last_line + 1
                ):  # Allow for comments between groups
                    self.errors.append(
                        (
                            line_no,
                            f"Import groups should be ordered: stdlib, third-party, local",
                        )
                    )

            last_group = current_group
            last_line = line_no

    def _get_import_group(self, module: str) -> str:
        """Determine which group an import belongs to."""
        module_name = module.split(".")[0]

        if module_name in self.stdlib_modules:
            return "stdlib"
        elif module_name in self.third_party_modules:
            return "third_party"
        elif module.startswith(".") or "hypermri" in module:
            return "local"
        else:
            return "third_party"

    def validate_standard_imports(self) -> None:
        """Validate that standard imports are present for certain file types."""
        filepath_str = str(self.filepath)

        # Files that should have typing imports
        if any(
            pattern in filepath_str for pattern in ["registry", "interface", "protocol"]
        ):
            has_typing = any("typing" in module for _, module, _ in self.imports)
            if not has_typing:
                self.errors.append(
                    (1, "Registry/Interface files should import from typing module")
                )

        # Data processing files should have numpy/xarray
        if any(
            pattern in filepath_str for pattern in ["recon", "process", "transform"]
        ):
            has_numpy = any("numpy" in module for _, module, _ in self.imports)
            has_xarray = any("xarray" in module for _, module, _ in self.imports)

            if not has_numpy and not has_xarray:
                self.errors.append(
                    (1, "Data processing files should import numpy and/or xarray")
                )


def validate_file(filepath: Path) -> List[Tuple[int, str]]:
    """Validate import standards in a single file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        validator = ImportValidator(filepath)
        validator.visit(tree)

        # Additional validations
        validator.validate_import_order()
        validator.validate_standard_imports()

        return validator.errors

    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e.msg}")]
    except Exception as e:
        return [(0, f"Error processing file: {str(e)}")]


def main() -> int:
    """Main entry point for pre-commit hook."""
    if len(sys.argv) < 2:
        print("Usage: validate_imports.py <file1.py> [file2.py ...]")
        return 1

    exit_code = 0

    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)

        if not filepath.suffix == ".py":
            continue

        errors = validate_file(filepath)

        if errors:
            print(f"\n❌ Import violations in {filepath}:")
            for line_no, message in errors:
                print(f"  Line {line_no}: {message}")
            exit_code = 1

    if exit_code == 0:
        print("✅ All files pass import standards checks")
    else:
        print(f"\n💡 See MIGRATION_TO_REGISTRY_SYSTEM.md for import standards")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
