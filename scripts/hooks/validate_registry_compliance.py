#!/usr/bin/env python3
"""
HyperMRI Registry Compliance Validator.

Validates that classes implement the correct protocols and follow
registry patterns as specified in MIGRATION_TO_REGISTRY_SYSTEM.md.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


class RegistryComplianceValidator(ast.NodeVisitor):
    """AST visitor to validate registry compliance."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.errors: List[Tuple[int, str]] = []

        # Expected protocol implementations
        self.protocol_requirements = {
            "Handler": "VendorHandlerProtocol",
            "Strategy": "ReconstructionStrategyProtocol",
            "Registry": "RegistryProtocol",
            "Plotter": "PlotterProtocol",
            "Sequence": "SequenceProtocol",
        }

        # Classes that should inherit from base classes
        self.base_class_requirements = {
            "Handler": "BaseVendorHandler",
            "Strategy": "BaseReconstructionStrategy",
            "Registry": "BaseRegistry",
            "Plotter": "BasePlotter",
        }

        # Methods that should be deprecated
        self.deprecated_methods = {
            "get_vendor_handler": "Use VendorRegistry.get_vendor_handler() instead",
            "get_reco_strategy": "Use ReconstructionRegistry.get_strategy() instead",
            "string2class": "Use proper registry lookup instead",
        }

        # (name, line, bases)
        self.classes_found: List[Tuple[str, int, List[str]]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Validate class definitions for registry compliance."""
        class_name = node.name
        line_no = node.lineno

        # Get base class names
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(
                    f"{base.value.id}.{base.attr}"
                    if isinstance(base.value, ast.Name)
                    else base.attr
                )

        self.classes_found.append((class_name, line_no, base_names))

        # Validate protocol compliance
        self._validate_protocol_compliance(node, class_name, line_no, base_names)

        # Validate base class usage
        self._validate_base_class_usage(node, class_name, line_no, base_names)

        # Check for registry pattern compliance
        self._validate_registry_patterns(node, class_name, line_no)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Validate function definitions for deprecated patterns."""
        func_name = node.name
        line_no = node.lineno

        # Check for deprecated methods
        if func_name in self.deprecated_methods:
            suggestion = self.deprecated_methods[func_name]
            self.errors.append(
                (line_no, f"Method '{func_name}' is deprecated. {suggestion}")
            )

        # Check for hardcoded logic patterns
        self._validate_hardcoded_patterns(node, func_name, line_no)

        self.generic_visit(node)

    def _validate_protocol_compliance(
        self, node: ast.ClassDef, class_name: str, line_no: int, base_names: List[str]
    ) -> None:
        """Validate that classes implement required protocols."""
        for class_type, required_protocol in self.protocol_requirements.items():
            if class_name.endswith(class_type):
                if not any(required_protocol in base for base in base_names):
                    self.errors.append(
                        (
                            line_no,
                            f"Class '{class_name}' should implement {required_protocol}",
                        )
                    )

    def _validate_base_class_usage(
        self, node: ast.ClassDef, class_name: str, line_no: int, base_names: List[str]
    ) -> None:
        """Validate that classes inherit from appropriate base classes."""
        for class_type, required_base in self.base_class_requirements.items():
            if class_name.endswith(class_type):
                if not any(required_base in base for base in base_names):
                    self.errors.append(
                        (
                            line_no,
                            f"Class '{class_name}' should inherit from {required_base}",
                        )
                    )

    def _validate_registry_patterns(
        self, node: ast.ClassDef, class_name: str, line_no: int
    ) -> None:
        """Validate registry-specific patterns."""
        if class_name.endswith("Registry"):
            # Check for required registry methods
            method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

            required_methods = ["register", "get", "list_registered"]
            missing_methods = [m for m in required_methods if m not in method_names]

            if missing_methods:
                self.errors.append(
                    (
                        line_no,
                        f"Registry '{class_name}' missing required methods: {', '.join(missing_methods)}",
                    )
                )

    def _validate_hardcoded_patterns(
        self, node: ast.FunctionDef, func_name: str, line_no: int
    ) -> None:
        """Validate against hardcoded logic patterns."""
        # Check for large if-elif chains
        if_elif_count = self._count_if_elif_chain(node)
        if if_elif_count > 10:
            self.errors.append(
                (
                    line_no,
                    f"Function '{func_name}' has large if-elif chain ({if_elif_count} branches). "
                    f"Consider using registry pattern.",
                )
            )

        # Check for string-based class instantiation
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Check for getattr patterns like getattr(sys.modules[...],
                # classname)
                if (
                    isinstance(child.func, ast.Name)
                    and child.func.id == "getattr"
                    and len(child.args) >= 2
                ):
                    self.errors.append(
                        (
                            line_no,
                            f"Function '{func_name}' uses string-based class instantiation. "
                            f"Use registry lookup instead.",
                        )
                    )

    def _count_if_elif_chain(self, node: ast.FunctionDef) -> int:
        """Count the length of if-elif chains in a function."""

        class IfElifCounter(ast.NodeVisitor):
            def __init__(self):
                self.current_chain = 0
                self.max_chain = 0

            def visit_If(self, node: ast.If) -> None:
                self.current_chain = 1
                current = node

                while hasattr(current, "orelse") and len(current.orelse) == 1:
                    next_node = current.orelse[0]
                    if isinstance(next_node, ast.If):
                        self.current_chain += 1
                        current = next_node
                    else:
                        break

                self.max_chain = max(self.max_chain, self.current_chain)
                self.generic_visit(node)

        counter = IfElifCounter()
        counter.visit(node)
        return counter.max_chain

    def validate_file_patterns(self) -> None:
        """Validate file-level patterns."""
        filepath_str = str(self.filepath)

        # Check if deprecated files still exist
        deprecated_files = {
            "brukerexp.py": "Should be moved to vendors/bruker/",
            "siemensexp.py": "Should be moved to vendors/siemens/",
            "recon_strategy.py": "Should be moved to reconstruction/strategies/",
        }

        for deprecated_file, suggestion in deprecated_files.items():
            if deprecated_file in filepath_str:
                self.errors.append(
                    (1, f"File '{deprecated_file}' is in legacy location. {suggestion}")
                )

        # Check for proper module structure
        if "registry.py" in filepath_str:
            # Registry files should have global registry instances
            has_global_instance = any(
                "_registry" in class_name.lower()
                for class_name, _, _ in self.classes_found
            )
            if not has_global_instance:
                self.errors.append(
                    (1, "Registry files should define global registry instances")
                )


def validate_file(filepath: Path) -> List[Tuple[int, str]]:
    """Validate registry compliance in a single file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        validator = RegistryComplianceValidator(filepath)
        validator.visit(tree)

        # Additional file-level validations
        validator.validate_file_patterns()

        return validator.errors

    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e.msg}")]
    except Exception as e:
        return [(0, f"Error processing file: {str(e)}")]


def main() -> int:
    """Main entry point for pre-commit hook."""
    if len(sys.argv) < 2:
        print("Usage: validate_registry_compliance.py <file1.py> [file2.py ...]")
        return 1

    exit_code = 0

    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)

        if not filepath.suffix == ".py":
            continue

        errors = validate_file(filepath)

        if errors:
            print(f"\n❌ Registry compliance violations in {filepath}:")
            for line_no, message in errors:
                print(f"  Line {line_no}: {message}")
            exit_code = 1

    if exit_code == 0:
        print("✅ All files pass registry compliance checks")
    else:
        print(f"\n💡 See MIGRATION_TO_REGISTRY_SYSTEM.md for registry patterns")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
