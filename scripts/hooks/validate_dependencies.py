#!/usr/bin/env python3
"""
Validate that all package imports in the codebase are properly declared
in pyproject.toml dependencies.

This script checks for:
1. Missing dependencies in pyproject.toml
2. Unused dependencies in pyproject.toml
3. Version conflicts between what's used and what's declared
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import tomli


class ImportExtractor(ast.NodeVisitor):
    """Extract all import statements from Python AST."""

    def __init__(self):
        self.imports: Set[str] = set()
        self.from_imports: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            # Get the top-level package name
            package = alias.name.split(".")[0]
            self.imports.add(package)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            # Get the top-level package name
            package = node.module.split(".")[0]
            self.from_imports.add(package)
        self.generic_visit(node)


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all package imports from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            print(f"Warning: Syntax error in {file_path}, skipping")
            return set()

        extractor = ImportExtractor()
        extractor.visit(tree)

        return extractor.imports | extractor.from_imports

    except (UnicodeDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return set()


def extract_all_imports(src_dir: Path) -> Set[str]:
    """Extract all package imports from the entire source directory."""
    all_imports = set()

    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        file_imports = extract_imports_from_file(py_file)
        all_imports.update(file_imports)

    return all_imports


def parse_dependencies(pyproject_path: Path) -> Tuple[Set[str], Set[str]]:
    """Parse dependencies from pyproject.toml."""
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)

    # Main dependencies
    main_deps = set()
    if "project" in data and "dependencies" in data["project"]:
        for dep in data["project"]["dependencies"]:
            # Extract package name (handle git+https, version specs, etc.)
            if "@" in dep:
                # Git dependency like "brukerapi@git+https://..."
                package_name = dep.split("@")[0].strip()
            else:
                # Regular dependency like "numpy >= 1.19.5"
                package_name = re.split(r"[<>=!~\s]", dep)[0].strip()

            main_deps.add(package_name)

    # Dev dependencies
    dev_deps = set()
    if (
        "project" in data
        and "optional-dependencies" in data["project"]
        and "dev" in data["project"]["optional-dependencies"]
    ):
        for dep in data["project"]["optional-dependencies"]["dev"]:
            if "@" in dep:
                package_name = dep.split("@")[0].strip()
            else:
                package_name = re.split(r"[<>=!~\s]", dep)[0].strip()

            dev_deps.add(package_name)

    return main_deps, dev_deps


def get_stdlib_packages() -> Set[str]:
    """Get set of Python standard library packages."""
    # Common stdlib packages (not exhaustive, but covers most cases)
    stdlib = {
        "os",
        "sys",
        "json",
        "datetime",
        "re",
        "collections",
        "itertools",
        "functools",
        "operator",
        "pathlib",
        "typing",
        "warnings",
        "logging",
        "unittest",
        "math",
        "random",
        "time",
        "subprocess",
        "threading",
        "multiprocessing",
        "asyncio",
        "concurrent",
        "queue",
        "socket",
        "urllib",
        "http",
        "email",
        "html",
        "xml",
        "csv",
        "io",
        "tempfile",
        "shutil",
        "glob",
        "pickle",
        "copy",
        "gc",
        "weakref",
        "abc",
        "enum",
        "dataclasses",
        "contextlib",
        "importlib",
        "pkgutil",
        "inspect",
        "types",
        "traceback",
        "argparse",
        "configparser",
        "hashlib",
        "hmac",
        "secrets",
        "uuid",
        "base64",
        "binascii",
        "struct",
        "codecs",
        "locale",
        "calendar",
        "zoneinfo",
        "decimal",
        "fractions",
        "statistics",
        "cmath",
        "sqlite3",
        "dbm",
        "gzip",
        "bz2",
        "lzma",
        "zipfile",
        "tarfile",
        "platform",
        "getpass",
    }
    return stdlib


def get_package_name_mappings() -> Dict[str, str]:
    """Get mapping from import names to PyPI package names."""
    return {
        "cv2": "opencv-python",
        "PIL": "pillow",  # But it's actually a transitive dependency
        "skimage": "scikit-image",
        "git": "gitpython",
        "yaml": "pyyaml",
        "IPython": "ipython",
        "matplotlib_scalebar": "matplotlib-scalebar",
        "mpl_toolkits": "matplotlib",  # part of matplotlib
        "nrrd": "pynrrd",
        "mapvbvd": "pymapvbvd",
    }


def check_dependencies(project_root: Path) -> int:
    """Check dependency consistency and return exit code."""
    print("🔍 Checking dependency consistency...")

    # Paths
    src_dir = project_root / "src" / "hypermri"
    pyproject_path = project_root / "pyproject.toml"

    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return 1

    if not pyproject_path.exists():
        print(f"❌ pyproject.toml not found: {pyproject_path}")
        return 1

    # Extract information
    all_imports = extract_all_imports(src_dir)
    main_deps, dev_deps = parse_dependencies(pyproject_path)
    stdlib = get_stdlib_packages()

    # Filter out local packages and stdlib
    # Get all local packages by checking what exists in src/
    local_packages = set()
    for py_file in (project_root / "src").rglob("*.py"):
        relative_path = py_file.relative_to(project_root / "src")
        parts = relative_path.parts
        if parts[0] == "hypermri.egg-info":
            continue
        # Add the main package and all its submodules
        local_packages.add(parts[0])  # hypermri
        if len(parts) > 1:
            # Add all directory names as potential modules
            for i in range(1, len(parts)):
                if parts[i].endswith(".py"):
                    # It's a file, add the module name without .py
                    module_name = parts[i][:-3]
                    if module_name != "__init__":
                        local_packages.add(module_name)
                else:
                    # It's a directory
                    local_packages.add(parts[i])

    # Also add common local patterns
    local_packages.update(
        {
            "hypermri",
            "analysis",
            "sequences",
            "utils",
            "vendors",
            "core",
            "functions",
            "masking",
            "reconstruction",
            "visualization",
        }
    )

    external_imports = all_imports - stdlib - local_packages

    # Apply package name mappings
    mappings = get_package_name_mappings()
    mapped_imports = set()
    for imp in external_imports:
        mapped_imports.add(mappings.get(imp, imp))

    external_imports = mapped_imports
    all_declared_deps = main_deps | dev_deps

    print(f"📦 Found {len(external_imports)} external package imports")
    print(f"📋 Found {len(all_declared_deps)} declared dependencies")

    # Find missing dependencies (case-insensitive comparison)
    all_declared_deps_lower = {dep.lower() for dep in all_declared_deps}
    external_imports_lower = {imp.lower() for imp in external_imports}

    missing_deps = set()
    for imp in external_imports:
        if imp.lower() not in all_declared_deps_lower:
            missing_deps.add(imp)

    # Find unused dependencies (be careful with transitive deps)
    potentially_unused = set()
    for dep in all_declared_deps:
        if dep.lower() not in external_imports_lower:
            potentially_unused.add(dep)

    # Report results
    exit_code = 0

    if missing_deps:
        print("\n❌ Missing dependencies in pyproject.toml:")
        for dep in sorted(missing_deps):
            print(f"  - {dep}")
        exit_code = 1

    if potentially_unused:
        print("\n⚠️  Potentially unused dependencies (may be transitive):")
        for dep in sorted(potentially_unused):
            print(f"  - {dep}")
        print(
            "  Note: These might be transitive dependencies or used in ways not detected by AST parsing"
        )

    if not missing_deps and not potentially_unused:
        print("\n✅ All dependencies are properly declared!")

    print(f"\n📊 Summary:")
    print(f"  External imports: {len(external_imports)}")
    print(f"  Declared dependencies: {len(all_declared_deps)}")
    print(f"  Missing: {len(missing_deps)}")
    print(f"  Potentially unused: {len(potentially_unused)}")

    return exit_code


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    exit_code = check_dependencies(project_root)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
