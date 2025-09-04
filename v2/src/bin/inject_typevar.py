#!/usr/bin/env python3
"""
Script to inject TypeVar declarations into pyo3-stub-gen generated .pyi files
"""

import sys
from pathlib import Path


def inject_typevar(file_path: Path, typevar_name: str):
    """
    Inject TypeVar declaration after the last import line in a .pyi file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # TypeVar declaration to inject
    typevar_line = f'{typevar_name} = typing.TypeVar("{typevar_name}")\n'

    # Check if TypeVar already exists
    for line in lines:
        if typevar_line.strip() in line.strip():
            print(f"TypeVar {typevar_name} already exists in {file_path}")
            return False

    # Find the last import line
    last_import_index = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_index = i

    if last_import_index == -1:
        print(f"No import statements found in {file_path}")
        return False

    # Insert TypeVar after the last import, with a blank line
    insert_index = last_import_index + 1
    lines.insert(insert_index, "\n")  # Add blank line
    lines.insert(insert_index + 1, typevar_line)

    # Write back to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Injected TypeVar {typevar_name} into {file_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python inject_typevar.py <file.pyi> <typevar_name>")
        print("Examples:")
        print("  python inject_typevar.py my_module.pyi CacheResultT")
        sys.exit(1)

    path = Path(sys.argv[1])
    typevar_name = sys.argv[2]

    if path.is_file():
        inject_typevar(path, typevar_name)
    else:
        print(f"Error: {path} is not a valid file")
        sys.exit(1)


if __name__ == "__main__":
    main()
