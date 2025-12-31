#!/usr/bin/env python
"""Copy DLL dependencies for Windows Python packages (using delvewheel's filtering)."""

import os
import re
import shutil
import sys
from collections import deque
from pathlib import Path

try:
    from delvewheel import _dll_list, _dll_utils
except ImportError:
    print("Error: delvewheel must be installed", file=sys.stderr)
    sys.exit(1)


def get_python_abi_tag(binary_path: str) -> str:
    """
    Try to detect Python ABI tag from the binary path.
    This is a heuristic - delvewheel gets this from the wheel metadata.
    """
    # Try to get from filename (e.g., _core.cp314-win_amd64.pyd)
    name = Path(binary_path).name
    match = re.search(r"\.cp(\d+)(?:t|d)?-", name)
    if match:
        return f"cp{match.group(1)}"

    # Fallback: use current Python version
    version_info = sys.version_info
    return f"cp{version_info.major}{version_info.minor}"


def get_platform_tag(machine_type: _dll_list.MachineType) -> str:
    """Get platform tag from machine type."""
    if machine_type == _dll_list.MachineType.AMD64:
        return "win_amd64"
    elif machine_type == _dll_list.MachineType.I386:
        return "win32"
    elif machine_type == _dll_list.MachineType.ARM64:
        return "win_arm64"
    return "win_amd64"


def should_ignore_dll(
    dll_name: str, machine_type: _dll_list.MachineType, abi_tag: str, platform_tag: str
) -> bool:
    """
    Check if a DLL should be ignored based on delvewheel's comprehensive rules.
    """
    dll_lower = dll_name.lower()

    # Check against regex patterns (python DLLs, API sets, etc.)
    for pattern in _dll_list.ignore_regexes:
        if pattern.fullmatch(dll_lower):
            return True

    # Check against architecture-specific ignore list (system DLLs)
    if machine_type in _dll_list.ignore_names:
        if dll_lower in _dll_list.ignore_names[machine_type]:
            return True

    # Check against ABI-specific ignore list (MSVC runtimes for specific Python versions)
    abi_platform = f"{abi_tag}-{platform_tag}"
    for pattern_str, ignore_set in _dll_list.ignore_by_abi_platform.items():
        if re.fullmatch(pattern_str, abi_platform):
            if dll_lower in ignore_set:
                return True

    return False


def get_machine_type(binary_path: str) -> _dll_list.MachineType:
    """Determine the machine type of a binary."""
    import pefile

    try:
        pe = pefile.PE(binary_path, fast_load=True)
        machine_type = _dll_list.MachineType.machine_field_to_type(
            pe.FILE_HEADER.Machine
        )
        pe.close()
        return machine_type or _dll_list.MachineType.AMD64
    except Exception:
        return _dll_list.MachineType.AMD64


def find_dll_path(
    dll_name: str, search_paths: list, machine_type: _dll_list.MachineType
) -> str:
    """Find the full path to a DLL by searching in the given paths."""
    # Use delvewheel's find_library function
    result = _dll_utils.find_library(dll_name, None, machine_type, False, False)
    if result:
        return result[0]

    # Fallback: manual search in add_paths
    for search_path in search_paths:
        candidate = os.path.join(search_path, dll_name)
        if os.path.exists(candidate):
            try:
                if _dll_utils.get_arch(candidate) == machine_type:
                    return candidate
            except Exception:
                pass

    return None


def get_all_dependencies_recursive(
    binary_path: str,
    machine_type: _dll_list.MachineType,
    abi_tag: str,
    platform_tag: str,
    add_paths: list = None,
):
    """
    Recursively get all DLL dependencies, mimicking delvewheel's behavior.
    Returns a dict of {dll_name: dll_path}
    """
    add_paths = add_paths or []
    all_deps = {}
    visited = set()
    to_process = deque([binary_path])

    while to_process:
        current_path = to_process.popleft()

        if current_path in visited:
            continue
        visited.add(current_path)

        try:
            direct_deps = _dll_utils.get_direct_needed(current_path)
        except Exception:
            continue

        for dll_name in direct_deps:
            dll_lower = dll_name.lower()

            # Skip if already processed
            if dll_lower in all_deps:
                continue

            # Check if should be ignored
            if should_ignore_dll(dll_name, machine_type, abi_tag, platform_tag):
                continue

            # Find the DLL
            dll_path = find_dll_path(dll_name, add_paths, machine_type)
            all_deps[dll_name] = dll_path

            # Queue for recursive processing if found
            if dll_path:
                to_process.append(dll_path)

    return all_deps


def copy_dlls(
    binary_path: str, dest_dir: str, add_paths: list = None, verbose: bool = False
):
    """
    Copy all required DLL dependencies to the destination directory.

    Args:
        binary_path: Path to the binary (.pyd, .dll, .exe) to analyze
        dest_dir: Destination directory to copy DLLs to
        add_paths: Additional paths to search for DLLs
        verbose: Whether to print verbose output

    Returns:
        Number of DLLs copied, or -1 on error
    """
    binary_path = Path(binary_path).resolve()
    dest_dir = Path(dest_dir).resolve()

    if not binary_path.exists():
        print(f"Error: File not found: {binary_path}", file=sys.stderr)
        return -1

    if not dest_dir.exists():
        print(f"Error: Destination directory not found: {dest_dir}", file=sys.stderr)
        return -1

    # Determine machine type and tags
    machine_type = get_machine_type(str(binary_path))
    abi_tag = get_python_abi_tag(str(binary_path))
    platform_tag = get_platform_tag(machine_type)

    if verbose:
        print(f"Analyzing: {binary_path}")
        print(f"Architecture: {machine_type.name}")
        print(f"ABI tag: {abi_tag}")
        print(f"Platform: {platform_tag}")
        print(f"Destination: {dest_dir}")
        print()

    # Set up search paths in environment
    if add_paths:
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{os.pathsep.join(add_paths)}{os.pathsep}{old_path}"

    # Get all dependencies recursively
    try:
        vendorable_dlls = get_all_dependencies_recursive(
            str(binary_path), machine_type, abi_tag, platform_tag, add_paths
        )
    except Exception as e:
        print(f"Error analyzing dependencies: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return -1

    # Copy each DLL
    copied_count = 0
    for dll_name, dll_path in sorted(
        vendorable_dlls.items(), key=lambda x: x[0].lower()
    ):
        if not dll_path:
            if verbose:
                print(f"Warning: DLL not found: {dll_name}", file=sys.stderr)
            continue

        dest_path = dest_dir / dll_name
        try:
            shutil.copy2(dll_path, dest_path)
            if verbose:
                print(f"Copied: {dll_name} -> {dest_path}")
            copied_count += 1
        except Exception as e:
            print(f"Error copying {dll_name}: {e}", file=sys.stderr)

    if verbose:
        print(f"\nCopied {copied_count} DLL(s)")

    return copied_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Copy DLL dependencies for Windows Python packages (using delvewheel's filtering)"
    )
    parser.add_argument(
        "binary", help="Path to the binary file (.pyd, .dll, .exe) to analyze"
    )
    parser.add_argument("dest_dir", help="Destination directory to copy DLLs to")
    parser.add_argument(
        "--add-path",
        action="append",
        default=[],
        help="Additional paths to search for DLLs",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Prepare add_paths
    add_paths = []
    for path_str in args.add_path:
        for path in path_str.split(os.pathsep):
            if path:
                add_paths.append(os.path.abspath(path))

    result = copy_dlls(args.binary, args.dest_dir, add_paths, args.verbose)
    sys.exit(0 if result >= 0 else 1)
