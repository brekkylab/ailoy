#!/usr/bin/env python
"""
Wrapper script for maturin develop that copies DLLs after the build completes.
This ensures DLLs are copied after _core.pyd is generated.
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Get the directory containing this script (bindings/python)
    project_root = Path(__file__).parent.parent.resolve()

    # Run maturin develop with all passed arguments
    cmd = [sys.executable, "-m", "maturin", "develop"] + sys.argv[1:]

    print("Running maturin develop...")
    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode != 0:
        sys.exit(result.returncode)

    # After successful build, copy DLLs on Windows
    if sys.platform == "win32":
        print("\nCopying DLL dependencies...")

        ailoy_dir = project_root / "ailoy"
        core_pyd = ailoy_dir / "_core.pyd"

        if not core_pyd.exists():
            print(f"Warning: {core_pyd} not found, skipping DLL copy")
            sys.exit(0)

        # Find target directory (root/target/release)
        target_dir = project_root.parent.parent / "target" / "release"
        if not target_dir.exists():
            # Try debug
            target_dir = project_root.parent.parent / "target" / "debug"

        if not target_dir.exists():
            print("Warning: target directory not found, skipping DLL copy")
            sys.exit(0)

        # Run the copy script
        copy_script = project_root / "scripts" / "windows_copy_dlls.py"
        if not copy_script.exists():
            print(f"Warning: {copy_script} not found, skipping DLL copy")
            sys.exit(0)

        copy_cmd = [
            sys.executable,
            str(copy_script),
            str(core_pyd),
            str(ailoy_dir),
            "--add-path",
            str(target_dir),
            "-v",
        ]

        copy_result = subprocess.run(copy_cmd)
        if copy_result.returncode != 0:
            print("Warning: DLL copy failed")

    sys.exit(0)


if __name__ == "__main__":
    main()
