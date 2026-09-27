"""
Trims framework dependencies for a lightweight Kaggle install.

Removes unused modules from the arc-agi / arcengine packages after
installation to reduce the install footprint and speed up notebook startup.

Usage:
  python scripts/slim_framework.py
"""

import os
import sys
import shutil
from pathlib import Path


def find_package_location(pkg_name: str) -> Path | None:
    """Find where a package is installed."""
    try:
        import importlib
        mod = importlib.import_module(pkg_name)
        if hasattr(mod, "__file__") and mod.__file__:
            return Path(mod.__file__).parent
    except ImportError:
        pass

    # Search common locations
    for prefix in [sys.prefix, os.path.expanduser("~/.local")]:
        for lib in ["lib/python3.12/site-packages", "site-packages"]:
            p = Path(prefix) / lib / pkg_name
            if p.exists():
                return p
    return None


def slim_arcagi() -> None:
    """Remove heavy/unused parts of arc-agi to reduce install size."""
    pkg = find_package_location("arcengine")
    if not pkg:
        print("arcengine not found — skipping slim")
        return

    removed = []

    # Remove test/fixture directories if they exist
    for subdir in ["tests", "test", "fixtures", "examples", "docs"]:
        d = pkg / subdir
        if d.exists():
            shutil.rmtree(d)
            removed.append(str(d))

    # Remove large data files
    for pattern in ["*.gif", "*.mp4", "*.woff", "*.ttf"]:
        for f in pkg.rglob(pattern):
            f.unlink()
            removed.append(str(f))

    if removed:
        print(f"Removed {len(removed)} items from arcengine")
    else:
        print("Nothing to slim in arcengine")


def main():
    slim_arcagi()
    print("Slim complete.")


if __name__ == "__main__":
    main()
