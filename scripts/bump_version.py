#!/usr/bin/env python3
"""
Version bumping script for GridGulp.

Usage:
    python scripts/bump_version.py patch  # 0.1.0 -> 0.1.1
    python scripts/bump_version.py minor  # 0.1.0 -> 0.2.0
    python scripts/bump_version.py major  # 0.1.0 -> 1.0.0
    python scripts/bump_version.py 1.2.3  # Set specific version
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_current_version() -> str:
    """Get the current version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    match = re.search(r'version = "(\d+\.\d+\.\d+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")

    return match.group(1)


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse version string into tuple of integers."""
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}")

    return tuple(map(int, parts))


def bump_version(current: str, bump_type: str) -> str:
    """Bump version based on type (major, minor, patch) or specific version."""
    # Check if bump_type is a specific version
    if re.match(r"\d+\.\d+\.\d+", bump_type):
        return bump_type

    major, minor, patch = parse_version(current)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_file(file_path: Path, old_version: str, new_version: str) -> bool:
    """Update version in a file."""
    if not file_path.exists():
        return False

    content = file_path.read_text()
    updated_content = content.replace(old_version, new_version)

    if content != updated_content:
        file_path.write_text(updated_content)
        return True
    return False


def update_changelog(new_version: str) -> None:
    """Update CHANGELOG.md with new version."""
    changelog_path = Path(__file__).parent.parent / "CHANGELOG.md"
    if not changelog_path.exists():
        return

    content = changelog_path.read_text()
    today = datetime.now().strftime("%Y-%m-%d")

    # Replace [Unreleased] with new version
    updated_content = re.sub(
        r"## \[Unreleased\]",
        f"## [Unreleased]\n\n## [{new_version}] - {today}",
        content,
        count=1,
    )

    # Update links at bottom
    updated_content = re.sub(
        r"\[Unreleased\]: (.+)/compare/(.+?)\.\.\.HEAD",
        f"[Unreleased]: \\1/compare/v{new_version}...HEAD\n[{new_version}]: \\1/compare/\\2...v{new_version}",
        updated_content,
    )

    changelog_path.write_text(updated_content)


def git_operations(old_version: str, new_version: str, commit: bool = True) -> None:
    """Perform git operations for version bump."""
    if not commit:
        return

    try:
        # Check if git is available and we're in a git repo
        subprocess.run(["git", "status"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Not in a git repository or git not available")
        return

    # Stage changed files
    files_to_stage = [
        "pyproject.toml",
        "CHANGELOG.md",
        "src/gridgulp/__init__.py",
    ]

    for file in files_to_stage:
        file_path = Path(__file__).parent.parent / file
        if file_path.exists():
            subprocess.run(["git", "add", str(file_path)], check=True)

    # Create commit
    commit_message = f"Bump version from {old_version} to {new_version}"
    subprocess.run(["git", "commit", "-m", commit_message], check=True)

    # Create tag
    tag_name = f"v{new_version}"
    subprocess.run(["git", "tag", tag_name], check=True)

    print("\nGit commit and tag created. To push:")
    print("  git push origin main")
    print(f"  git push origin {tag_name}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    bump_type = sys.argv[1]

    try:
        # Get current version
        old_version = get_current_version()
        print(f"Current version: {old_version}")

        # Calculate new version
        new_version = bump_version(old_version, bump_type)
        print(f"New version: {new_version}")

        # Update files
        files_updated = []

        # Update pyproject.toml
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if update_file(pyproject_path, f'version = "{old_version}"', f'version = "{new_version}"'):
            files_updated.append("pyproject.toml")

        # Update __init__.py if it exists
        init_path = Path(__file__).parent.parent / "src" / "gridgulp" / "__init__.py"
        if init_path.exists():
            # Create __init__.py with version if it doesn't have one
            content = init_path.read_text()
            if "__version__" not in content:
                content = (
                    f'"""GridGulp - Intelligent spreadsheet ingestion framework."""\n\n__version__ = "{new_version}"\n'
                    + content
                )
                init_path.write_text(content)
                files_updated.append("src/gridgulp/__init__.py")
            else:
                if update_file(
                    init_path,
                    f'__version__ = "{old_version}"',
                    f'__version__ = "{new_version}"',
                ):
                    files_updated.append("src/gridgulp/__init__.py")

        # Update CHANGELOG.md
        update_changelog(new_version)
        files_updated.append("CHANGELOG.md")

        print(f"\nFiles updated: {', '.join(files_updated)}")

        # Git operations
        git_operations(old_version, new_version)

        print(f"\nVersion bumped successfully from {old_version} to {new_version}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
