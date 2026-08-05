#!/usr/bin/env python3
"""
Build and upload UPXO to TestPyPI for verification.
"""

import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ Success: {description}")
    return result.returncode == 0

def main():
    """Build and upload to TestPyPI."""

    repo_root = Path(__file__).parent
    dist_dir = repo_root / "dist"
    build_dir = repo_root / "build"

    # Clean previous builds
    print("🧹 Cleaning previous builds...")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # Step 1: Install/upgrade build tools
    run_command(
        f"{sys.executable} -m pip install --upgrade build twine",
        "Installing build tools (build, twine)"
    )

    # Step 2: Build distribution
    run_command(
        f"cd {repo_root} && {sys.executable} -m build",
        "Building distribution (sdist + wheel)"
    )

    # Step 3: Verify build
    run_command(
        f"{sys.executable} -m twine check {dist_dir}/*",
        "Verifying package metadata"
    )

    # Step 4: Show build artifacts
    print("\n📁 Build artifacts created:")
    for file in sorted(dist_dir.glob("*")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name} ({size_mb:.2f} MB)")

    # Step 5: Upload to TestPyPI
    print(f"\n{'='*60}")
    print("🚀 Uploading to TestPyPI")
    print(f"{'='*60}")
    print("\nYou will be prompted for credentials.")
    print("Use __token__ as username and your TestPyPI API token as password.\n")

    result = subprocess.run(
        f"{sys.executable} -m twine upload --repository testpypi {dist_dir}/*",
        shell=True
    )

    if result.returncode == 0:
        print(f"\n{'='*60}")
        print("✅ Upload successful!")
        print(f"{'='*60}")
        print("\n📍 Verify at: https://test.pypi.org/project/upxo/")
        print("\n🧪 Test installation with:")
        print("   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ upxo==1.1.0")
        return 0
    else:
        print(f"\n{'='*60}")
        print("❌ Upload failed")
        print(f"{'='*60}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
