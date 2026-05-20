"""
Cross-platform Clinical File Organizer
CCI Session 9 - Lesson 6 Lab (Solution)

Mirrors the nightly etl_orchestrator.py at KHCC: organize daily VistA extracts
into a dated archive folder, zip yesterday's archive, and print a summary.

Uses only stdlib — pathlib, shutil, subprocess, datetime — so it runs identically
on Mac, Linux, and Windows.
"""

import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path


def main():
    extracts_dir = Path("extracts")
    archive_root = Path("archive")
    backups_dir = Path("backups")

    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # 1. Create today's archive folder (cross-platform mkdir -p)
    today_folder = archive_root / today
    today_folder.mkdir(parents=True, exist_ok=True)

    # 2. Ensure backups/ exists
    backups_dir.mkdir(parents=True, exist_ok=True)

    # 3. Move every CSV from extracts/ into today's archive folder
    moved_count = 0
    if extracts_dir.exists():
        for csv_file in extracts_dir.glob("*.csv"):
            shutil.move(str(csv_file), str(today_folder / csv_file.name))
            moved_count += 1

    # 4. Zip yesterday's archive (if it exists) into backups/
    yesterday_folder = archive_root / yesterday
    backup_zip_path = None
    backup_size = 0
    if yesterday_folder.exists():
        base_name = str(backups_dir / f"archive_{yesterday}")
        backup_zip_path = Path(shutil.make_archive(
            base_name, "zip",
            root_dir=str(archive_root),
            base_dir=yesterday,
        ))
        backup_size = backup_zip_path.stat().st_size

    # 5. Summary
    print(f"Moved {moved_count} CSV files to {today_folder}")
    if backup_zip_path:
        print(f"Backed up {yesterday_folder} -> {backup_zip_path} ({backup_size} bytes)")
    else:
        print("No yesterday archive to back up.")

    # 6. Bonus: confirm git state at end of run
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            check=True,
        )
        first_line = result.stdout.splitlines()[0] if result.stdout.strip() else "clean"
        print(f"git status: {first_line}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("git not available — skipping git check")


if __name__ == "__main__":
    main()
