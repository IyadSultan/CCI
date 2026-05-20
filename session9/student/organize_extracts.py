"""
Cross-platform Clinical File Organizer
CCI Session 9 - Lesson 6 Lab

Mirrors the nightly etl_orchestrator.py at KHCC: organize daily VistA extracts
into a dated archive folder, zip yesterday's archive, and print a summary.

YOUR TASK: Complete the TODOs using only pathlib, os, shutil, and subprocess —
NO third-party packages. The script must run on Mac, Linux, and Windows.
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

    # TODO 1: Create today's archive folder under archive/
    # Hint: use Path / today, then .mkdir(parents=True, exist_ok=True)
    today_folder = None

    # TODO 2: Make sure backups/ exists
    # Hint: same pattern as above

    # TODO 3: Move every .csv file from extracts/ into today's archive folder
    # Hint: loop over extracts_dir.glob("*.csv"); use shutil.move(str(src), str(dst))
    # Count how many you moved.
    moved_count = 0

    # TODO 4: If archive/yesterday/ exists, zip it into backups/
    # Hint: shutil.make_archive(base_name, "zip", root_dir, base_dir)
    #   base_name = "backups/archive_<yesterday>" (no .zip extension)
    #   root_dir  = "archive"
    #   base_dir  = "<yesterday>"
    backup_zip_path = None
    backup_size = 0

    # TODO 5: Print a clean summary (use the format from the lesson)
    # Moved X CSV files to archive/YYYY-MM-DD
    # Backed up archive/YYYY-MM-DD -> backups/archive_YYYY-MM-DD.zip (N bytes)
    # OR: No yesterday archive to back up.

    # TODO 6 (bonus): Run subprocess.run(["git", "status"]) and print one line of output.
    # Use args as list, capture_output=True, text=True. Catch errors with try/except.


if __name__ == "__main__":
    main()
