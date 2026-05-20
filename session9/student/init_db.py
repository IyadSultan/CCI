"""
Initialize the nephrotoxic drugs database.

YOUR TASK: Complete the script to create drugs.db with 5 nephrotoxic drugs.
"""

import sqlite3

# TODO: Connect to drugs.db (it will be created if missing)
# conn = sqlite3.connect(...)
# cur = conn.cursor()

# TODO: Create the nephrotoxic_drugs table with columns:
# - name (TEXT PRIMARY KEY)
# - class (TEXT)
# - crcl_threshold (INTEGER)
# - note (TEXT)
# Use CREATE TABLE IF NOT EXISTS

# TODO: Insert these 5 drugs:
# ("Vancomycin",   "Glycopeptide",    50, "Adjust dose if CrCl < 50")
# ("Gentamicin",   "Aminoglycoside",  60, "Avoid if CrCl < 30")
# ("Cisplatin",    "Chemotherapy",    60, "Hold if CrCl < 60")
# ("Methotrexate", "Antimetabolite",  60, "Reduce dose if CrCl < 60")
# ("Acyclovir",    "Antiviral",       50, "Adjust dose if CrCl < 50")
# Hint: Use cur.executemany() with "INSERT OR REPLACE INTO ..."

# TODO: Commit and close the connection

print("Database initialized.")
