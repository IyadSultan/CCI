"""Initialize the nephrotoxic drugs database."""

import sqlite3

conn = sqlite3.connect("drugs.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS nephrotoxic_drugs (
    name TEXT PRIMARY KEY,
    class TEXT,
    crcl_threshold INTEGER,
    note TEXT
)
""")

drugs = [
    ("Vancomycin",   "Glycopeptide",    50, "Adjust dose if CrCl < 50"),
    ("Gentamicin",   "Aminoglycoside",  60, "Avoid if CrCl < 30"),
    ("Cisplatin",    "Chemotherapy",    60, "Hold if CrCl < 60"),
    ("Methotrexate", "Antimetabolite",  60, "Reduce dose if CrCl < 60"),
    ("Acyclovir",    "Antiviral",       50, "Adjust dose if CrCl < 50"),
]

cur.executemany("INSERT OR REPLACE INTO nephrotoxic_drugs VALUES (?, ?, ?, ?)", drugs)
conn.commit()
conn.close()
print("Database initialized.")
