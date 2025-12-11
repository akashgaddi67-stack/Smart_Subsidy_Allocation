import sqlite3
import os
import shutil

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "applications.db")

if not os.path.exists(DB_PATH):
    print("No applications.db found — nothing to dedupe.")
    raise SystemExit(0)

# Make a backup before modifying
bak = DB_PATH + ".bak"
shutil.copyfile(DB_PATH, bak)
print("Backup created at:", bak)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Check if duplicates exist
c.execute("SELECT aadhaar, COUNT(*) FROM applications GROUP BY aadhaar HAVING COUNT(*)>1")
dups = c.fetchall()
print("Duplicate aadhaar groups found:", len(dups))

# Strategy: keep the latest row per aadhaar (by created_at)
for row in dups:
    aadhaar = row[0]
    # fetch ids for this aadhaar ordered by created_at desc
    c.execute("SELECT id FROM applications WHERE aadhaar=? ORDER BY created_at DESC", (aadhaar,))
    ids = [r[0] for r in c.fetchall()]
    # keep first, delete the rest
    to_delete = ids[1:]
    if to_delete:
        placeholders = ",".join("?" for _ in to_delete)
        c.execute(f"DELETE FROM applications WHERE id IN ({placeholders})", to_delete)
        print(f"Deduped {aadhaar}: kept {ids[0]}, removed {len(to_delete)} older rows")

conn.commit()

# Now create unique index (if not exists)
try:
    c.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_app_unique_aadhaar ON applications(aadhaar);"
    )
    conn.commit()
    print("Unique index created (or already existed).")
except Exception as e:
    print("Failed to create unique index:", e)

conn.close()
print("Dedupe run complete.")
