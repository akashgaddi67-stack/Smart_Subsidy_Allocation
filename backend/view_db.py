import sqlite3
import json

conn = sqlite3.connect("applications.db")
c = conn.cursor()

c.execute("SELECT * FROM applications")
rows = c.fetchall()

print("\n=== APPLICATIONS TABLE ===\n")

for row in rows:
    print(row)

conn.close()
