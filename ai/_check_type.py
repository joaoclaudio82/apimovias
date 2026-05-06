import sqlite3

conn = sqlite3.connect("database.db")

# Check distinct features for feature_class '1' and '2'
for fc in ['1', '2']:
    cur = conn.execute(f"SELECT DISTINCT feature FROM vehicle_profile WHERE feature_class='{fc}' ORDER BY feature")
    rows = cur.fetchall()
    print(f"feature_class='{fc}' ({len(rows)} features):")
    for r in rows:
        print(f"  {r[0]}")
    print()

conn.close()
