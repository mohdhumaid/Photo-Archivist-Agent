"""Step 8: Offline reverse geocode — LIVE SQLite gazetteer lookup + loader.

Never calls online services: GPS traces must not leave the machine. Without a
gazetteer DB the function degrades to an explicit low-confidence stub.
"""
from __future__ import annotations
import csv
import os
import sqlite3

STUB_SOURCE = "gps_reverse_geocode_offline_stub"
OFFLINE_SOURCE = "gps_reverse_geocode_offline"


def reverse_geocode(lat: float, lon: float, db_path: str | None = None) -> dict:
    """Nearest gazetteer place within ±0.5°, or the offline stub."""
    if db_path and os.path.exists(db_path):
        try:
            c = sqlite3.connect(db_path)
            row = c.execute(
                """SELECT name, admin1, country FROM geonames
                   WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
                   ORDER BY (lat - ?) * (lat - ?) + (lon - ?) * (lon - ?)
                   LIMIT 1""",
                (lat - 0.5, lat + 0.5, lon - 0.5, lon + 0.5,
                 lat, lat, lon, lon)).fetchone()
            c.close()
            if row:
                value = ", ".join(str(x) for x in row if x)
                return {"value": value, "source": OFFLINE_SOURCE,
                        "gps": [lat, lon], "confidence": 0.9}
        except Exception:
            pass
    return {"value": f"{lat:.4f},{lon:.4f}", "source": STUB_SOURCE,
            "gps": [lat, lon], "confidence": 0.5}


def load_gazetteer(tsv_path: str, db_path: str) -> int:
    """Build the SQLite gazetteer from a GeoNames dump (e.g. cities500/cities15000 TSV).

    Columns used: 0 geonameid, 1 name, 4 lat, 5 lon, 8 country, 10 admin1.
    Returns the number of rows loaded.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    c = sqlite3.connect(db_path)
    c.executescript(
        """CREATE TABLE IF NOT EXISTS geonames(
               geoid INTEGER PRIMARY KEY, name TEXT, admin1 TEXT,
               country TEXT, lat REAL, lon REAL);
           CREATE INDEX IF NOT EXISTS idx_geo_latlon ON geonames(lat, lon);""")
    n = 0
    with open(tsv_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 11:
                continue
            try:
                c.execute("INSERT OR REPLACE INTO geonames VALUES(?,?,?,?,?,?)",
                          (int(row[0]), row[1], row[10] or "", row[8],
                           float(row[4]), float(row[5])))
                n += 1
            except Exception:
                continue
    c.commit()
    c.close()
    return n
