"""Step 8+offline geocode stub + Step 8 embed interface."""
from __future__ import annotations


def reverse_geocode(lat: float, lon: float, db_path: str | None = None) -> dict:
    """Offline reverse geocode. Without a gazetteer DB, return coords with low conf.

    A GeoNames offline dump can back this later; never call online services
    (GPS traces must not leave the machine).
    """
    return {"value": f"{lat:.4f},{lon:.4f}", "source": "gps_reverse_geocode_offline_stub",
            "gps": [lat, lon], "confidence": 0.5}
