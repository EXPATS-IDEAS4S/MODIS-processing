from __future__ import annotations

from pathlib import Path
import datetime as dt


def _parse_time_from_name(path: Path) -> dt.datetime:
    parts = path.stem.split(".")
    # Example: MOD021KM.A2024123.1050.061.2024123133333
    if len(parts) < 3:
        return dt.datetime.utcnow()
    julian = parts[1][1:]
    hhmm = parts[2]
    year = int(julian[:4])
    doy = int(julian[4:])
    hour = int(hhmm[:2])
    minute = int(hhmm[2:4])
    return dt.datetime(year, 1, 1, hour=hour, minute=minute) + dt.timedelta(days=doy - 1)
