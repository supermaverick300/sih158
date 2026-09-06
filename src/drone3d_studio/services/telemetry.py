"""Explicit timestamped WGS84 camera positions; never invent missing GPS."""
import csv
from pathlib import Path
import numpy as np
from drone3d_studio.domain.models import GPSFix


def load_csv(path: Path):
    """time_s is relative recording time; altitude_m is WGS84 ellipsoidal height."""
    fixes = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"time_s", "latitude", "longitude", "altitude_m"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires columns: time_s, latitude, longitude, altitude_m. Use absolute camera altitude, not height above takeoff.")
        for row_number, row in enumerate(reader, 2):
            try:
                fixes.append(GPSFix(time=float(row["time_s"]), latitude=float(row["latitude"]), longitude=float(row["longitude"]), altitude_m=float(row["altitude_m"])))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Telemetry row {row_number}: {exc}") from exc
    if len(fixes) < 3:
        raise ValueError("Telemetry requires at least three timestamped positions.")
    if any(b.time <= a.time for a, b in zip(fixes, fixes[1:])):
        raise ValueError("Telemetry time_s must be strictly increasing, with no duplicates.")
    return fixes


def at_time(fixes, seconds, max_gap=2):
    if not fixes or seconds < fixes[0].time or seconds > fixes[-1].time:
        return None
    index = int(np.searchsorted([f.time for f in fixes], seconds))
    if fixes[index].time == seconds:
        return fixes[index].model_copy(deep=True)
    a, b = fixes[index - 1:index + 1]
    if b.time - a.time > max_gap:
        return None
    weight = (seconds - a.time) / (b.time - a.time)
    # Interpolate longitude across the short arc at the antimeridian.
    longitude = (a.longitude + weight * ((b.longitude - a.longitude + 180) % 360 - 180) + 180) % 360 - 180
    return GPSFix(time=seconds, latitude=a.latitude + weight * (b.latitude - a.latitude), longitude=longitude, altitude_m=a.altitude_m + weight * (b.altitude_m - a.altitude_m))


def ecef(fix):
    lat, lon = np.radians([fix.latitude, fix.longitude])
    a, eccentricity = 6378137., 6.69437999014e-3
    n = a / np.sqrt(1 - eccentricity * np.sin(lat) ** 2)
    return np.array([(n + fix.altitude_m) * np.cos(lat) * np.cos(lon), (n + fix.altitude_m) * np.cos(lat) * np.sin(lon), (n * (1 - eccentricity) + fix.altitude_m) * np.sin(lat)])


def distance(a, b):
    return float(np.linalg.norm(ecef(a) - ecef(b)))


def references(frames, path):
    tagged = [(i, f.gps) for i, f in enumerate(frames) if f.gps is not None]
    if len(tagged) < 3:
        raise ValueError("GPS alignment requires at least three accepted, GPS-tagged frames. Import synchronized telemetry and analyze again.")
    centers = np.array([ecef(f) for _, f in tagged])
    singular = np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)
    if singular[0] < 1 or singular[1] < max(.1, singular[0] * .001):
        raise ValueError("GPS positions have insufficient spread or are nearly collinear. Alignment requires a trajectory with lateral variation.")
    path.write_text("".join(f"{i:06d}.jpg {f.latitude:.10f} {f.longitude:.10f} {f.altitude_m:.6f}\n" for i, f in tagged), encoding="utf-8")
    return tagged[0][1]
