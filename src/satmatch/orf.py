from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


ORF_LINE = re.compile(
    r"^(?P<date>\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"(?P<cycle>\d+)\s+(?P<pass>\d+)\s+(?P<rev>\d+)\s+"
    r"(?P<lon>[+-]?\d+(?:\.\d+)?)\s+(?P<lat>[+-]?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class PassWindow:
    """Demi-orbite pôle-à-pôle centrée sur un passage à l'équateur."""

    uid: int
    cycle: int
    pass_number: int
    revolution: int
    start: pd.Timestamp
    equator: pd.Timestamp
    end: pd.Timestamp
    ascending: bool


def read_orf(path: str | Path) -> pd.DataFrame:
    """Lit les lignes de données d'un Orbit Revolution File."""
    records: list[tuple[str, int, int, int, float, float]] = []
    with Path(path).open("rt", encoding="latin-1") as stream:
        for line in stream:
            match = ORF_LINE.match(line)
            if match is None:
                continue
            item = match.groupdict()
            records.append(
                (
                    item["date"],
                    int(item["cycle"]),
                    int(item["pass"]),
                    int(item["rev"]),
                    float(item["lon"]),
                    float(item["lat"]),
                )
            )
    if not records:
        raise ValueError(f"Aucune ligne ORF reconnue dans {path}")

    frame = pd.DataFrame(
        records, columns=["time", "cycle", "pass", "revolution", "lon", "lat"]
    )
    frame["time"] = pd.to_datetime(frame["time"], format="%Y/%m/%d %H:%M:%S.%f", utc=True)
    frame["lon"] = ((frame["lon"] + 180.0) % 360.0) - 180.0
    if not frame["time"].is_monotonic_increasing:
        frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    return frame


def pass_windows(
    events: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> list[PassWindow]:
    """Repère les triplets extrême de latitude / équateur / extrême."""
    lat = events["lat"].to_numpy()
    equators = np.flatnonzero(
        (np.abs(lat) < 1.0)
        & (np.r_[False, np.abs(lat[:-1]) > 50.0])
        & (np.r_[np.abs(lat[1:]) > 50.0, False])
    )
    times = events["time"].array
    if start is not None:
        equators = equators[times[equators + 1] >= start]
    if end is not None:
        equators = equators[times[equators - 1] <= end]
    cycles = events["cycle"].to_numpy()
    passes = events["pass"].to_numpy()
    revolutions = events["revolution"].to_numpy()
    windows: list[PassWindow] = []
    for uid, idx in enumerate(equators):
        windows.append(
            PassWindow(
                uid=uid,
                cycle=int(cycles[idx]),
                pass_number=int(passes[idx]),
                revolution=int(revolutions[idx]),
                start=pd.Timestamp(times[idx - 1]),
                equator=pd.Timestamp(times[idx]),
                end=pd.Timestamp(times[idx + 1]),
                ascending=bool(lat[idx - 1] < lat[idx + 1]),
            )
        )
    if not windows:
        raise ValueError("Aucune demi-orbite complète détectée dans l'ORF")
    return windows


def lonlat_to_unit(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        (cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad))
    )


def unit_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    lon = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lat = np.rad2deg(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    return lon, lat


class OrbitInterpolator:
    """Spline régulière des sous-points ORF sur la sphère terrestre."""

    def __init__(self, events: pd.DataFrame):
        self.epoch = events.iloc[0]["time"]
        seconds = (events["time"] - self.epoch).dt.total_seconds().to_numpy()
        xyz = lonlat_to_unit(events["lon"].to_numpy(), events["lat"].to_numpy())
        self._spline = CubicSpline(seconds, xyz, axis=0)

    def at(self, times: pd.DatetimeIndex | list[pd.Timestamp]) -> tuple[np.ndarray, np.ndarray]:
        index = pd.DatetimeIndex(times)
        seconds = (index - self.epoch).total_seconds().to_numpy()
        return unit_to_lonlat(self._spline(seconds))

    def sample(self, window: PassWindow, step_seconds: float) -> pd.DataFrame:
        duration = (window.end - window.start).total_seconds()
        count = max(2, int(np.ceil(duration / step_seconds)) + 1)
        times = pd.date_range(window.start, window.end, periods=count)
        lon, lat = self.at(times)
        return pd.DataFrame({"time": times, "lon": lon, "lat": lat})


def select_windows(
    windows: list[PassWindow], start: pd.Timestamp | None, end: pd.Timestamp | None
) -> list[PassWindow]:
    return [
        item
        for item in windows
        if (start is None or item.end >= start) and (end is None or item.start <= end)
    ]
