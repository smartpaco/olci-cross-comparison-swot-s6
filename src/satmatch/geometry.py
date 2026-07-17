from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Transformer
from scipy.spatial import cKDTree
from shapely import LineString, make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from .ocean import OceanFraction
from .orf import PassWindow, lonlat_to_unit, unit_to_lonlat


EARTH_RADIUS_M = 6_371_008.8
GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class GeometryOptions:
    olci_half_swath_km: float = 635.0
    karin_inner_km: float = 10.0
    karin_outer_km: float = 60.0
    max_along_track_km: float = 50.0
    max_dt_minutes: float = 30.0
    min_area_km2: float = 0.0
    min_ocean_percent: float = 0.0
    min_latitude: float = -66.0
    max_latitude: float = 66.0


def cumulative_distance(frame: pd.DataFrame) -> np.ndarray:
    lon = frame["lon"].to_numpy()
    lat = frame["lat"].to_numpy()
    _, _, distance = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return np.r_[0.0, np.cumsum(distance)]


def interpolate_track(frame: pd.DataFrame, distances: np.ndarray) -> pd.DataFrame:
    cumulative = cumulative_distance(frame)
    xyz = lonlat_to_unit(frame["lon"].to_numpy(), frame["lat"].to_numpy())
    out = np.column_stack([np.interp(distances, cumulative, xyz[:, i]) for i in range(3)])
    lon, lat = unit_to_lonlat(out)
    time_ns = frame["time"].astype("int64").to_numpy()
    values = np.interp(distances, cumulative, time_ns).astype("int64")
    return pd.DataFrame({"time": pd.to_datetime(values, utc=True), "lon": lon, "lat": lat})


def candidate_ranges(
    s3: pd.DataFrame, swot: pd.DataFrame, options: GeometryOptions
) -> list[tuple[float, float]]:
    s3_spacing = float(np.median(np.diff(cumulative_distance(s3))))
    swot_spacing = float(np.median(np.diff(cumulative_distance(swot))))
    sampling_padding_m = 0.5 * (s3_spacing + swot_spacing) + 5_000.0
    radius_m = (
        (options.olci_half_swath_km + options.karin_outer_km) * 1000.0
        + sampling_padding_m
    )
    chord = 2.0 * math.sin(radius_m / (2.0 * EARTH_RADIUS_M))
    tree = cKDTree(lonlat_to_unit(s3["lon"].to_numpy(), s3["lat"].to_numpy()))
    neighbors = tree.query_ball_point(
        lonlat_to_unit(swot["lon"].to_numpy(), swot["lat"].to_numpy()), chord
    )
    s3_time_ns = s3["time"].astype("int64").to_numpy()
    swot_time_ns = swot["time"].astype("int64").to_numpy()
    s3_step_s = float(np.median(np.diff(s3_time_ns))) / 1e9
    swot_step_s = float(np.median(np.diff(swot_time_ns))) / 1e9
    time_limit_ns = int(
        (options.max_dt_minutes * 60.0 + s3_step_s + swot_step_s) * 1e9
    )
    valid = np.fromiter(
        (
            bool(indices)
            and bool(np.any(np.abs(s3_time_ns[np.asarray(indices)] - swot_time_ns[index]) <= time_limit_ns))
            for index, indices in enumerate(neighbors)
        ),
        dtype=bool,
        count=len(neighbors),
    )
    swot_latitude = swot["lat"].to_numpy()
    valid &= (
        (swot_latitude >= options.min_latitude)
        & (swot_latitude <= options.max_latitude)
    )
    if not valid.any():
        return []

    along = cumulative_distance(swot)
    spacing = float(np.median(np.diff(along)))
    indices = np.flatnonzero(valid)
    breaks = np.flatnonzero(np.diff(indices) > 2) + 1
    groups = np.split(indices, breaks)
    ranges = [
        (max(0.0, along[group[0]] - spacing), min(along[-1], along[group[-1]] + spacing))
        for group in groups
    ]
    merged: list[tuple[float, float]] = []
    for start, end in ranges:
        if merged and start - merged[-1][1] < 2.0 * spacing:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def passes_may_overlap(
    s3: pd.DataFrame,
    swot: pd.DataFrame,
    options: GeometryOptions,
    sample_seconds: float,
) -> bool:
    """Préfiltre conservateur espace-temps pour une paire de demi-orbites."""
    ground_speed_padding_km = 8.0 * sample_seconds
    radius_m = (
        options.olci_half_swath_km
        + options.karin_outer_km
        + ground_speed_padding_km
    ) * 1000.0
    chord = 2.0 * math.sin(radius_m / (2.0 * EARTH_RADIUS_M))
    allowed = (
        (swot["lat"].to_numpy() >= options.min_latitude)
        & (swot["lat"].to_numpy() <= options.max_latitude)
    )
    if not allowed.any():
        return False
    swot_allowed = swot.loc[allowed]
    tree = cKDTree(lonlat_to_unit(s3["lon"].to_numpy(), s3["lat"].to_numpy()))
    neighbors = tree.query_ball_point(
        lonlat_to_unit(
            swot_allowed["lon"].to_numpy(), swot_allowed["lat"].to_numpy()
        ),
        chord,
    )
    s3_time_ns = s3["time"].astype("int64").to_numpy()
    swot_time_ns = swot_allowed["time"].astype("int64").to_numpy()
    time_limit_ns = int((options.max_dt_minutes * 60.0 + 2.0 * sample_seconds) * 1e9)
    return any(
        bool(indices)
        and bool(np.any(np.abs(s3_time_ns[np.asarray(indices)] - swot_time_ns[index]) <= time_limit_ns))
        for index, indices in enumerate(neighbors)
    )


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if hasattr(geometry, "geoms"):
        return [item for item in geometry.geoms if isinstance(item, Polygon)]
    return []


def _time_at_projection(line: LineString, times: pd.Series, point) -> pd.Timestamp:
    coordinates = np.asarray(line.coords)
    segment = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    distance = line.project(point)
    time_ns = times.astype("int64").to_numpy()
    return pd.to_datetime(int(np.interp(distance, cumulative, time_ns)), utc=True)


def make_vignettes(
    s3: pd.DataFrame,
    swot: pd.DataFrame,
    s3_pass: PassWindow,
    swot_pass: PassWindow,
    options: GeometryOptions,
    ocean: OceanFraction,
) -> list[dict]:
    results: list[dict] = []
    s3_along = cumulative_distance(s3)
    max_length = options.max_along_track_km * 1000.0

    for range_start, range_end in candidate_ranges(s3, swot, options):
        count = max(1, math.ceil((range_end - range_start) / max_length))
        edges = np.linspace(range_start, range_end, count + 1)
        for along_start, along_end in zip(edges[:-1], edges[1:]):
            swot_piece = interpolate_track(swot, np.linspace(along_start, along_end, 9))
            center = swot_piece.iloc[len(swot_piece) // 2]
            lon0, lat0 = float(center["lon"]), float(center["lat"])
            if lat0 < options.min_latitude or lat0 > options.max_latitude:
                continue
            local_crs = CRS.from_proj4(
                f"+proj=aeqd +lat_0={lat0:.10f} +lon_0={lon0:.10f} +datum=WGS84 +units=m +no_defs"
            )
            to_local = Transformer.from_crs(4326, local_crs, always_xy=True)
            to_wgs84 = Transformer.from_crs(local_crs, 4326, always_xy=True)

            center_xyz = lonlat_to_unit(np.array([lon0]), np.array([lat0]))[0]
            s3_xyz = lonlat_to_unit(s3["lon"].to_numpy(), s3["lat"].to_numpy())
            chord = np.linalg.norm(s3_xyz - center_xyz, axis=1)
            nearby = np.flatnonzero(chord < 2.0 * math.sin(1_500_000.0 / (2.0 * EARTH_RADIUS_M)))
            if nearby.size < 2:
                continue
            i0, i1 = max(0, nearby[0] - 1), min(len(s3) - 1, nearby[-1] + 1)
            dense_s = np.linspace(s3_along[i0], s3_along[i1], max(12, (i1 - i0) * 2))
            s3_piece = interpolate_track(s3, dense_s)

            s3_line = LineString(
                [to_local.transform(lon, lat) for lon, lat in zip(s3_piece["lon"], s3_piece["lat"])]
            )
            swot_line = LineString(
                [to_local.transform(lon, lat) for lon, lat in zip(swot_piece["lon"], swot_piece["lat"])]
            )
            if s3_line.length == 0.0 or swot_line.length == 0.0:
                continue

            olci_fov = s3_line.buffer(options.olci_half_swath_km * 1000.0, cap_style="flat")
            karin_fovs = {
                "left": swot_line.buffer(
                    options.karin_outer_km * 1000.0, single_sided=True
                ).difference(
                    swot_line.buffer(options.karin_inner_km * 1000.0, single_sided=True)
                ),
                "right": swot_line.buffer(
                    -options.karin_outer_km * 1000.0, single_sided=True
                ).difference(
                    swot_line.buffer(-options.karin_inner_km * 1000.0, single_sided=True)
                ),
            }
            for swot_side, karin_fov in karin_fovs.items():
                overlap = olci_fov.intersection(karin_fov)
                for part in _polygon_parts(overlap):
                    if part.area < max(1_000.0, options.min_area_km2 * 1_000_000.0):
                        continue
                    centroid = part.centroid
                    s3_time = _time_at_projection(s3_line, s3_piece["time"], centroid)
                    swot_time = _time_at_projection(swot_line, swot_piece["time"], centroid)
                    dt_seconds = abs((s3_time - swot_time).total_seconds())
                    if dt_seconds > options.max_dt_minutes * 60.0:
                        continue
                    ocean_percent = ocean.percentage(part, lon0, lat0, to_local)
                    if ocean_percent < options.min_ocean_percent:
                        continue
                    output_geometry = make_valid(transform(to_wgs84.transform, part))
                    results.append(
                        {
                        "s3_cycle": s3_pass.cycle,
                        "s3_pass": s3_pass.pass_number,
                        "s3_revolution": s3_pass.revolution,
                        "s3_ascending": s3_pass.ascending,
                        "swot_cycle": swot_pass.cycle,
                        "swot_pass": swot_pass.pass_number,
                        "swot_revolution": swot_pass.revolution,
                        "swot_ascending": swot_pass.ascending,
                        "s3_time": s3_time,
                        "swot_time": swot_time,
                        "dt_minutes": dt_seconds / 60.0,
                        "swot_side": swot_side,
                        "along_start_km": along_start / 1000.0,
                        "along_end_km": along_end / 1000.0,
                        "area_km2": part.area / 1_000_000.0,
                        "ocean_percent": ocean_percent,
                        "ocean_area_km2": part.area * ocean_percent / 100_000_000.0,
                        "center_lon": float(output_geometry.centroid.x),
                        "center_lat": float(output_geometry.centroid.y),
                        "olci_half_swath_km": options.olci_half_swath_km,
                        "karin_inner_km": options.karin_inner_km,
                        "karin_outer_km": options.karin_outer_km,
                            "geometry": output_geometry,
                        }
                    )
    return results
