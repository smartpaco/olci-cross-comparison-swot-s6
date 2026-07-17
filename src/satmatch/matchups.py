from __future__ import annotations

from collections.abc import Iterator

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from .geometry import GeometryOptions, make_vignettes, passes_may_overlap
from .ocean import OceanFraction
from .orf import OrbitInterpolator, PassWindow, pass_windows, read_orf


def candidate_pass_pairs(
    s3: list[PassWindow], swot: list[PassWindow], max_dt_minutes: float
) -> Iterator[tuple[PassWindow, PassWindow]]:
    margin = pd.Timedelta(minutes=max_dt_minutes)
    left = 0
    for swot_pass in swot:
        while left < len(s3) and s3[left].end < swot_pass.start - margin:
            left += 1
        cursor = left
        while cursor < len(s3) and s3[cursor].start <= swot_pass.end + margin:
            yield s3[cursor], swot_pass
            cursor += 1


def find_matchups(
    s3_orf: str,
    swot_orf: str,
    land_mask: str,
    options: GeometryOptions,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    sample_seconds: float = 5.0,
    prefilter_seconds: float = 60.0,
) -> gpd.GeoDataFrame:
    s3_events, swot_events = read_orf(s3_orf), read_orf(swot_orf)
    s3_interp, swot_interp = OrbitInterpolator(s3_events), OrbitInterpolator(swot_events)
    s3_passes = pass_windows(s3_events, start, end)
    swot_passes = pass_windows(swot_events, start, end)
    pairs = list(candidate_pass_pairs(s3_passes, swot_passes, options.max_dt_minutes))
    ocean = OceanFraction(land_mask)

    records: list[dict] = []
    coarse_s3: dict[int, pd.DataFrame] = {}
    coarse_swot: dict[int, pd.DataFrame] = {}
    rejected = 0
    for s3_pass, swot_pass in tqdm(pairs, desc="Paires de demi-orbites"):
        if prefilter_seconds > 0.0:
            if s3_pass.uid not in coarse_s3:
                coarse_s3[s3_pass.uid] = s3_interp.sample(s3_pass, prefilter_seconds)
            if swot_pass.uid not in coarse_swot:
                coarse_swot[swot_pass.uid] = swot_interp.sample(swot_pass, prefilter_seconds)
            s3_coarse = coarse_s3[s3_pass.uid]
            swot_coarse = coarse_swot[swot_pass.uid]
            if not passes_may_overlap(
                s3_coarse, swot_coarse, options, prefilter_seconds
            ):
                rejected += 1
                continue
        s3_track = s3_interp.sample(s3_pass, sample_seconds)
        swot_track = swot_interp.sample(swot_pass, sample_seconds)
        records.extend(
            make_vignettes(s3_track, swot_track, s3_pass, swot_pass, options, ocean)
        )
    if pairs:
        print(f"Préfiltre espace-temps: {rejected}/{len(pairs)} paires rejetées")
    if not records:
        return gpd.GeoDataFrame(
            columns=["s3_time", "swot_time", "dt_minutes", "area_km2", "ocean_percent", "geometry"],
            geometry="geometry",
            crs=4326,
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=4326).sort_values("swot_time")
