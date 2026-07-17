from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely import box, make_valid, union_all
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.strtree import STRtree


NATURAL_EARTH_LAND_URLS = {
    resolution: f"https://naturalearth.s3.amazonaws.com/{resolution}_physical/ne_{resolution}_land.zip"
    for resolution in ("10m", "50m", "110m")
}


def ensure_land_mask(path: str | Path | None, resolution: str = "50m") -> Path:
    if resolution not in NATURAL_EARTH_LAND_URLS:
        raise ValueError(f"Résolution Natural Earth inconnue: {resolution}")
    target = Path(path) if path else Path(f"data/natural_earth/ne_{resolution}_land.zip")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(NATURAL_EARTH_LAND_URLS[resolution], target)
    return target


class OceanFraction:
    """Fraction océanique calculée comme le complément du masque terrestre choisi."""

    def __init__(self, land_path: str | Path):
        frame = gpd.read_file(land_path).to_crs(4326)
        self._land = np.asarray(frame.geometry.values, dtype=object)
        self._tree = STRtree(self._land)

    @staticmethod
    def _search_boxes(lon: float, lat: float, radius_km: float = 120.0):
        dlat = radius_km / 110.5
        dlon = min(180.0, radius_km / max(12.0, 111.0 * np.cos(np.deg2rad(lat))))
        low, high = lon - dlon, lon + dlon
        ymin, ymax = max(-90.0, lat - dlat), min(90.0, lat + dlat)
        if low < -180.0:
            return [box(-180.0, ymin, high, ymax), box(low + 360.0, ymin, 180.0, ymax)]
        if high > 180.0:
            return [box(low, ymin, 180.0, ymax), box(-180.0, ymin, high - 360.0, ymax)]
        return [box(low, ymin, high, ymax)]

    def percentage(
        self,
        vignette_local: BaseGeometry,
        lon0: float,
        lat0: float,
        to_local: Transformer,
    ) -> float:
        total_area = vignette_local.area
        if total_area <= 0.0:
            return float("nan")

        pieces: list[BaseGeometry] = []
        seen: set[int] = set()
        for search_box in self._search_boxes(lon0, lat0):
            for index in self._tree.query(search_box):
                idx = int(index)
                if idx in seen:
                    continue
                seen.add(idx)
                clipped = self._land[idx].intersection(search_box)
                if not clipped.is_empty:
                    local_piece = make_valid(transform(to_local.transform, clipped))
                    if not local_piece.is_empty:
                        pieces.append(local_piece)
        if not pieces:
            return 100.0

        land_union = make_valid(union_all(pieces))
        land = make_valid(vignette_local).intersection(land_union).area
        return float(np.clip(100.0 * (1.0 - land / total_area), 0.0, 100.0))
