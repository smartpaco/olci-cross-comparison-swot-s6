from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
import pytest
from shapely.geometry import box
import xarray as xr

from scripts.screen_olci_clear_sky import (
    browse_clear_percent,
    load_scene_catalogs,
    rectangle_dimensions_km,
    scene_passes_clear_sky,
    wqsf_clear_percent,
)


def test_rectangle_dimensions_are_reported_in_km() -> None:
    geometry = box(-0.18, -0.135, 0.18, 0.135)
    frame = gpd.GeoDataFrame(geometry=[geometry], crs=4326)

    width, length = rectangle_dimensions_km(frame)

    assert width[0] == pytest.approx(29.85, abs=0.3)
    assert length[0] == pytest.approx(40.08, abs=0.3)


def write_tie_file(path: Path) -> None:
    longitude = np.tile(np.array([-1.5, 1.5]), (4, 1))
    latitude = np.tile(np.linspace(1.5, -1.5, 4)[:, None], (1, 2))
    xr.Dataset(
        {
            "longitude": (("tie_rows", "tie_columns"), longitude),
            "latitude": (("tie_rows", "tie_columns"), latitude),
        },
        attrs={"ac_subsampling_factor": 3},
    ).to_netcdf(path)


def test_browse_clear_percent_uses_white_as_unavailable(tmp_path: Path) -> None:
    tie_path = tmp_path / "tie.nc"
    browse_path = tmp_path / "browse.jpg"
    write_tie_file(tie_path)
    image = np.full((4, 4, 3), 80, dtype=np.uint8)
    image[0, 0] = 255
    Image.fromarray(image).save(browse_path, quality=100, subsampling=0)

    percent, pixels = browse_clear_percent(
        box(-2.0, -2.0, 2.0, 2.0), browse_path, tie_path, 245
    )

    assert pixels == 16
    assert percent == pytest.approx(93.75)


def test_wqsf_clear_percent_masks_cloud_flags(tmp_path: Path) -> None:
    tie_path = tmp_path / "tie.nc"
    quality_path = tmp_path / "wqsf.nc"
    write_tie_file(tie_path)
    meanings = "INVALID LAND CLOUD CLOUD_AMBIGUOUS CLOUD_MARGIN SNOW_ICE WV_FAIL"
    masks = np.array([1, 2, 4, 8, 16, 32, 64], dtype=np.uint64)
    quality = np.zeros((4, 4), dtype=np.uint64)
    quality[0, 0] = 4
    xr.Dataset(
        {
            "WQSF": (
                ("rows", "columns"),
                quality,
                {"flag_meanings": meanings, "flag_masks": masks},
            )
        }
    ).to_netcdf(quality_path)

    percent, pixels, clear = wqsf_clear_percent(
        box(-2.0, -2.0, 2.0, 2.0), quality_path, tie_path
    )

    assert pixels == 16
    assert clear == 15
    assert percent == pytest.approx(93.75)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (95.0, 20.0, True),
        (20.0, 95.0, True),
        (95.0, 96.0, True),
        (89.9, 89.9, False),
        (float("nan"), 91.0, True),
        (float("nan"), float("nan"), False),
    ],
)
def test_complete_scene_passes_when_either_swath_is_clear(
    left: float, right: float, expected: bool
) -> None:
    assert scene_passes_clear_sky(left, right, 90.0) is expected


def test_load_scene_catalog_requires_two_swaths(tmp_path: Path) -> None:
    path = tmp_path / "paired.gpkg"
    scenes = gpd.GeoDataFrame(
        {"vignette_id": ["scene-1"]},
        geometry=[box(-2.0, -1.0, 2.0, 1.0)],
        crs=4326,
    )
    swaths = gpd.GeoDataFrame(
        {
            "vignette_id": ["scene-1", "scene-1"],
            "swot_side": ["left", "right"],
        },
        geometry=[
            box(-2.0, -1.0, -0.2, 1.0),
            box(0.2, -1.0, 2.0, 1.0),
        ],
        crs=4326,
    )
    scenes.to_file(path, layer="vignettes", driver="GPKG")
    swaths.to_file(path, layer="swaths", driver="GPKG", mode="a")

    loaded_scenes, loaded_swaths = load_scene_catalogs([str(path)])

    assert len(loaded_scenes) == 1
    assert set(loaded_swaths["swot_side"]) == {"left", "right"}
    assert loaded_scenes.iloc[0]["_scene_key"] == loaded_swaths.iloc[0]["_scene_key"]
