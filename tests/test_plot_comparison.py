from pathlib import Path
import os
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box
import xarray as xr

from scripts.plot_swot_validation import olci_coordinates_like


def test_olci_coordinates_broadcast_to_delay_grid() -> None:
    dataset = xr.Dataset(
        {
            "wet_tropo_path_delay": (
                ("y", "x"),
                np.ones((2, 3), dtype=np.float32),
            )
        },
        coords={
            "latitude": ("y", [45.0, 46.0]),
            "longitude": ("x", [350.0, 351.0, 352.0]),
        },
    )
    longitude, latitude = olci_coordinates_like(
        dataset, "wet_tropo_path_delay"
    )
    assert longitude.shape == (2, 3)
    assert latitude.shape == (2, 3)
    assert longitude[0] == pytest.approx([-10.0, -9.0, -8.0])
    assert latitude[:, 0] == pytest.approx([45.0, 46.0])


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed:RuntimeWarning")
def test_three_panel_comparison_plot(tmp_path: Path) -> None:
    swot_path = tmp_path / "swot_subset.nc"
    swot_model_path = tmp_path / "swot_expert.nc"
    olci_path = tmp_path / "olci_delay.nc"
    vignette_path = tmp_path / "vignette.gpkg"
    land_path = tmp_path / "land.gpkg"
    output_path = tmp_path / "comparison.png"

    lines, pixels = 4, 3
    longitude = np.tile(np.linspace(-0.3, 0.3, pixels), (lines, 1))
    latitude = np.tile(np.linspace(-0.3, 0.3, lines)[:, None], (1, pixels))
    quality = np.zeros((lines, pixels), dtype=np.uint32)
    xr.Dataset(
        {
            "longitude_signed": (("line", "pixel"), longitude),
            "latitude": (("line", "pixel"), latitude),
            "ssh_karin_2": (
                ("line", "pixel"),
                1.2 + np.linspace(-0.03, 0.03, lines * pixels).reshape(lines, pixels),
            ),
            "sig0_karin_2": (
                ("line", "pixel"),
                np.linspace(0.08, 0.16, lines * pixels).reshape(lines, pixels),
            ),
            "ssh_karin_2_qual": (("line", "pixel"), quality),
            "sig0_karin_2_qual": (("line", "pixel"), quality),
            "in_vignette": (("line", "pixel"), np.ones((lines, pixels), dtype=bool)),
            "ancillary_surface_classification_flag": (
                ("line", "pixel"),
                np.zeros((lines, pixels), dtype=np.uint8),
            ),
            "time": (
                "line",
                np.arange(
                    np.datetime64("2026-06-17T15:47:00"),
                    np.datetime64("2026-06-17T15:51:00"),
                    np.timedelta64(1, "m"),
                ),
            ),
        },
        attrs={"source_product": "SWOT_L2_LR_SSH_051_533_example.nc"},
    ).to_netcdf(swot_path)

    xr.Dataset(
        {
            "wet_tropo_path_delay": (
                ("y", "x"),
                0.22
                + np.linspace(-0.04, 0.04, 20, dtype=np.float32).reshape(4, 5),
                {"units": "m", "source_variable": "IWV_W"},
            )
        },
        coords={
            "latitude": ("y", np.linspace(-0.35, 0.35, 4)),
            "longitude": ("x", np.linspace(-0.4, 0.4, 5)),
        },
    ).to_netcdf(olci_path)

    xr.Dataset(
        {
            "longitude": (("line", "pixel"), longitude),
            "latitude": (("line", "pixel"), latitude),
            "model_wet_tropo_cor": (
                ("line", "pixel"),
                -0.22
                + np.linspace(0.03, -0.03, lines * pixels).reshape(
                    lines, pixels
                ),
            ),
            "ssh_karin_2_qual": (("line", "pixel"), quality),
            "ancillary_surface_classification_flag": (
                ("line", "pixel"),
                np.zeros((lines, pixels), dtype=np.uint8),
            ),
        }
    ).to_netcdf(swot_model_path)

    gpd.GeoDataFrame(geometry=[box(-0.5, -0.5, 0.5, 0.5)], crs=4326).to_file(
        vignette_path
    )
    gpd.GeoDataFrame(
        geometry=[box(-0.5, -0.5, -0.35, 0.5)], crs=4326
    ).to_file(land_path)

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "plot_swot_validation.py"),
            "--subset",
            str(swot_path),
            "--olci",
            str(olci_path),
            "--swot-model",
            str(swot_model_path),
            "--vignette",
            str(vignette_path),
            "--land",
            str(land_path),
            "--output",
            str(output_path),
            "--title",
            "synthetic coastal matchup",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert output_path.stat().st_size > 20_000
    assert "OLCI_VALID_PIXELS=16" in result.stdout
    assert "SWOT_MODEL_VALID_PIXELS=12" in result.stdout
    assert "SHARED_ANOMALY_LIMIT_M=" in result.stdout
