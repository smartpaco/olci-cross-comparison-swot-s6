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
def test_four_panel_comparison_plot(tmp_path: Path) -> None:
    swot_path = tmp_path / "swot_subset.nc"
    swot_expert_path = tmp_path / "swot_expert.nc"
    olci_path = tmp_path / "olci_delay.nc"
    vignette_path = tmp_path / "vignette.gpkg"
    land_path = tmp_path / "land.gpkg"
    output_path = tmp_path / "comparison.png"

    lines, pixels = 4, 3
    longitude = np.tile(np.linspace(-0.3, 0.3, pixels), (lines, 1))
    latitude = np.tile(np.linspace(-0.3, 0.3, lines)[:, None], (1, pixels))
    quality = np.zeros((lines, pixels), dtype=np.uint32)
    amr_quality = quality.copy()
    amr_quality[1, :] = np.uint32(1 << 28)
    radiometer_surface = np.zeros((lines, 2), dtype=np.uint8)
    radiometer_surface[0, 0] = 2
    xr.Dataset(
        {
            "longitude_signed": (("line", "pixel"), longitude),
            "latitude": (("line", "pixel"), latitude),
            "ssh_karin_2": (
                ("line", "pixel"),
                1.2 + np.linspace(-0.03, 0.03, lines * pixels).reshape(lines, pixels),
            ),
            "ssha_karin_2": (
                ("line", "pixel"),
                0.05
                + np.linspace(-0.03, 0.03, lines * pixels).reshape(lines, pixels),
            ),
            "sig0_karin_2": (
                ("line", "pixel"),
                np.linspace(0.08, 0.16, lines * pixels).reshape(lines, pixels),
            ),
            "ssh_karin_2_qual": (("line", "pixel"), quality),
            "ssha_karin_2_qual": (("line", "pixel"), quality),
            "height_cor_xover": (
                ("line", "pixel"),
                np.full((lines, pixels), 0.01, dtype=np.float32),
            ),
            "height_cor_xover_qual": (
                ("line", "pixel"),
                np.zeros((lines, pixels), dtype=np.uint8),
            ),
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
        attrs={
            "source_product": "SWOT_L2_LR_SSH_051_533_example.nc",
            "swot_side": "left",
        },
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
            "rad_wet_tropo_cor": (
                ("line", "pixel"),
                -0.22
                + np.linspace(0.03, -0.03, lines * pixels).reshape(
                    lines, pixels
                ),
            ),
            "cross_track_distance": (
                ("line", "pixel"),
                np.tile(np.array([-20_000.0, -10_000.0, 10_000.0]), (lines, 1)),
            ),
            "ssh_karin_qual": (("line", "pixel"), amr_quality),
            "ancillary_surface_classification_flag": (
                ("line", "pixel"),
                np.zeros((lines, pixels), dtype=np.uint8),
            ),
            "rad_surface_type_flag": (
                ("line", "side"),
                radiometer_surface,
            ),
        }
    ).to_netcdf(swot_expert_path)

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
            "--swot-expert",
            str(swot_expert_path),
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
    assert "SSHA_REFERENCE_M=0.060000" in result.stdout
    assert "XCAL_REFERENCE_M=0.010000" in result.stdout
    assert "XCAL_GOOD_PIXELS=12" in result.stdout
    assert "OLCI_VALID_PIXELS=16" in result.stdout
    assert "OLCI_CLEAR_SKY_PERCENT=100.000000" in result.stdout
    assert "SWOT_AMR_SIDE=left" in result.stdout
    assert "SWOT_AMR_VALID_PIXELS=4" in result.stdout
    assert "SHARED_ANOMALY_LIMIT_M=" in result.stdout
    assert "WET_DELAY_PLOT_LIMIT_M=" in result.stdout
