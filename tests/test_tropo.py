from pathlib import Path
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box
import xarray as xr

from satmatch.tropo import tcwv_to_zenith_wet_delay, wet_delay_factor
from scripts.convert_olci_tcwv import find_tcwv_variable, load_vignette_subset


def test_default_wet_delay_factor_is_about_6_4_mm_per_kg_m2() -> None:
    assert wet_delay_factor() * 1000.0 == pytest.approx(6.38812, abs=1.0e-5)


def test_tcwv_to_wet_delay_preserves_amplitude_and_missing_values() -> None:
    tcwv = np.array([0.0, 10.0, 50.0, np.nan])
    delay = tcwv_to_zenith_wet_delay(tcwv)
    assert delay[:3] == pytest.approx([0.0, 0.0638812, 0.319406], abs=1.0e-6)
    assert np.isnan(delay[3])


def test_temperature_dependent_conversion() -> None:
    cold = tcwv_to_zenith_wet_delay(20.0, 250.0)
    warm = tcwv_to_zenith_wet_delay(20.0, 290.0)
    assert cold > warm


def test_find_iwv_w_variable() -> None:
    dataset = xr.Dataset({"IWV_W": (("y", "x"), np.ones((2, 3)))})
    assert find_tcwv_variable(dataset) == "IWV_W"


def test_load_vignette_subset_crops_large_olci_grid(tmp_path: Path) -> None:
    rows, columns = 8, 10
    longitude = np.tile(np.linspace(-2.0, 2.0, columns), (rows, 1))
    latitude = np.tile(np.linspace(40.0, 44.0, rows)[:, None], (1, columns))
    dataset = xr.Dataset(
        {
            "tcwv": (("rows", "columns"), np.full((rows, columns), 20.0)),
            "lon": (("rows", "columns"), longitude),
            "lat": (("rows", "columns"), latitude),
            "qi": (("rows", "columns"), np.full((rows, columns), 2)),
        }
    )
    vignette_path = tmp_path / "vignette.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(-0.3, 41.5, 0.3, 42.5)], crs=4326
    ).to_file(vignette_path)

    subset = load_vignette_subset(dataset, "tcwv", vignette_path)

    assert subset.sizes["rows"] < rows
    assert subset.sizes["columns"] < columns
    assert int(subset["in_vignette"].sum()) > 0


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed:RuntimeWarning")
def test_conversion_script_writes_wet_delay(tmp_path: Path) -> None:
    input_path = tmp_path / "TCWV.nc"
    output_path = tmp_path / "TCWV_with_wet_delay.nc"
    xr.Dataset(
        {"IWV_W": (("y", "x"), np.array([[10.0, 20.0]], dtype=np.float32))}
    ).to_netcdf(input_path)

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "convert_olci_tcwv.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with xr.open_dataset(output_path) as converted:
        assert converted["wet_tropo_path_delay"].attrs["units"] == "m"
        assert converted["wet_tropo_path_delay"].values[0] == pytest.approx(
            [0.0638812, 0.1277624], abs=1.0e-6
        )


def test_conversion_supports_wfr_geolocation_and_wqsf(tmp_path: Path) -> None:
    input_path = tmp_path / "iwv.nc"
    geolocation_path = tmp_path / "geo_coordinates.nc"
    quality_path = tmp_path / "wqsf.nc"
    output_path = tmp_path / "IWV_with_wet_delay.nc"
    xr.Dataset(
        {"IWV": (("rows", "columns"), np.full((2, 2), 10.0, dtype=np.float32))}
    ).to_netcdf(input_path)
    xr.Dataset(
        {
            "longitude": (("rows", "columns"), [[-1.0, 0.0], [-1.0, 0.0]]),
            "latitude": (("rows", "columns"), [[40.0, 40.0], [41.0, 41.0]]),
        }
    ).to_netcdf(geolocation_path)
    meanings = "INVALID LAND CLOUD CLOUD_AMBIGUOUS CLOUD_MARGIN SNOW_ICE WV_FAIL"
    masks = np.array([1, 2, 4, 8, 16, 32, 64], dtype=np.uint64)
    xr.Dataset(
        {
            "WQSF": (
                ("rows", "columns"),
                np.array([[0, 4], [0, 0]], dtype=np.uint64),
                {"flag_meanings": meanings, "flag_masks": masks},
            )
        }
    ).to_netcdf(quality_path)

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "convert_olci_tcwv.py"),
            "--input",
            str(input_path),
            "--geolocation",
            str(geolocation_path),
            "--quality-file",
            str(quality_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with xr.open_dataset(output_path) as converted:
        delay = converted["wet_tropo_path_delay"].values
        assert np.isfinite(delay).sum() == 3
        assert np.isnan(delay[0, 1])
        assert converted["wet_tropo_path_delay"].attrs["source_variable"] == "IWV"
