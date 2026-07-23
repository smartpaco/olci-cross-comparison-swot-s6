from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr


GRID_VARIABLES = (
    "longitude_signed",
    "latitude",
    "ssh_karin_2",
    "ssh_karin_2_qual",
    "ssha_karin_2",
    "ssha_karin_2_qual",
    "height_cor_xover",
    "height_cor_xover_qual",
    "sig0_karin_2",
    "sig0_karin_2_qual",
    "in_vignette",
    "ancillary_surface_classification_flag",
)


def write_subset(path: Path, side: str, longitude_offset: float) -> None:
    shape = (2, 3)
    variables = {}
    for name in GRID_VARIABLES:
        if name == "longitude_signed":
            values = np.full(shape, longitude_offset, dtype=np.float32)
        elif name == "latitude":
            values = np.zeros(shape, dtype=np.float32)
        elif name == "in_vignette":
            values = np.ones(shape, dtype=bool)
        elif name.endswith("_qual"):
            values = np.zeros(shape, dtype=np.uint32)
        else:
            values = np.ones(shape, dtype=np.float32)
        variables[name] = (("line", "pixel"), values)
    variables["time"] = (
        "line",
        np.array(
            ["2026-06-11T15:15:10", "2026-06-11T15:15:11"],
            dtype="datetime64[s]",
        ),
    )
    xr.Dataset(
        variables,
        attrs={"swot_side": side, "source_product": "shared_source.nc"},
    ).to_netcdf(path)


def test_combine_swot_swaths_preserves_native_samples(tmp_path: Path) -> None:
    left = tmp_path / "left.nc"
    right = tmp_path / "right.nc"
    output = tmp_path / "both.nc"
    write_subset(left, "left", -1.0)
    write_subset(right, "right", 1.0)

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "combine_swot_swaths.py"),
            "--subset",
            str(left),
            "--subset",
            str(right),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with xr.open_dataset(output) as combined:
        assert combined.attrs["swot_side"] == "both"
        assert combined.attrs["source_swaths"] == "left,right"
        assert combined.sizes["sample"] == 12
        assert np.sum(combined["longitude_signed"].values < 0.0) == 6
        assert np.sum(combined["longitude_signed"].values > 0.0) == 6
