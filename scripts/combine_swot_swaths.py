from __future__ import annotations

import argparse
from pathlib import Path

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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Combine exact left/right SWOT vignette subsets on one sample axis."
    )
    result.add_argument("--subset", action="append", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if len(args.subset) < 2:
        raise SystemExit("Repeat --subset for at least two SWOT swaths")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}; use --overwrite")

    values: dict[str, list[np.ndarray]] = {
        name: [] for name in (*GRID_VARIABLES, "time")
    }
    sides: list[str] = []
    source_products: set[str] = set()
    variable_attributes: dict[str, dict] = {}
    for path in args.subset:
        with xr.open_dataset(path) as dataset:
            missing = [name for name in GRID_VARIABLES if name not in dataset]
            if missing:
                raise RuntimeError(f"{path} is missing: {', '.join(missing)}")
            side = str(dataset.attrs.get("swot_side", "")).casefold()
            if side not in {"left", "right"}:
                raise RuntimeError(f"{path} has invalid swot_side={side!r}")
            if side in sides:
                raise RuntimeError(f"Duplicate SWOT side: {side}")
            sides.append(side)
            source_products.add(str(dataset.attrs.get("source_product", "")))
            grid = dataset["longitude_signed"]
            for name in GRID_VARIABLES:
                values[name].append(np.asarray(dataset[name].values).reshape(-1))
                variable_attributes.setdefault(name, dict(dataset[name].attrs))
            values["time"].append(
                np.asarray(dataset["time"].broadcast_like(grid).values).reshape(-1)
            )
            variable_attributes.setdefault("time", dict(dataset["time"].attrs))

    if len(source_products) != 1:
        raise RuntimeError("All swaths must come from the same SWOT product")
    combined = xr.Dataset(
        {
            name: xr.DataArray(
                np.concatenate(parts),
                dims=("sample",),
                attrs=variable_attributes.get(name, {}),
            )
            for name, parts in values.items()
        },
        attrs={
            "swot_side": "both",
            "source_product": source_products.pop(),
            "source_swaths": ",".join(sorted(sides)),
            "spatial_selection": "union of exact left/right vignette polygons",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(output)
    print(f"COMBINED_SUBSET={output}")
    print(f"SWOT_SIDES={','.join(sorted(sides))}")
    print(f"SAMPLES={combined.sizes['sample']}")


if __name__ == "__main__":
    main()
