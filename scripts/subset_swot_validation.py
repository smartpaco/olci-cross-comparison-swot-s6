from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import contains_xy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--vignette-id", required=True)
    parser.add_argument("--swot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-margin-seconds", type=float, default=120.0)
    args = parser.parse_args()

    catalog = gpd.read_file(args.catalog)
    selected = catalog[catalog["vignette_id"] == args.vignette_id].copy()
    if len(selected) != 1:
        raise RuntimeError(f"Vignette {args.vignette_id} introuvable ou dupliquée")
    geometry = selected.geometry.iloc[0]
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    swot_side = str(selected["swot_side"].iloc[0])
    target_time = pd.Timestamp(selected["swot_time"].iloc[0]).to_datetime64()
    margin = np.timedelta64(int(round(args.time_margin_seconds * 1000)), "ms")

    with xr.open_dataset(args.swot, group=swot_side) as source:
        times = source["time"].values
        time_rows = np.flatnonzero((times >= target_time - margin) & (times <= target_time + margin))
        if time_rows.size == 0:
            raise RuntimeError("Le temps de la vignette est hors du granule SWOT")
        probe_slice = slice(
            max(0, int(time_rows.min()) - 2),
            min(source.sizes["num_lines"], int(time_rows.max()) + 3),
        )
        probe = source.isel(num_lines=probe_slice)
        latitude = probe["latitude"].values
        longitude = ((probe["longitude"].values + 180.0) % 360.0) - 180.0
        bbox = (
            (longitude >= min_lon - 0.2)
            & (longitude <= max_lon + 0.2)
            & (latitude >= min_lat - 0.1)
            & (latitude <= max_lat + 0.1)
        )
        rows, columns = np.where(bbox)
        if rows.size == 0:
            raise RuntimeError("Aucun pixel SWOT dans l'emprise de recherche spatio-temporelle")
        row_offset = int(probe_slice.start)
        row_slice = slice(
            max(0, row_offset + int(rows.min()) - 2),
            min(source.sizes["num_lines"], row_offset + int(rows.max()) + 3),
        )
        column_slice = slice(max(0, int(columns.min()) - 2), int(columns.max()) + 3)
        names = [
            "latitude",
            "longitude",
            "time",
            "ssh_karin_2",
            "ssh_karin_2_qual",
            "height_cor_xover",
            "height_cor_xover_qual",
            "ssha_karin_2",
            "ssha_karin_2_qual",
            "sig0_karin_2",
            "sig0_karin_2_qual",
            "ancillary_surface_classification_flag",
        ]
        subset = source[names].isel(num_lines=row_slice, num_pixels=column_slice).load()

    signed_lon = ((subset["longitude"].values + 180.0) % 360.0) - 180.0
    inside = contains_xy(geometry, signed_lon, subset["latitude"].values)
    subset["longitude_signed"] = xr.DataArray(
        signed_lon,
        dims=subset["longitude"].dims,
        attrs={"units": "degrees_east", "long_name": "longitude in [-180, 180]"},
    )
    subset["in_vignette"] = xr.DataArray(
        inside,
        dims=subset["longitude"].dims,
        attrs={"long_name": args.vignette_id},
    )
    subset.attrs.update(
        {
            "vignette_id": args.vignette_id,
            "source_product": Path(args.swot).name,
            "spatial_selection": "exact vignette polygon",
            "swot_side": swot_side,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_netcdf(output)
    selected.to_file(output.with_suffix(".gpkg"), layer="vignette", driver="GPKG")
    selected.drop(columns="geometry").to_csv(output.with_suffix(".csv"), index=False)

    valid_ssha = (
        inside
        & np.isfinite(subset["ssha_karin_2"].values)
        & np.isfinite(subset["height_cor_xover"].values)
        & np.isfinite(subset["ssha_karin_2_qual"].values)
        & (subset["height_cor_xover_qual"].values == 0)
    )
    valid_sig0 = inside & np.isfinite(subset["sig0_karin_2"].values)
    print(f"SWOT_SUBSET={output}")
    print(f"SHAPE={inside.shape} INSIDE={int(inside.sum())}")
    print(
        f"SSHA_XCAL_VALID={int(valid_ssha.sum())} "
        f"SIG0_VALID={int(valid_sig0.sum())}"
    )


if __name__ == "__main__":
    main()
