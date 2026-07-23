from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely import Polygon, contains_xy, make_valid
import xarray as xr


GEOD = Geod(ellps="WGS84")
SCIENCE_VARIABLES = (
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
    "cross_track_distance",
)


def signed_longitude(values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=float) + 180.0) % 360.0) - 180.0


def native_footprint(
    longitude: np.ndarray, latitude: np.ndarray, selected: np.ndarray
):
    """Build a footprint from the inner and outer pixel-centre boundaries."""
    valid_rows = np.flatnonzero(selected.any(axis=1))
    if valid_rows.size < 2:
        raise ValueError("At least two native SWOT lines are required")
    first = []
    last = []
    for row in valid_rows:
        columns = np.flatnonzero(selected[row])
        first.append((longitude[row, columns[0]], latitude[row, columns[0]]))
        last.append((longitude[row, columns[-1]], latitude[row, columns[-1]]))
    geometry = make_valid(Polygon(first + last[::-1]))
    if geometry.geom_type == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda item: item.area)
    return geometry


def representative_spacing_m(
    longitude: np.ndarray,
    latitude: np.ndarray,
    cross_track_distance: np.ndarray,
) -> float:
    _, _, distance = GEOD.inv(
        longitude[:-1],
        latitude[:-1],
        longitude[1:],
        latitude[1:],
    )
    valid = (
        np.isfinite(distance)
        & np.isfinite(cross_track_distance[:-1])
        & np.isfinite(cross_track_distance[1:])
    )
    return float(np.nanmedian(distance[valid]))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Replace ORF-based SWOT polygons by native KaRIn geolocation and "
            "write one topology-preserving subset per selected swath."
        )
    )
    result.add_argument("--catalog", required=True)
    result.add_argument("--vignette-id", action="append", required=True)
    result.add_argument("--swot", required=True)
    result.add_argument("--output-vignette", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--time-margin-seconds", type=float, default=120.0)
    return result


def main() -> None:
    args = parser().parse_args()
    catalog = gpd.read_file(args.catalog).to_crs(4326)
    if "vignette_id" not in catalog:
        raise SystemExit("The catalogue has no vignette_id field")
    selected_rows = catalog[catalog["vignette_id"].isin(args.vignette_id)].copy()
    missing = sorted(set(args.vignette_id) - set(selected_rows["vignette_id"]))
    if missing:
        raise SystemExit("Unknown vignette IDs: " + ", ".join(missing))
    if selected_rows["swot_side"].duplicated().any():
        raise SystemExit("Select at most one catalogue feature per SWOT side")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    refined_records: list[dict] = []

    for record in selected_rows.itertuples():
        side = str(record.swot_side).casefold()
        if side not in {"left", "right"}:
            raise SystemExit(f"Unsupported SWOT side: {side!r}")
        target_time = pd.Timestamp(record.swot_time).to_datetime64()
        margin = np.timedelta64(
            int(round(args.time_margin_seconds * 1000.0)), "ms"
        )
        with xr.open_dataset(args.swot, group=side) as source:
            times = source["time"].values
            time_rows = np.flatnonzero(
                (times >= target_time - margin) & (times <= target_time + margin)
            )
            if time_rows.size == 0:
                raise RuntimeError(
                    f"{record.vignette_id}: target time is outside the SWOT granule"
                )
            probe_slice = slice(int(time_rows.min()), int(time_rows.max()) + 1)
            probe = source[
                ["longitude", "latitude", "cross_track_distance", "time"]
            ].isel(num_lines=probe_slice).load()
            probe_lon = signed_longitude(probe["longitude"].values)
            probe_lat = np.asarray(probe["latitude"].values, dtype=float)
            seed_inside = contains_xy(record.geometry, probe_lon, probe_lat)
            seed_lines = np.flatnonzero(seed_inside.any(axis=1))
            if seed_lines.size == 0:
                raise RuntimeError(
                    f"{record.vignette_id}: the ORF polygon does not touch the "
                    f"native {side} swath within the time margin"
                )

            centre_probe_line = int(np.rint(np.median(seed_lines)))
            centre_source_line = int(probe_slice.start) + centre_probe_line
            spacing_m = representative_spacing_m(
                probe_lon,
                probe_lat,
                np.asarray(probe["cross_track_distance"].values, dtype=float),
            )
            along_length_m = (
                float(record.along_end_km) - float(record.along_start_km)
            ) * 1000.0
            half_lines = max(1, int(np.rint(along_length_m / spacing_m / 2.0)))
            row_slice = slice(
                max(0, centre_source_line - half_lines),
                min(source.sizes["num_lines"], centre_source_line + half_lines + 1),
            )

            broad = source[list(SCIENCE_VARIABLES)].isel(
                num_lines=row_slice
            ).load()
            cross_track = np.asarray(
                broad["cross_track_distance"].values, dtype=float
            )
            inner_m = float(record.karin_inner_km) * 1000.0
            outer_m = float(record.karin_outer_km) * 1000.0
            native_mask = (
                (np.abs(cross_track) >= inner_m)
                & (np.abs(cross_track) <= outer_m)
                & np.isfinite(cross_track)
            )
            native_columns = np.flatnonzero(native_mask.any(axis=0))
            if native_columns.size == 0:
                raise RuntimeError(
                    f"{record.vignette_id}: no native pixels in the requested "
                    "cross-track interval"
                )
            column_slice = slice(
                int(native_columns.min()), int(native_columns.max()) + 1
            )
            subset = broad.isel(num_pixels=column_slice)
            native_mask = native_mask[:, column_slice]

        longitude = signed_longitude(subset["longitude"].values)
        latitude = np.asarray(subset["latitude"].values, dtype=float)
        native_mask &= np.isfinite(longitude) & np.isfinite(latitude)
        geometry = native_footprint(longitude, latitude, native_mask)
        subset["longitude_signed"] = xr.DataArray(
            longitude,
            dims=subset["longitude"].dims,
            attrs={"units": "degrees_east", "long_name": "longitude in [-180, 180]"},
        )
        subset["in_vignette"] = xr.DataArray(
            native_mask,
            dims=subset["longitude"].dims,
            attrs={
                "long_name": (
                    f"native {side} KaRIn swath between "
                    f"{record.karin_inner_km:g} and {record.karin_outer_km:g} km"
                )
            },
        )
        subset.attrs.update(
            {
                "vignette_id": record.vignette_id,
                "source_product": Path(args.swot).name,
                "spatial_selection": "native KaRIn geolocation",
                "swot_side": side,
                "native_track_refinement": "true",
                "seed_catalog": Path(args.catalog).name,
                "native_along_track_spacing_m": spacing_m,
            }
        )
        output_subset = output_dir / f"swot_subset_{side}_native.nc"
        subset.to_netcdf(output_subset)

        output_record = selected_rows[
            selected_rows["vignette_id"] == record.vignette_id
        ].iloc[0].drop(labels="geometry").to_dict()
        output_record.update(
            {
                "geometry": geometry,
                "area_km2": np.nan,
                "center_lon": float(geometry.centroid.x),
                "center_lat": float(geometry.centroid.y),
                "geometry_source": "native KaRIn geolocation",
            }
        )
        refined_records.append(output_record)
        print(f"SWOT_{side.upper()}_SUBSET={output_subset}")
        print(f"SWOT_{side.upper()}_NATIVE_PIXELS={int(native_mask.sum())}")

    refined = gpd.GeoDataFrame(refined_records, crs=4326)
    areas = refined.to_crs(refined.estimate_utm_crs()).area / 1_000_000.0
    refined["area_km2"] = areas.to_numpy()
    output_vignette = Path(args.output_vignette)
    output_vignette.parent.mkdir(parents=True, exist_ok=True)
    refined.to_file(output_vignette, layer="vignettes", driver="GPKG")
    refined.drop(columns="geometry").to_csv(
        output_vignette.with_suffix(".csv"), index=False
    )
    print(f"NATIVE_VIGNETTES={output_vignette}")


if __name__ == "__main__":
    main()
