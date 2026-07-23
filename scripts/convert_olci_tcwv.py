from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import contains_xy
import xarray as xr

from satmatch.tropo import (
    DEFAULT_MEAN_ATMOSPHERIC_TEMPERATURE_K,
    tcwv_to_zenith_wet_delay,
    wet_delay_factor,
)


TCWV_CANDIDATES = ("IWV_W", "TCWV", "tcwv", "IWV", "iwv")
LONGITUDE_CANDIDATES = ("longitude", "lon")
LATITUDE_CANDIDATES = ("latitude", "lat")
WQSF_INVALID_FLAGS = (
    "INVALID",
    "LAND",
    "CLOUD",
    "CLOUD_AMBIGUOUS",
    "CLOUD_MARGIN",
    "SNOW_ICE",
    "WV_FAIL",
)


def find_tcwv_variable(dataset: xr.Dataset, requested: str | None = None) -> str:
    if requested is not None:
        if requested not in dataset:
            raise ValueError(f"TCWV variable {requested!r} is absent from the input file")
        return requested

    by_casefold = {name.casefold(): name for name in dataset.data_vars}
    for candidate in TCWV_CANDIDATES:
        if candidate in dataset:
            return candidate
        if candidate.casefold() in by_casefold:
            return by_casefold[candidate.casefold()]
    raise ValueError(
        "No TCWV variable found; use --tcwv-variable to identify the OLCI field"
    )


def find_coordinate_variable(
    dataset: xr.Dataset, candidates: tuple[str, ...], role: str
) -> str:
    available = {name.casefold(): name for name in dataset.variables}
    for candidate in candidates:
        if candidate in dataset.variables:
            return candidate
        if candidate.casefold() in available:
            return available[candidate.casefold()]
    raise ValueError(f"No {role} variable found in the OLCI file")


def load_vignette_subset(
    source: xr.Dataset,
    tcwv_name: str,
    vignette_path: str | Path,
    vignette_ids: list[str] | None = None,
) -> xr.Dataset:
    """Load only the OLCI rows and columns surrounding selected vignettes."""
    vignette = gpd.read_file(vignette_path).to_crs(4326)
    if vignette_ids:
        if "vignette_id" not in vignette:
            raise ValueError("The vignette file has no vignette_id field")
        vignette = vignette[vignette["vignette_id"].isin(vignette_ids)]
        missing = sorted(set(vignette_ids) - set(vignette["vignette_id"]))
        if missing:
            raise ValueError("Unknown vignette IDs: " + ", ".join(missing))
    elif len(vignette) != 1:
        raise ValueError(
            "The vignette file must contain exactly one feature unless "
            "--vignette-id is supplied"
        )
    polygon = vignette.geometry.union_all()
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    longitude_name = find_coordinate_variable(
        source, LONGITUDE_CANDIDATES, "longitude"
    )
    latitude_name = find_coordinate_variable(source, LATITUDE_CANDIDATES, "latitude")
    longitude = np.asarray(source[longitude_name].values, dtype=float)
    longitude = ((longitude + 180.0) % 360.0) - 180.0
    latitude = np.asarray(source[latitude_name].values, dtype=float)
    bbox = (
        (longitude >= min_lon - 0.05)
        & (longitude <= max_lon + 0.05)
        & (latitude >= min_lat - 0.05)
        & (latitude <= max_lat + 0.05)
    )
    rows, columns = np.where(bbox)
    if rows.size == 0:
        raise ValueError("No OLCI pixels overlap the vignette bounding box")
    row_dimension, column_dimension = source[tcwv_name].dims
    row_slice = slice(max(0, int(rows.min()) - 2), int(rows.max()) + 3)
    column_slice = slice(max(0, int(columns.min()) - 2), int(columns.max()) + 3)
    keep = [tcwv_name, longitude_name, latitude_name]
    keep.extend(
        name
        for name in ("quality_flags", "qi", "unc", "WQSF", "IWV_unc")
        if name in source
    )
    subset = source[keep].isel(
        {row_dimension: row_slice, column_dimension: column_slice}
    ).load()
    subset_longitude = np.asarray(subset[longitude_name].values, dtype=float)
    subset_longitude = ((subset_longitude + 180.0) % 360.0) - 180.0
    subset[longitude_name] = xr.DataArray(
        subset_longitude,
        dims=subset[longitude_name].dims,
        attrs=subset[longitude_name].attrs,
    )
    subset["in_vignette"] = xr.DataArray(
        contains_xy(polygon, subset_longitude, subset[latitude_name].values),
        dims=subset[tcwv_name].dims,
        attrs={"long_name": "pixel centre inside the matchup vignette"},
    )
    subset.attrs.update(source.attrs)
    subset.attrs["spatial_subset"] = Path(vignette_path).name
    if vignette_ids:
        subset.attrs["vignette_ids"] = ",".join(vignette_ids)
    return subset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Convert OLCI TCWV/IWV_W to positive one-way zenith wet path delay."
    )
    result.add_argument("--input", required=True, help="Input OLCI TCWV NetCDF file")
    result.add_argument(
        "--geolocation",
        help=(
            "Optional separate OLCI geolocation NetCDF, for example "
            "geo_coordinates.nc from OL_2_WFR"
        ),
    )
    result.add_argument(
        "--quality-file",
        help=(
            "Optional separate OLCI quality NetCDF containing WQSF; cloud, "
            "invalid, land, snow/ice, and WV-failure pixels are masked"
        ),
    )
    result.add_argument("--output", required=True, help="Output NetCDF file")
    result.add_argument(
        "--vignette",
        help="Optional GeoPackage used to crop the OLCI granule before conversion",
    )
    result.add_argument(
        "--vignette-id",
        action="append",
        help="Vignette ID to include; repeat to convert the union of both SWOT swaths",
    )
    result.add_argument(
        "--tcwv-variable",
        help="TCWV variable name; automatically detects IWV_W, TCWV, or IWV when omitted",
    )
    result.add_argument(
        "--mean-temperature-k",
        type=float,
        default=DEFAULT_MEAN_ATMOSPHERIC_TEMPERATURE_K,
        help="Water-vapour-weighted mean atmospheric temperature Tm (default: 270 K)",
    )
    result.add_argument(
        "--tm-variable",
        help="Optional per-pixel Tm variable in the input file; overrides the scalar value",
    )
    result.add_argument(
        "--output-variable",
        default="wet_tropo_path_delay",
        help="Name of the output wet-delay variable",
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.vignette_id and not args.vignette:
        argument_parser.error("--vignette-id requires --vignette")
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        argument_parser.error("--output must differ from --input")
    if output_path.exists() and not args.overwrite:
        argument_parser.error(f"Output already exists: {output_path}; use --overwrite")

    try:
        sources = [xr.open_dataset(input_path, mask_and_scale=True)]
        try:
            if args.geolocation:
                sources.append(
                    xr.open_dataset(args.geolocation, mask_and_scale=True)
                )
            if args.quality_file:
                sources.append(
                    xr.open_dataset(args.quality_file, mask_and_scale=False)
                )
            source = xr.merge(sources, compat="override", combine_attrs="override")
            tcwv_name = find_tcwv_variable(source, args.tcwv_variable)
            dataset = (
                load_vignette_subset(
                    source, tcwv_name, args.vignette, args.vignette_id
                )
                if args.vignette
                else source.load()
            )
        finally:
            for opened in sources:
                opened.close()
    except ValueError as error:
        argument_parser.error(str(error))
    valid_tcwv = dataset[tcwv_name] >= 0.0
    if "qi" in dataset:
        valid_tcwv &= dataset["qi"] > 0
    if "WQSF" in dataset:
        quality = dataset["WQSF"]
        meanings = str(quality.attrs.get("flag_meanings", "")).split()
        masks = np.asarray(quality.attrs.get("flag_masks", []), dtype=np.uint64)
        if len(meanings) != len(masks):
            argument_parser.error(
                "WQSF flag_meanings and flag_masks metadata are inconsistent"
            )
        by_name = dict(zip(meanings, masks, strict=True))
        missing_flags = [name for name in WQSF_INVALID_FLAGS if name not in by_name]
        if missing_flags:
            argument_parser.error(
                "WQSF is missing required flags: " + ", ".join(missing_flags)
            )
        quality_values = np.asarray(quality.values, dtype=np.uint64)
        invalid_quality = np.zeros(quality_values.shape, dtype=bool)
        for name in WQSF_INVALID_FLAGS:
            invalid_quality |= (quality_values & by_name[name]) != 0
        valid_tcwv &= ~xr.DataArray(
            invalid_quality, dims=quality.dims, coords=quality.coords
        )
    tcwv = dataset[tcwv_name].where(valid_tcwv)

    if args.tm_variable:
        if args.tm_variable not in dataset:
            argument_parser.error(
                f"Tm variable {args.tm_variable!r} is absent from the input file"
            )
        mean_temperature = dataset[args.tm_variable]
        method = f"pixel-wise Tm from {args.tm_variable}"
    else:
        mean_temperature = args.mean_temperature_k
        method = f"constant Tm={args.mean_temperature_k:g} K"

    try:
        delay = tcwv_to_zenith_wet_delay(tcwv, mean_temperature)
        factor = wet_delay_factor(mean_temperature)
    except ValueError as error:
        argument_parser.error(str(error))

    delay = delay.astype(np.float32)
    delay.name = args.output_variable
    delay.attrs = {
        "long_name": "positive one-way zenith wet tropospheric path delay",
        "units": "m",
        "source_variable": tcwv_name,
        "conversion_method": (
            "Bennartz et al. (2017), Appendix A, Eq. A15; " + method
        ),
        "reference": "https://doi.org/10.5194/amt-10-1387-2017",
        "sign_convention": "positive excess propagation path; not an SSH correction sign",
    }
    dataset[args.output_variable] = delay
    dataset.attrs["wet_tropo_conversion"] = delay.attrs["conversion_method"]

    # File-specific read metadata are not valid NetCDF write encodings.
    for variable in dataset.variables.values():
        variable.encoding.pop("source", None)
        variable.encoding.pop("original_shape", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)

    valid_delay = np.asarray(delay.values, dtype=float)
    valid_delay = valid_delay[np.isfinite(valid_delay)]
    factor_values = np.asarray(factor, dtype=float)
    factor_values = factor_values[np.isfinite(factor_values)]
    print(f"INPUT_VARIABLE={tcwv_name}")
    print(f"OUTPUT_VARIABLE={args.output_variable}")
    print(
        "FACTOR_MM_PER_KG_M2="
        f"{float(np.nanmin(factor_values)) * 1000.0:.6f},"
        f"{float(np.nanmedian(factor_values)) * 1000.0:.6f},"
        f"{float(np.nanmax(factor_values)) * 1000.0:.6f}"
    )
    if valid_delay.size:
        print(
            "DELAY_M_MIN_MEDIAN_MAX="
            f"{float(np.nanmin(valid_delay)):.6f},"
            f"{float(np.nanmedian(valid_delay)):.6f},"
            f"{float(np.nanmax(valid_delay)):.6f}"
        )
    print(f"OUTPUT={output_path}")


if __name__ == "__main__":
    main()
