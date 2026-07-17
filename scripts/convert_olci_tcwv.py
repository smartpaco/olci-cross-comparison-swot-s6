from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from satmatch.tropo import (
    DEFAULT_MEAN_ATMOSPHERIC_TEMPERATURE_K,
    tcwv_to_zenith_wet_delay,
    wet_delay_factor,
)


TCWV_CANDIDATES = ("IWV_W", "TCWV", "tcwv", "IWV", "iwv")


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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Convert OLCI TCWV/IWV_W to positive one-way zenith wet path delay."
    )
    result.add_argument("--input", required=True, help="Input OLCI TCWV NetCDF file")
    result.add_argument("--output", required=True, help="Output NetCDF file")
    result.add_argument(
        "--tcwv-variable",
        help="TCWV variable name; automatically detects IWV_W or TCWV when omitted",
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
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        argument_parser.error("--output must differ from --input")
    if output_path.exists() and not args.overwrite:
        argument_parser.error(f"Output already exists: {output_path}; use --overwrite")

    with xr.open_dataset(input_path, mask_and_scale=True) as source:
        dataset = source.load()

    try:
        tcwv_name = find_tcwv_variable(dataset, args.tcwv_variable)
    except ValueError as error:
        argument_parser.error(str(error))
    tcwv = dataset[tcwv_name].where(dataset[tcwv_name] >= 0.0)

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
