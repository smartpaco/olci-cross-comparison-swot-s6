from __future__ import annotations

import argparse
from pathlib import Path
import re

import cmocean
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from pyproj import CRS, Transformer
from shapely import contains_xy
from shapely.affinity import scale
import xarray as xr


BAD_NOT_USABLE = np.uint32(1 << 31)
BAD_OUTSIDE_RANGE = np.uint32(1 << 29)
BAD_RADIOMETER_CORR_MISSING = np.uint32(1 << 28)
OLCI_LONGITUDE_CANDIDATES = ("longitude", "lon", "longitude_signed")
OLCI_LATITUDE_CANDIDATES = ("latitude", "lat")


def usable_mask(dataset: xr.Dataset, variable: str, quality: str) -> np.ndarray:
    values = dataset[variable].values
    flags = np.nan_to_num(dataset[quality].values, nan=2**32 - 1).astype(np.uint32)
    return (
        dataset["in_vignette"].values
        & (dataset["ancillary_surface_classification_flag"].values == 0)
        & np.isfinite(values)
        & ((flags & BAD_NOT_USABLE) == 0)
        & ((flags & BAD_OUTSIDE_RANGE) == 0)
    )


def find_dataset_variable(
    dataset: xr.Dataset,
    requested: str | None,
    candidates: tuple[str, ...],
    role: str,
) -> str:
    """Resolve a variable or coordinate name, ignoring case when needed."""
    available = {name.casefold(): name for name in dataset.variables}
    if requested is not None:
        if requested not in dataset.variables:
            raise ValueError(f"{role} variable {requested!r} is absent from the file")
        return requested
    for candidate in candidates:
        if candidate in dataset.variables:
            return candidate
        if candidate.casefold() in available:
            return available[candidate.casefold()]
    raise ValueError(
        f"Unable to find the {role} variable; specify it explicitly on the command line"
    )


def olci_coordinates_like(
    dataset: xr.Dataset,
    data_variable: str,
    longitude_variable: str | None = None,
    latitude_variable: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast 1-D or 2-D OLCI coordinates to the science-variable grid."""
    longitude_name = find_dataset_variable(
        dataset,
        longitude_variable,
        OLCI_LONGITUDE_CANDIDATES,
        "OLCI longitude",
    )
    latitude_name = find_dataset_variable(
        dataset,
        latitude_variable,
        OLCI_LATITUDE_CANDIDATES,
        "OLCI latitude",
    )
    data = dataset[data_variable]
    longitude = dataset[longitude_name].broadcast_like(data)
    latitude = dataset[latitude_name].broadcast_like(data)
    longitude_values = np.asarray(longitude.values, dtype=float)
    longitude_values = ((longitude_values + 180.0) % 360.0) - 180.0
    return longitude_values, np.asarray(latitude.values, dtype=float)


def require_valid(mask: np.ndarray, label: str) -> None:
    if not np.any(mask):
        raise ValueError(f"No valid {label} pixels are available inside the vignette")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Plot colocated XCAL-corrected SWOT SSHA, SWOT Sigma0, "
            "SWOT AMR wet path delay, and OLCI wet path delay."
        )
    )
    result.add_argument("--subset", required=True, help="SWOT vignette NetCDF")
    result.add_argument(
        "--olci",
        required=True,
        help="Converted OLCI NetCDF containing wet_tropo_path_delay",
    )
    result.add_argument(
        "--swot-expert",
        "--swot-model",
        dest="swot_expert",
        required=True,
        help="SWOT L2 LR SSH Expert NetCDF containing rad_wet_tropo_cor",
    )
    result.add_argument("--vignette", required=True, help="Vignette GeoPackage")
    result.add_argument(
        "--vignette-id",
        action="append",
        help="Vignette ID to include; repeat for the left and right swaths",
    )
    result.add_argument("--land", required=True, help="Land polygon dataset")
    result.add_argument("--output", required=True, help="Output figure path")
    result.add_argument("--title", default="coastal matchup")
    result.add_argument(
        "--olci-delay-variable", default="wet_tropo_path_delay"
    )
    result.add_argument(
        "--swot-radiometer-variable",
        "--swot-model-variable",
        dest="swot_radiometer_variable",
        default="rad_wet_tropo_cor",
        help="AMR wet-troposphere correction variable in the Expert product",
    )
    result.add_argument("--olci-longitude-variable")
    result.add_argument("--olci-latitude-variable")
    result.add_argument(
        "--scale-mode",
        choices=("shared", "independent"),
        default="shared",
        help=(
            "shared uses one metre scale for SSHA and both wet delays; "
            "independent enhances patterns with separate robust scales"
        ),
    )
    return result


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()

    vignette = gpd.read_file(args.vignette).to_crs(4326)
    if args.vignette_id:
        if "vignette_id" not in vignette:
            argument_parser.error("The vignette file has no vignette_id field")
        vignette = vignette[vignette["vignette_id"].isin(args.vignette_id)].copy()
        missing = sorted(set(args.vignette_id) - set(vignette["vignette_id"]))
        if missing:
            argument_parser.error("Unknown vignette IDs: " + ", ".join(missing))
    elif len(vignette) != 1:
        argument_parser.error(
            "The vignette file must contain exactly one feature unless "
            "--vignette-id is supplied"
        )
    polygon = vignette.geometry.union_all()
    lon0, lat0 = float(polygon.centroid.x), float(polygon.centroid.y)
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.10f} +lon_0={lon0:.10f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(4326, local_crs, always_xy=True)
    local_vignette = vignette.to_crs(local_crs)
    local_vignette.geometry = local_vignette.geometry.map(
        lambda geometry: scale(
            geometry, xfact=0.001, yfact=0.001, origin=(0.0, 0.0)
        )
    )

    land_wgs84 = gpd.read_file(args.land, bbox=polygon.bounds).to_crs(4326)
    land_wgs84 = land_wgs84.clip(polygon.buffer(1.0))
    land_geometry = (
        land_wgs84.geometry.union_all() if not land_wgs84.empty else None
    )
    land = land_wgs84.to_crs(local_crs)
    land.geometry = land.geometry.map(
        lambda geometry: scale(
            geometry, xfact=0.001, yfact=0.001, origin=(0.0, 0.0)
        )
    )

    with xr.open_dataset(args.subset) as dataset:
        swot_lon = np.asarray(dataset["longitude_signed"].values, dtype=float)
        swot_lat = np.asarray(dataset["latitude"].values, dtype=float)
        swot_x, swot_y = transformer.transform(swot_lon, swot_lat)
        swot_x, swot_y = swot_x / 1000.0, swot_y / 1000.0

        missing_xcal = [
            name
            for name in ("height_cor_xover", "height_cor_xover_qual")
            if name not in dataset
        ]
        if missing_xcal:
            argument_parser.error(
                "SWOT subset is missing XCAL fields: " + ", ".join(missing_xcal)
            )
        ssha_native = np.asarray(dataset["ssha_karin_2"].values, dtype=float)
        height_cor_xover = np.asarray(
            dataset["height_cor_xover"].values, dtype=float
        )
        xcal_quality = np.nan_to_num(
            dataset["height_cor_xover_qual"].values, nan=255
        ).astype(np.uint8)
        xcal_good = np.isfinite(height_cor_xover) & (xcal_quality == 0)
        ssha = ssha_native + height_cor_xover
        sig0 = np.asarray(dataset["sig0_karin_2"].values, dtype=float)
        ssha_mask = (
            usable_mask(dataset, "ssha_karin_2", "ssha_karin_2_qual")
            & xcal_good
        )
        sig0_mask = usable_mask(
            dataset, "sig0_karin_2", "sig0_karin_2_qual"
        ) & (sig0 > 0.0)
        require_valid(ssha_mask, "SWOT SSHA")
        require_valid(sig0_mask, "SWOT Sigma0")

        ssha_reference = float(np.nanmedian(ssha[ssha_mask]))
        ssha_anomaly = ssha - ssha_reference
        xcal_reference = float(np.nanmedian(height_cor_xover[ssha_mask]))
        xcal_anomaly = height_cor_xover - xcal_reference
        xcal_limit = float(
            np.nanpercentile(np.abs(xcal_anomaly[ssha_mask]), 98.0)
        )
        sig0_db = np.full(sig0.shape, np.nan, dtype=float)
        sig0_db[sig0_mask] = 10.0 * np.log10(sig0[sig0_mask])
        sig0_low, sig0_high = np.nanpercentile(sig0_db[sig0_mask], [2.0, 98.0])

        ssha_flags = np.nan_to_num(
            dataset["ssha_karin_2_qual"].values, nan=2**32 - 1
        ).astype(np.uint32)
        degraded = int(
            np.sum(ssha_mask & ((ssha_flags & np.uint32(1 << 30)) != 0))
        )
        time_start = np.datetime_as_string(dataset["time"].values[0], unit="s")
        time_end = np.datetime_as_string(dataset["time"].values[-1], unit="s")
        swot_side = str(dataset.attrs.get("swot_side", "")).casefold()
        if swot_side not in {"left", "right", "both"}:
            argument_parser.error(
                "The SWOT subset must identify its swot_side as left, right, or both"
            )
        source = dataset.attrs.get("source_product", "")
        marker = re.search(r"_(\d{3})_(\d{3})_", source)
        source_label = (
            f"SWOT L2 LR SSH Unsmoothed — cycle {marker.group(1)}, "
            f"pass {marker.group(2)}"
            if marker
            else source
        )

    with xr.open_dataset(args.olci, mask_and_scale=True) as dataset:
        if args.olci_delay_variable not in dataset:
            argument_parser.error(
                f"OLCI delay variable {args.olci_delay_variable!r} is absent; "
                "run convert_olci_tcwv.py first"
            )
        try:
            olci_lon, olci_lat = olci_coordinates_like(
                dataset,
                args.olci_delay_variable,
                args.olci_longitude_variable,
                args.olci_latitude_variable,
            )
        except ValueError as error:
            argument_parser.error(str(error))
        olci_delay = np.asarray(
            dataset[args.olci_delay_variable].values, dtype=float
        )
        olci_source_variable = dataset[args.olci_delay_variable].attrs.get(
            "source_variable", "TCWV"
        )
        olci_conversion_method = dataset[args.olci_delay_variable].attrs.get(
            "conversion_method", "conversion method not recorded"
        )

    olci_inside = contains_xy(polygon, olci_lon, olci_lat)
    olci_on_land = (
        contains_xy(land_geometry, olci_lon, olci_lat)
        if land_geometry is not None
        else np.zeros(olci_lon.shape, dtype=bool)
    )
    olci_ocean_footprint = (
        olci_inside
        & ~olci_on_land
        & np.isfinite(olci_lon)
        & np.isfinite(olci_lat)
    )
    olci_mask = olci_ocean_footprint & np.isfinite(olci_delay)
    require_valid(olci_mask, "OLCI wet path delay")
    olci_clear_sky_percent = (
        100.0 * float(olci_mask.sum()) / float(olci_ocean_footprint.sum())
    )
    olci_x, olci_y = transformer.transform(olci_lon, olci_lat)
    olci_x, olci_y = olci_x / 1000.0, olci_y / 1000.0
    olci_reference = float(np.nanmedian(olci_delay[olci_mask]))
    olci_anomaly = olci_delay - olci_reference

    with xr.open_dataset(args.swot_expert, mask_and_scale=True) as dataset:
        required_amr_variables = (
            args.swot_radiometer_variable,
            "longitude",
            "latitude",
            "cross_track_distance",
            "ssh_karin_qual",
            "ancillary_surface_classification_flag",
            "rad_surface_type_flag",
        )
        missing = [name for name in required_amr_variables if name not in dataset]
        if missing:
            argument_parser.error(
                "SWOT Expert file is missing AMR fields: " + ", ".join(missing)
            )
        amr_lon = np.asarray(dataset["longitude"].values, dtype=float)
        amr_lon = ((amr_lon + 180.0) % 360.0) - 180.0
        amr_lat = np.asarray(dataset["latitude"].values, dtype=float)
        amr_correction = np.asarray(
            dataset[args.swot_radiometer_variable].values, dtype=float
        )
        amr_cross_track = np.asarray(
            dataset["cross_track_distance"].values, dtype=float
        )
        amr_flags = np.nan_to_num(
            dataset["ssh_karin_qual"].values, nan=2**32 - 1
        ).astype(np.uint32)
        amr_surface = np.asarray(
            dataset["ancillary_surface_classification_flag"].values
        )
        radiometer_surface = np.asarray(
            dataset["rad_surface_type_flag"].values
        )

    amr_inside = contains_xy(polygon, amr_lon, amr_lat)
    if swot_side == "left":
        amr_selected_side = amr_cross_track < 0.0
    elif swot_side == "right":
        amr_selected_side = amr_cross_track > 0.0
    else:
        amr_selected_side = amr_cross_track != 0.0
    # The two num_sides columns are left and right, respectively. Reject lines
    # for which the selected AMR footprint is invalid because of land
    # contamination; both open-ocean and coastal-ocean retrievals are retained.
    left_surface_valid = np.isin(radiometer_surface[:, 0], (0, 1))
    right_surface_valid = np.isin(radiometer_surface[:, 1], (0, 1))
    if swot_side == "left":
        amr_surface_valid = left_surface_valid[:, np.newaxis]
    elif swot_side == "right":
        amr_surface_valid = right_surface_valid[:, np.newaxis]
    else:
        amr_surface_valid = (
            ((amr_cross_track < 0.0) & left_surface_valid[:, np.newaxis])
            | ((amr_cross_track > 0.0) & right_surface_valid[:, np.newaxis])
        )
    amr_mask = (
        amr_inside
        & amr_selected_side
        & (amr_surface == 0)
        & amr_surface_valid
        & np.isfinite(amr_lon)
        & np.isfinite(amr_lat)
        & np.isfinite(amr_correction)
        & ((amr_flags & BAD_RADIOMETER_CORR_MISSING) == 0)
        & ((amr_flags & BAD_NOT_USABLE) == 0)
        & ((amr_flags & BAD_OUTSIDE_RANGE) == 0)
    )
    require_valid(amr_mask, "SWOT AMR wet-troposphere correction")
    amr_x, amr_y = transformer.transform(amr_lon, amr_lat)
    amr_x, amr_y = amr_x / 1000.0, amr_y / 1000.0
    # SWOT stores a negative range correction. Negating it gives a positive
    # equivalent vertical wet path delay comparable to the OLCI-derived delay.
    amr_delay = -amr_correction
    amr_reference = float(np.nanmedian(amr_delay[amr_mask]))
    amr_anomaly = amr_delay - amr_reference

    ssha_limit = float(np.nanpercentile(np.abs(ssha_anomaly[ssha_mask]), 98.0))
    olci_limit = float(np.nanpercentile(np.abs(olci_anomaly[olci_mask]), 98.0))
    amr_limit = float(
        np.nanpercentile(np.abs(amr_anomaly[amr_mask]), 98.0)
    )
    shared_limit = max(ssha_limit, olci_limit, amr_limit, 1.0e-6)
    shared_norm = TwoSlopeNorm(
        vmin=-shared_limit, vcenter=0.0, vmax=shared_limit
    )
    if args.scale_mode == "shared":
        ssha_norm = olci_norm = amr_norm = shared_norm
        wet_delay_plot_limit = shared_limit
        scale_description = (
            "SSHA, OLCI delay, and SWOT AMR-delay anomalies share "
            f"±{shared_limit:.3f} m colour limits."
        )
    else:
        ssha_norm = TwoSlopeNorm(
            vmin=-max(ssha_limit, 1.0e-6),
            vcenter=0.0,
            vmax=max(ssha_limit, 1.0e-6),
        )
        olci_norm = TwoSlopeNorm(
            vmin=-max(olci_limit, 1.0e-6),
            vcenter=0.0,
            vmax=max(olci_limit, 1.0e-6),
        )
        # Keep both wet-delay panels on the OLCI amplitude scale so their
        # spatial variations remain directly comparable.
        amr_norm = olci_norm
        wet_delay_plot_limit = olci_limit
        scale_description = (
            "SSHA uses an independent symmetric 98th-percentile scale; OLCI "
            "and SWOT AMR wet delays share the OLCI 98th-percentile scale."
        )

    figure, axes = plt.subplots(1, 4, figsize=(26.0, 7.2), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.045, right=0.99, bottom=0.24, top=0.82, wspace=0.11
    )
    panels = [
        {
            "axis": axes[0],
            "x": swot_x,
            "y": swot_y,
            "mask": ssha_mask,
            "values": ssha_anomaly,
            "cmap": cmocean.cm.balance,
            "norm": ssha_norm,
            "title": (
                "XCAL-corrected SWOT SSHA − median "
                f"({ssha_reference:.3f} m)"
            ),
            "unit": "SSHA anomaly (m)",
            "size": 4.0,
        },
        {
            "axis": axes[1],
            "x": swot_x,
            "y": swot_y,
            "mask": sig0_mask,
            "values": sig0_db,
            "cmap": cmocean.cm.amp,
            "norm": plt.Normalize(vmin=float(sig0_low), vmax=float(sig0_high)),
            "title": "SWOT Sigma0 with model atmospheric correction",
            "unit": "Sigma0 (dB)",
            "size": 4.0,
        },
        {
            "axis": axes[2],
            "x": amr_x,
            "y": amr_y,
            "mask": amr_mask,
            "values": amr_anomaly,
            "cmap": cmocean.cm.balance,
            "norm": amr_norm,
            "title": (
                "SWOT AMR wet path delay − median "
                f"({amr_reference:.3f} m)"
            ),
            "unit": "AMR wet path-delay anomaly (m)",
            "size": 4.0,
            "render": "mesh",
        },
        {
            "axis": axes[3],
            "x": olci_x,
            "y": olci_y,
            "mask": olci_mask,
            "values": olci_anomaly,
            "cmap": cmocean.cm.balance,
            "norm": olci_norm,
            "title": f"OLCI wet path delay − median ({olci_reference:.3f} m)",
            "unit": "Wet path-delay anomaly (m)",
            "size": 5.0,
        },
    ]

    bounds = local_vignette.total_bounds
    for panel in panels:
        axis = panel["axis"]
        if not land.empty:
            land.plot(
                ax=axis,
                facecolor="#d8d4cc",
                edgecolor="#68645e",
                linewidth=0.45,
                zorder=1,
            )
        mask = panel["mask"]
        if panel.get("render") == "mesh":
            # The Expert product is a coarser curvilinear grid. Rendering its
            # native cells produces a continuous image without inventing a
            # finer resolution through interpolation.
            artist = axis.pcolormesh(
                panel["x"],
                panel["y"],
                np.ma.masked_where(~mask, panel["values"]),
                shading="auto",
                linewidth=0,
                cmap=panel["cmap"],
                norm=panel["norm"],
                rasterized=True,
                zorder=2,
            )
        else:
            artist = axis.scatter(
                panel["x"][mask],
                panel["y"][mask],
                c=panel["values"][mask],
                s=panel["size"],
                marker="s",
                linewidths=0,
                cmap=panel["cmap"],
                norm=panel["norm"],
                rasterized=True,
                zorder=2,
            )
        local_vignette.boundary.plot(
            ax=axis, color="#111111", linewidth=1.0, zorder=3
        )
        if len(local_vignette) > 1 and "swot_side" in local_vignette:
            for vignette_row in local_vignette.itertuples():
                centroid = vignette_row.geometry.centroid
                axis.text(
                    centroid.x,
                    centroid.y,
                    str(vignette_row.swot_side).upper(),
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#111111",
                    zorder=4,
                )
        colorbar = figure.colorbar(
            artist, ax=axis, orientation="horizontal", pad=0.14, fraction=0.055
        )
        colorbar.set_label(panel["unit"])
        axis.set_title(panel["title"], fontsize=10.5)
        axis.set_aspect("equal")
        axis.set_xlim(bounds[0] - 3.0, bounds[2] + 3.0)
        axis.set_ylim(bounds[1] - 3.0, bounds[3] + 3.0)
        axis.set_xlabel("Local easting (km)")
        axis.grid(color="#8a8a8a", linewidth=0.35, alpha=0.35)
    axes[0].set_ylabel("Local northing (km)")

    pixel_summary = (
        f"valid pixels: SSHA {int(ssha_mask.sum()):,}, "
        f"Sigma0 {int(sig0_mask.sum()):,}, OLCI delay {int(olci_mask.sum()):,}, "
        f"SWOT AMR delay {int(amr_mask.sum()):,}; "
        f"OLCI clear-sky coverage {olci_clear_sky_percent:.1f}%"
    )
    figure.suptitle(
        f"OLCI–SWOT wet-troposphere comparison — {args.title}\n"
        f"SWOT {time_start} to {time_end} UTC · {pixel_summary}",
        fontsize=13,
        y=0.96,
    )
    figure.text(
        0.5,
        0.025,
        f"Native sensor grids; no spatial resampling. {scale_description} "
        "Sigma0 can contain both surface and atmospheric signatures.\n"
        f"Vignette and open-ocean SWOT quality masks applied; {degraded} degraded "
        "SSHA pixels retained. SSHA = ssha_karin_2 + height_cor_xover; only good "
        f"XCAL retained (median {xcal_reference:.3f} m). OLCI delay source: "
        f"{olci_source_variable}; "
        f"{olci_conversion_method}. "
        f"SWOT AMR delay = −{args.swot_radiometer_variable}; "
        "radiometer land-contaminated lines excluded. "
        f"AMR swath selection: {swot_side}. "
        f"{source_label}",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#444444",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"PLOT={output}")
    print(f"SSHA_REFERENCE_M={ssha_reference:.6f}")
    print(f"XCAL_REFERENCE_M={xcal_reference:.6f}")
    print(f"XCAL_ANOMALY_LIMIT_M={xcal_limit:.6f}")
    print(f"XCAL_GOOD_PIXELS={int(np.sum(ssha_mask))}")
    print(f"OLCI_DELAY_REFERENCE_M={olci_reference:.6f}")
    print(f"SWOT_AMR_DELAY_REFERENCE_M={amr_reference:.6f}")
    print(f"SHARED_ANOMALY_LIMIT_M={shared_limit:.6f}")
    print(f"SSHA_ANOMALY_LIMIT_M={ssha_limit:.6f}")
    print(f"OLCI_ANOMALY_LIMIT_M={olci_limit:.6f}")
    print(f"SWOT_AMR_ANOMALY_LIMIT_M={amr_limit:.6f}")
    print(f"WET_DELAY_PLOT_LIMIT_M={wet_delay_plot_limit:.6f}")
    print(f"SCALE_MODE={args.scale_mode}")
    print(f"SIG0_DB_LIMITS={sig0_low:.6f},{sig0_high:.6f}")
    print(f"OLCI_VALID_PIXELS={int(olci_mask.sum())}")
    print(f"OLCI_CLEAR_SKY_PERCENT={olci_clear_sky_percent:.6f}")
    print(f"SWOT_AMR_SIDE={swot_side}")
    print(f"SWOT_AMR_VALID_PIXELS={int(amr_mask.sum())}")


if __name__ == "__main__":
    main()
