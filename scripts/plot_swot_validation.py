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
            "Plot colocated corrected SWOT SSH, SWOT Sigma0, and OLCI wet path delay."
        )
    )
    result.add_argument("--subset", required=True, help="SWOT vignette NetCDF")
    result.add_argument(
        "--olci",
        required=True,
        help="Converted OLCI NetCDF containing wet_tropo_path_delay",
    )
    result.add_argument(
        "--swot-model",
        required=True,
        help="SWOT L2 LR SSH Expert NetCDF containing model_wet_tropo_cor",
    )
    result.add_argument("--vignette", required=True, help="Vignette GeoPackage")
    result.add_argument("--land", required=True, help="Land polygon dataset")
    result.add_argument("--output", required=True, help="Output figure path")
    result.add_argument("--title", default="coastal matchup")
    result.add_argument(
        "--olci-delay-variable", default="wet_tropo_path_delay"
    )
    result.add_argument(
        "--swot-model-variable", default="model_wet_tropo_cor"
    )
    result.add_argument("--olci-longitude-variable")
    result.add_argument("--olci-latitude-variable")
    result.add_argument(
        "--scale-mode",
        choices=("shared", "independent"),
        default="shared",
        help=(
            "shared uses one metre scale for SSH and both wet delays; "
            "independent enhances patterns with separate robust scales"
        ),
    )
    return result


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()

    vignette = gpd.read_file(args.vignette).to_crs(4326)
    if len(vignette) != 1:
        argument_parser.error("The vignette file must contain exactly one feature")
    polygon = vignette.geometry.iloc[0]
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

        ssh = np.asarray(dataset["ssh_karin_2"].values, dtype=float)
        sig0 = np.asarray(dataset["sig0_karin_2"].values, dtype=float)
        ssh_mask = usable_mask(dataset, "ssh_karin_2", "ssh_karin_2_qual")
        sig0_mask = usable_mask(
            dataset, "sig0_karin_2", "sig0_karin_2_qual"
        ) & (sig0 > 0.0)
        require_valid(ssh_mask, "SWOT SSH")
        require_valid(sig0_mask, "SWOT Sigma0")

        ssh_reference = float(np.nanmedian(ssh[ssh_mask]))
        ssh_anomaly = ssh - ssh_reference
        sig0_db = np.full(sig0.shape, np.nan, dtype=float)
        sig0_db[sig0_mask] = 10.0 * np.log10(sig0[sig0_mask])
        sig0_low, sig0_high = np.nanpercentile(sig0_db[sig0_mask], [2.0, 98.0])

        ssh_flags = np.nan_to_num(
            dataset["ssh_karin_2_qual"].values, nan=2**32 - 1
        ).astype(np.uint32)
        degraded = int(
            np.sum(ssh_mask & ((ssh_flags & np.uint32(1 << 30)) != 0))
        )
        time_start = np.datetime_as_string(dataset["time"].values[0], unit="s")
        time_end = np.datetime_as_string(dataset["time"].values[-1], unit="s")
        swot_side = str(dataset.attrs.get("swot_side", "")).casefold()
        if swot_side not in {"left", "right"}:
            argument_parser.error(
                "The SWOT subset must identify its swot_side as left or right"
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

    with xr.open_dataset(args.swot_model, mask_and_scale=True) as dataset:
        required_model_variables = (
            args.swot_model_variable,
            "longitude",
            "latitude",
            "cross_track_distance",
            "ssh_karin_2_qual",
            "ancillary_surface_classification_flag",
        )
        missing = [name for name in required_model_variables if name not in dataset]
        if missing:
            argument_parser.error(
                "SWOT model file is missing: " + ", ".join(missing)
            )
        model_lon = np.asarray(dataset["longitude"].values, dtype=float)
        model_lon = ((model_lon + 180.0) % 360.0) - 180.0
        model_lat = np.asarray(dataset["latitude"].values, dtype=float)
        model_correction = np.asarray(
            dataset[args.swot_model_variable].values, dtype=float
        )
        model_cross_track = np.asarray(
            dataset["cross_track_distance"].values, dtype=float
        )
        model_flags = np.nan_to_num(
            dataset["ssh_karin_2_qual"].values, nan=2**32 - 1
        ).astype(np.uint32)
        model_surface = np.asarray(
            dataset["ancillary_surface_classification_flag"].values
        )

    model_inside = contains_xy(polygon, model_lon, model_lat)
    model_selected_side = (
        model_cross_track < 0.0 if swot_side == "left" else model_cross_track > 0.0
    )
    model_mask = (
        model_inside
        & model_selected_side
        & (model_surface == 0)
        & np.isfinite(model_lon)
        & np.isfinite(model_lat)
        & np.isfinite(model_correction)
        & ((model_flags & BAD_NOT_USABLE) == 0)
        & ((model_flags & BAD_OUTSIDE_RANGE) == 0)
    )
    require_valid(model_mask, "SWOT model wet-troposphere correction")
    model_x, model_y = transformer.transform(model_lon, model_lat)
    model_x, model_y = model_x / 1000.0, model_y / 1000.0
    # SWOT stores a negative range correction. Negating it gives a positive
    # equivalent vertical wet path delay comparable to the OLCI-derived delay.
    model_delay = -model_correction
    model_reference = float(np.nanmedian(model_delay[model_mask]))
    model_anomaly = model_delay - model_reference

    ssh_limit = float(np.nanpercentile(np.abs(ssh_anomaly[ssh_mask]), 98.0))
    olci_limit = float(np.nanpercentile(np.abs(olci_anomaly[olci_mask]), 98.0))
    model_limit = float(
        np.nanpercentile(np.abs(model_anomaly[model_mask]), 98.0)
    )
    shared_limit = max(ssh_limit, olci_limit, model_limit, 1.0e-6)
    shared_norm = TwoSlopeNorm(
        vmin=-shared_limit, vcenter=0.0, vmax=shared_limit
    )
    if args.scale_mode == "shared":
        ssh_norm = olci_norm = model_norm = shared_norm
        wet_delay_plot_limit = shared_limit
        scale_description = (
            "SSH, OLCI delay, and SWOT model-delay anomalies share "
            f"±{shared_limit:.3f} m colour limits."
        )
    else:
        ssh_norm = TwoSlopeNorm(
            vmin=-max(ssh_limit, 1.0e-6),
            vcenter=0.0,
            vmax=max(ssh_limit, 1.0e-6),
        )
        olci_norm = TwoSlopeNorm(
            vmin=-max(olci_limit, 1.0e-6),
            vcenter=0.0,
            vmax=max(olci_limit, 1.0e-6),
        )
        # Keep both wet-delay panels on the OLCI amplitude scale so their
        # spatial variations remain directly comparable.
        model_norm = olci_norm
        wet_delay_plot_limit = olci_limit
        scale_description = (
            "SSH uses an independent symmetric 98th-percentile scale; OLCI "
            "and SWOT model wet delays share the OLCI 98th-percentile scale."
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
            "mask": ssh_mask,
            "values": ssh_anomaly,
            "cmap": cmocean.cm.balance,
            "norm": ssh_norm,
            "title": f"Corrected SWOT SSH − median ({ssh_reference:.3f} m)",
            "unit": "SSH anomaly (m)",
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
            "x": model_x,
            "y": model_y,
            "mask": model_mask,
            "values": model_anomaly,
            "cmap": cmocean.cm.balance,
            "norm": model_norm,
            "title": (
                "SWOT model wet path delay − median "
                f"({model_reference:.3f} m)"
            ),
            "unit": "Model wet path-delay anomaly (m)",
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
        f"valid pixels: SSH {int(ssh_mask.sum()):,}, "
        f"Sigma0 {int(sig0_mask.sum()):,}, OLCI delay {int(olci_mask.sum()):,}, "
        f"SWOT model delay {int(model_mask.sum()):,}; "
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
        f"SSH pixels retained. OLCI delay source: {olci_source_variable}; "
        f"{olci_conversion_method}. "
        "SWOT model delay = −model_wet_tropo_cor. "
        f"Model side: {swot_side}. "
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
    print(f"SSH_REFERENCE_M={ssh_reference:.6f}")
    print(f"OLCI_DELAY_REFERENCE_M={olci_reference:.6f}")
    print(f"SWOT_MODEL_DELAY_REFERENCE_M={model_reference:.6f}")
    print(f"SHARED_ANOMALY_LIMIT_M={shared_limit:.6f}")
    print(f"SSH_ANOMALY_LIMIT_M={ssh_limit:.6f}")
    print(f"OLCI_ANOMALY_LIMIT_M={olci_limit:.6f}")
    print(f"SWOT_MODEL_ANOMALY_LIMIT_M={model_limit:.6f}")
    print(f"WET_DELAY_PLOT_LIMIT_M={wet_delay_plot_limit:.6f}")
    print(f"SCALE_MODE={args.scale_mode}")
    print(f"SIG0_DB_LIMITS={sig0_low:.6f},{sig0_high:.6f}")
    print(f"OLCI_VALID_PIXELS={int(olci_mask.sum())}")
    print(f"OLCI_CLEAR_SKY_PERCENT={olci_clear_sky_percent:.6f}")
    print(f"SWOT_MODEL_SIDE={swot_side}")
    print(f"SWOT_MODEL_VALID_PIXELS={int(model_mask.sum())}")


if __name__ == "__main__":
    main()
