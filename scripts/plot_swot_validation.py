from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import cmocean
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from pyproj import CRS, Transformer
from shapely import contains_xy
from shapely.affinity import scale
import xarray as xr


BAD_NOT_USABLE = np.uint32(1 << 31)
BAD_OUTSIDE_RANGE = np.uint32(1 << 29)
BAD_RADIOMETER_CORR_MISSING = np.uint32(1 << 28)
OLCI_LONGITUDE_CANDIDATES = ("longitude", "lon", "longitude_signed")
OLCI_LATITUDE_CANDIDATES = ("latitude", "lat")


@dataclass
class SwotLayer:
    x: np.ndarray
    y: np.ndarray
    ssha: np.ndarray
    sig0_db: np.ndarray
    ssha_mask: np.ndarray
    sig0_mask: np.ndarray
    side: str
    time: np.ndarray
    source: str
    degraded: int
    xcal: np.ndarray


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


def cropped_mesh(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    rows, columns = np.where(mask)
    if rows.size == 0:
        raise ValueError("Cannot crop an empty mesh")
    row_slice = slice(max(0, int(rows.min()) - 1), min(x.shape[0], int(rows.max()) + 2))
    column_slice = slice(
        max(0, int(columns.min()) - 1), min(x.shape[1], int(columns.max()) + 2)
    )
    local_mask = mask[row_slice, column_slice]
    return (
        x[row_slice, column_slice],
        y[row_slice, column_slice],
        np.ma.masked_where(~local_mask, values[row_slice, column_slice]),
    )


def draw_native_grid(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    *,
    cmap,
    norm,
    zorder: int = 2,
):
    """Draw a native 2-D grid as cells; retain scatter for legacy flat subsets."""
    if x.ndim == 2 and min(x.shape) > 1:
        mesh_x, mesh_y, mesh_values = cropped_mesh(x, y, values, mask)
        return axis.pcolormesh(
            mesh_x,
            mesh_y,
            mesh_values,
            shading="nearest",
            linewidth=0,
            cmap=cmap,
            norm=norm,
            rasterized=True,
            zorder=zorder,
        )
    return axis.scatter(
        x[mask],
        y[mask],
        c=values[mask],
        s=4.0,
        marker="s",
        linewidths=0,
        cmap=cmap,
        norm=norm,
        rasterized=True,
        zorder=zorder,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Plot colocated XCAL-corrected SWOT SSHA, SWOT Sigma0, "
            "SWOT AMR wet path delay, and OLCI wet path delay."
        )
    )
    result.add_argument(
        "--subset",
        action="append",
        required=True,
        help="SWOT vignette NetCDF; repeat to retain both native swath grids",
    )
    result.add_argument("--olci", required=True)
    result.add_argument(
        "--swot-expert",
        "--swot-model",
        dest="swot_expert",
        required=True,
    )
    result.add_argument("--vignette", required=True)
    result.add_argument("--vignette-id", action="append")
    result.add_argument("--land", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--title", default="coastal matchup")
    result.add_argument("--olci-delay-variable", default="wet_tropo_path_delay")
    result.add_argument(
        "--swot-radiometer-variable",
        "--swot-model-variable",
        dest="swot_radiometer_variable",
        default="rad_wet_tropo_cor",
    )
    result.add_argument("--olci-longitude-variable")
    result.add_argument("--olci-latitude-variable")
    result.add_argument(
        "--scale-mode",
        choices=("shared", "independent"),
        default="shared",
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

    swot_layers: list[SwotLayer] = []
    for subset_path in args.subset:
        with xr.open_dataset(subset_path) as dataset:
            missing_xcal = [
                name
                for name in ("height_cor_xover", "height_cor_xover_qual")
                if name not in dataset
            ]
            if missing_xcal:
                argument_parser.error(
                    f"{subset_path} is missing XCAL fields: "
                    + ", ".join(missing_xcal)
                )
            side = str(dataset.attrs.get("swot_side", "")).casefold()
            if side not in {"left", "right", "both"}:
                argument_parser.error(
                    f"{subset_path} must identify swot_side as left, right, or both"
                )
            longitude = np.asarray(dataset["longitude_signed"].values, dtype=float)
            latitude = np.asarray(dataset["latitude"].values, dtype=float)
            x, y = transformer.transform(longitude, latitude)
            x, y = np.asarray(x) / 1000.0, np.asarray(y) / 1000.0
            xcal = np.asarray(dataset["height_cor_xover"].values, dtype=float)
            xcal_quality = np.nan_to_num(
                dataset["height_cor_xover_qual"].values, nan=255
            ).astype(np.uint8)
            xcal_good = np.isfinite(xcal) & (xcal_quality == 0)
            ssha_native = np.asarray(dataset["ssha_karin_2"].values, dtype=float)
            ssha = ssha_native + xcal
            sig0 = np.asarray(dataset["sig0_karin_2"].values, dtype=float)
            ssha_mask = (
                usable_mask(dataset, "ssha_karin_2", "ssha_karin_2_qual")
                & xcal_good
            )
            sig0_mask = usable_mask(
                dataset, "sig0_karin_2", "sig0_karin_2_qual"
            ) & (sig0 > 0.0)
            require_valid(ssha_mask, f"SWOT {side} SSHA")
            require_valid(sig0_mask, f"SWOT {side} Sigma0")
            sig0_db = np.full(sig0.shape, np.nan, dtype=float)
            sig0_db[sig0_mask] = 10.0 * np.log10(sig0[sig0_mask])
            ssha_flags = np.nan_to_num(
                dataset["ssha_karin_2_qual"].values, nan=2**32 - 1
            ).astype(np.uint32)
            degraded = int(
                np.sum(ssha_mask & ((ssha_flags & np.uint32(1 << 30)) != 0))
            )
            swot_layers.append(
                SwotLayer(
                    x=x,
                    y=y,
                    ssha=ssha,
                    sig0_db=sig0_db,
                    ssha_mask=ssha_mask,
                    sig0_mask=sig0_mask,
                    side=side,
                    time=np.asarray(dataset["time"].values),
                    source=str(dataset.attrs.get("source_product", "")),
                    degraded=degraded,
                    xcal=xcal,
                )
            )

    ssha_values = np.concatenate(
        [layer.ssha[layer.ssha_mask] for layer in swot_layers]
    )
    xcal_values = np.concatenate(
        [layer.xcal[layer.ssha_mask] for layer in swot_layers]
    )
    sig0_values = np.concatenate(
        [layer.sig0_db[layer.sig0_mask] for layer in swot_layers]
    )
    ssha_reference = float(np.nanmedian(ssha_values))
    xcal_reference = float(np.nanmedian(xcal_values))
    xcal_limit = float(np.nanpercentile(np.abs(xcal_values - xcal_reference), 98.0))
    sig0_low, sig0_high = np.nanpercentile(sig0_values, [2.0, 98.0])
    ssha_anomalies = [layer.ssha - ssha_reference for layer in swot_layers]
    ssha_anomaly_values = np.concatenate(
        [
            anomaly[layer.ssha_mask]
            for anomaly, layer in zip(ssha_anomalies, swot_layers, strict=True)
        ]
    )
    sides = {
        side
        for layer in swot_layers
        for side in ({"left", "right"} if layer.side == "both" else {layer.side})
    }
    swot_side = "both" if sides == {"left", "right"} else next(iter(sides))
    time_start = np.datetime_as_string(
        min(layer.time.min() for layer in swot_layers), unit="s"
    )
    time_end = np.datetime_as_string(
        max(layer.time.max() for layer in swot_layers), unit="s"
    )
    source = next((layer.source for layer in swot_layers if layer.source), "")
    marker = re.search(r"_(\d{3})_(\d{3})_", source)
    source_label = (
        f"SWOT L2 LR SSH Unsmoothed - cycle {marker.group(1)}, pass {marker.group(2)}"
        if marker
        else source
    )

    with xr.open_dataset(args.olci, mask_and_scale=True) as dataset:
        if args.olci_delay_variable not in dataset:
            argument_parser.error(
                f"OLCI delay variable {args.olci_delay_variable!r} is absent"
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
    olci_x, olci_y = np.asarray(olci_x) / 1000.0, np.asarray(olci_y) / 1000.0
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
        radiometer_surface = np.asarray(dataset["rad_surface_type_flag"].values)

    amr_inside = contains_xy(polygon, amr_lon, amr_lat)
    left_surface_valid = np.isin(radiometer_surface[:, 0], (0, 1))
    right_surface_valid = np.isin(radiometer_surface[:, 1], (0, 1))
    amr_base_mask = (
        amr_inside
        & (amr_surface == 0)
        & np.isfinite(amr_lon)
        & np.isfinite(amr_lat)
        & np.isfinite(amr_correction)
        & ((amr_flags & BAD_RADIOMETER_CORR_MISSING) == 0)
        & ((amr_flags & BAD_NOT_USABLE) == 0)
        & ((amr_flags & BAD_OUTSIDE_RANGE) == 0)
    )
    amr_side_masks: list[tuple[str, np.ndarray]] = []
    if "left" in sides:
        amr_side_masks.append(
            (
                "left",
                amr_base_mask
                & (amr_cross_track < 0.0)
                & left_surface_valid[:, np.newaxis],
            )
        )
    if "right" in sides:
        amr_side_masks.append(
            (
                "right",
                amr_base_mask
                & (amr_cross_track > 0.0)
                & right_surface_valid[:, np.newaxis],
            )
        )
    amr_mask = np.logical_or.reduce([mask for _, mask in amr_side_masks])
    require_valid(amr_mask, "SWOT AMR wet-troposphere correction")
    amr_x, amr_y = transformer.transform(amr_lon, amr_lat)
    amr_x, amr_y = np.asarray(amr_x) / 1000.0, np.asarray(amr_y) / 1000.0
    amr_delay = -amr_correction
    amr_reference = float(np.nanmedian(amr_delay[amr_mask]))
    amr_anomaly = amr_delay - amr_reference

    ssha_limit = float(np.nanpercentile(np.abs(ssha_anomaly_values), 98.0))
    olci_limit = float(np.nanpercentile(np.abs(olci_anomaly[olci_mask]), 98.0))
    amr_limit = float(np.nanpercentile(np.abs(amr_anomaly[amr_mask]), 98.0))
    shared_limit = max(ssha_limit, olci_limit, amr_limit, 1.0e-6)
    shared_norm = TwoSlopeNorm(
        vmin=-shared_limit, vcenter=0.0, vmax=shared_limit
    )
    if args.scale_mode == "shared":
        ssha_norm = olci_norm = amr_norm = shared_norm
        wet_delay_plot_limit = shared_limit
        scale_description = (
            "SSHA, OLCI delay, and SWOT AMR-delay anomalies share "
            f"+/-{shared_limit:.3f} m colour limits."
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
    bounds = local_vignette.total_bounds
    for axis in axes:
        if not land.empty:
            land.plot(
                ax=axis,
                facecolor="#d8d4cc",
                edgecolor="#68645e",
                linewidth=0.45,
                zorder=1,
            )

    for layer, anomaly in zip(swot_layers, ssha_anomalies, strict=True):
        draw_native_grid(
            axes[0],
            layer.x,
            layer.y,
            anomaly,
            layer.ssha_mask,
            cmap=cmocean.cm.balance,
            norm=ssha_norm,
        )
        draw_native_grid(
            axes[1],
            layer.x,
            layer.y,
            layer.sig0_db,
            layer.sig0_mask,
            cmap=cmocean.cm.amp,
            norm=Normalize(vmin=float(sig0_low), vmax=float(sig0_high)),
        )
    for _, side_mask in amr_side_masks:
        if np.any(side_mask):
            draw_native_grid(
                axes[2],
                amr_x,
                amr_y,
                amr_anomaly,
                side_mask,
                cmap=cmocean.cm.balance,
                norm=amr_norm,
            )
    draw_native_grid(
        axes[3],
        olci_x,
        olci_y,
        olci_anomaly,
        olci_mask,
        cmap=cmocean.cm.balance,
        norm=olci_norm,
    )

    panel_metadata = (
        (
            axes[0],
            "XCAL-corrected SWOT SSHA - median "
            f"({ssha_reference:.3f} m)",
            "SSHA anomaly (m)",
            cmocean.cm.balance,
            ssha_norm,
        ),
        (
            axes[1],
            "SWOT Sigma0 with model atmospheric correction",
            "Sigma0 (dB)",
            cmocean.cm.amp,
            Normalize(vmin=float(sig0_low), vmax=float(sig0_high)),
        ),
        (
            axes[2],
            "SWOT AMR wet path delay - median "
            f"({amr_reference:.3f} m)",
            "AMR wet path-delay anomaly (m)",
            cmocean.cm.balance,
            amr_norm,
        ),
        (
            axes[3],
            f"OLCI wet path delay - median ({olci_reference:.3f} m)",
            "Wet path-delay anomaly (m)",
            cmocean.cm.balance,
            olci_norm,
        ),
    )
    for axis, title, unit, cmap, norm in panel_metadata:
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
            ScalarMappable(norm=norm, cmap=cmap),
            ax=axis,
            orientation="horizontal",
            pad=0.14,
            fraction=0.055,
        )
        colorbar.set_label(unit)
        axis.set_title(title, fontsize=10.5)
        axis.set_aspect("equal")
        axis.set_xlim(bounds[0] - 3.0, bounds[2] + 3.0)
        axis.set_ylim(bounds[1] - 3.0, bounds[3] + 3.0)
        axis.set_xlabel("Local easting (km)")
        axis.grid(color="#8a8a8a", linewidth=0.35, alpha=0.35)
    axes[0].set_ylabel("Local northing (km)")

    ssha_count = sum(int(layer.ssha_mask.sum()) for layer in swot_layers)
    sig0_count = sum(int(layer.sig0_mask.sum()) for layer in swot_layers)
    degraded = sum(layer.degraded for layer in swot_layers)
    pixel_summary = (
        f"valid pixels: SSHA {ssha_count:,}, Sigma0 {sig0_count:,}, "
        f"OLCI delay {int(olci_mask.sum()):,}, SWOT AMR delay "
        f"{int(amr_mask.sum()):,}; OLCI clear-sky coverage "
        f"{olci_clear_sky_percent:.1f}%"
    )
    figure.suptitle(
        f"OLCI-SWOT wet-troposphere comparison - {args.title}\n"
        f"SWOT {time_start} to {time_end} UTC - {pixel_summary}",
        fontsize=13,
        y=0.96,
    )
    figure.text(
        0.5,
        0.025,
        f"Native sensor grids; no spatial resampling. {scale_description} "
        "Sigma0 can contain both surface and atmospheric signatures.\n"
        f"Native-geolocation vignette and open-ocean quality masks applied; "
        f"{degraded} degraded SSHA pixels retained. "
        "SSHA = ssha_karin_2 + height_cor_xover; only good XCAL retained "
        f"(median {xcal_reference:.3f} m). OLCI delay source: "
        f"{olci_source_variable}; {olci_conversion_method}. "
        f"SWOT AMR delay = -{args.swot_radiometer_variable}; "
        "left and right AMR meshes are rendered separately. "
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
    print(f"XCAL_GOOD_PIXELS={ssha_count}")
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
