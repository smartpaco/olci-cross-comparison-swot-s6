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
from shapely.affinity import scale
import xarray as xr


BAD_NOT_USABLE = np.uint32(1 << 31)
BAD_OUTSIDE_RANGE = np.uint32(1 << 29)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--vignette", required=True)
    parser.add_argument("--land", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="vignette côtière")
    args = parser.parse_args()

    vignette = gpd.read_file(args.vignette).to_crs(4326)
    polygon = vignette.geometry.iloc[0]
    lon0, lat0 = float(polygon.centroid.x), float(polygon.centroid.y)
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.10f} +lon_0={lon0:.10f} +datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(4326, local_crs, always_xy=True)
    local_vignette = vignette.to_crs(local_crs)
    local_vignette.geometry = local_vignette.geometry.map(
        lambda geometry: scale(geometry, xfact=0.001, yfact=0.001, origin=(0.0, 0.0))
    )

    land = gpd.read_file(args.land, bbox=polygon.bounds).to_crs(4326)
    land = land.clip(polygon.buffer(1.0)).to_crs(local_crs)
    land.geometry = land.geometry.map(
        lambda geometry: scale(geometry, xfact=0.001, yfact=0.001, origin=(0.0, 0.0))
    )

    with xr.open_dataset(args.subset) as dataset:
        lon = dataset["longitude_signed"].values
        lat = dataset["latitude"].values
        x, y = transformer.transform(lon, lat)
        x, y = x / 1000.0, y / 1000.0

        ssh = dataset["ssh_karin_2"].values
        sig0 = dataset["sig0_karin_2"].values
        ssh_mask = usable_mask(dataset, "ssh_karin_2", "ssh_karin_2_qual")
        sig0_mask = usable_mask(dataset, "sig0_karin_2", "sig0_karin_2_qual") & (sig0 > 0.0)

        ssh_reference = float(np.nanmedian(ssh[ssh_mask]))
        ssh_local = ssh - ssh_reference
        ssh_limit = float(np.nanpercentile(np.abs(ssh_local[ssh_mask]), 98.0))
        sig0_db = np.full(sig0.shape, np.nan, dtype=float)
        sig0_db[sig0_mask] = 10.0 * np.log10(sig0[sig0_mask])
        sig0_low, sig0_high = np.nanpercentile(sig0_db[sig0_mask], [2.0, 98.0])

        ssh_flags = np.nan_to_num(dataset["ssh_karin_2_qual"].values, nan=2**32 - 1).astype(np.uint32)
        degraded = int(np.sum(ssh_mask & ((ssh_flags & np.uint32(1 << 30)) != 0)))
        time_start = np.datetime_as_string(dataset["time"].values[0], unit="s")
        time_end = np.datetime_as_string(dataset["time"].values[-1], unit="s")
        source = dataset.attrs.get("source_product", "")
        marker = re.search(r"_(\d{3})_(\d{3})_", source)
        source_label = (
            f"SWOT L2 LR SSH Unsmoothed — cycle {marker.group(1)}, passe {marker.group(2)}"
            if marker else source
        )

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.83, wspace=0.13)
    panels = [
        (
            axes[0], ssh_mask, ssh_local, cmocean.cm.balance,
            TwoSlopeNorm(vmin=-ssh_limit, vcenter=0.0, vmax=ssh_limit),
            f"SSH KaRIn corrigée − médiane ({ssh_reference:.3f} m)", "m",
        ),
        (
            axes[1], sig0_mask, sig0_db, cmocean.cm.amp,
            plt.Normalize(vmin=float(sig0_low), vmax=float(sig0_high)),
            "Sigma0 KaRIn corrigé", "dB",
        ),
    ]

    bounds = local_vignette.total_bounds
    for axis, mask, values, cmap, norm, title, unit in panels:
        if not land.empty:
            land.plot(ax=axis, facecolor="#d8d4cc", edgecolor="#68645e", linewidth=0.45, zorder=1)
        points = axis.scatter(
            x[mask], y[mask], c=values[mask], s=4.0, marker="s", linewidths=0,
            cmap=cmap, norm=norm, rasterized=True, zorder=2,
        )
        local_vignette.boundary.plot(ax=axis, color="#111111", linewidth=1.0, zorder=3)
        colorbar = fig.colorbar(points, ax=axis, orientation="horizontal", pad=0.08, fraction=0.055)
        colorbar.set_label(unit)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlim(bounds[0] - 3.0, bounds[2] + 3.0)
        axis.set_ylim(bounds[1] - 3.0, bounds[3] + 3.0)
        axis.set_xlabel("Est local (km)")
        axis.grid(color="#8a8a8a", linewidth=0.35, alpha=0.35)
    axes[0].set_ylabel("Nord local (km)")
    fig.suptitle(
        f"SWOT L2 LR Unsmoothed — {args.title}\n"
        f"{time_start} à {time_end} UTC · pixels océan utilisables : "
        f"SSH {int(ssh_mask.sum()):,}, Sigma0 {int(sig0_mask.sum()):,}".replace(",", " "),
        fontsize=13, y=0.96,
    )
    fig.text(
        0.5, 0.025,
        f"Masque : vignette ∩ open_ocean ; bad_not_usable et bad_outside_of_range exclus ; "
        f"{degraded} pixels SSH dégradés conservés\n{source_label}",
        ha="center", va="bottom", fontsize=8.5, color="#444444",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"PLOT={output}")
    print(f"SSH_REFERENCE_M={ssh_reference:.6f} SSH_LIMIT_M={ssh_limit:.6f}")
    print(f"SIG0_DB_LIMITS={sig0_low:.6f},{sig0_high:.6f}")


if __name__ == "__main__":
    main()
