from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .geometry import GeometryOptions
from .matchups import find_matchups
from .ocean import ensure_land_mask


S3_PLATFORM_PATTERN = re.compile(r"(?:^|[^A-Z0-9])(S3[AB])(?:[^A-Z0-9]|$)")


def resolve_s3_platform(orf_path: str | Path, requested: str | None = None) -> str:
    """Resolve S3A/S3B from the CLI option and, when possible, the ORF name."""
    match = S3_PLATFORM_PATTERN.search(Path(orf_path).name.upper())
    inferred = match.group(1) if match else None
    platform = requested.upper() if requested else inferred
    if platform is None:
        raise ValueError(
            "Unable to infer the Sentinel-3 platform from the ORF filename; "
            "use --s3-platform S3A or --s3-platform S3B"
        )
    if inferred is not None and requested is not None and inferred != platform:
        raise ValueError(
            f"--s3-platform {platform} does not match ORF filename {Path(orf_path).name}"
        )
    return platform


def utc_time(value: str | None):
    return None if value is None else pd.Timestamp(value, tz="UTC")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Intersections sur océan entre les FOV S3A ou S3B OLCI et SWOT/KaRIn à partir des ORF."
    )
    result.add_argument(
        "--s3-orf", required=True, help="Fichier ORF Sentinel-3A ou Sentinel-3B"
    )
    result.add_argument(
        "--s3-platform",
        choices=("S3A", "S3B"),
        help="Plateforme OLCI; déduite du nom du fichier ORF si cette option est omise",
    )
    result.add_argument("--swot-orf", required=True, help="Fichier ORF SWOT")
    result.add_argument(
        "--output",
        help="GeoPackage de sortie (défaut: outputs/s3a_swot.gpkg ou outputs/s3b_swot.gpkg)",
    )
    result.add_argument("--start", help="Début UTC inclus, ex. 2023-07-20")
    result.add_argument("--end", help="Fin UTC incluse, ex. 2023-07-22")
    result.add_argument("--dt-minutes", type=float, default=30.0)
    result.add_argument("--sample-seconds", type=float, default=5.0)
    result.add_argument("--max-along-track-km", type=float, default=50.0)
    result.add_argument("--olci-half-swath-km", type=float, default=635.0)
    result.add_argument(
        "--min-ocean-percent", type=float, default=50.0,
        help="Fraction océanique minimale en pourcentage (défaut: 50)",
    )
    result.add_argument(
        "--min-area-km2", type=float, default=400.0,
        help="Surface minimale de l'intersection en km² (défaut: 400)",
    )
    result.add_argument("--land-mask", help="Masque terrestre local (remplace --land-resolution)")
    result.add_argument(
        "--land-resolution",
        choices=("10m", "50m", "110m"),
        default="50m",
        help="Résolution Natural Earth (défaut: 50m; 110m pour un criblage rapide)",
    )
    result.add_argument(
        "--prefilter-seconds",
        type=float,
        default=60.0,
        help="Pas du préfiltre espace-temps des demi-orbites; 0 le désactive",
    )
    result.add_argument("--min-latitude", type=float, default=-66.0)
    result.add_argument("--max-latitude", type=float, default=66.0)
    return result


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        s3_platform = resolve_s3_platform(args.s3_orf, args.s3_platform)
    except ValueError as error:
        argument_parser.error(str(error))
    if not (-90.0 <= args.min_latitude < args.max_latitude <= 90.0):
        raise SystemExit(
            "Les limites doivent vérifier "
            "-90 <= min-latitude < max-latitude <= 90"
        )
    output = Path(args.output or f"outputs/{s3_platform.lower()}_swot.gpkg")
    output.parent.mkdir(parents=True, exist_ok=True)
    land_mask = ensure_land_mask(args.land_mask, args.land_resolution)
    options = GeometryOptions(
        olci_half_swath_km=args.olci_half_swath_km,
        max_along_track_km=args.max_along_track_km,
        max_dt_minutes=args.dt_minutes,
        min_area_km2=args.min_area_km2,
        min_ocean_percent=args.min_ocean_percent,
        min_latitude=args.min_latitude,
        max_latitude=args.max_latitude,
    )
    matches = find_matchups(
        args.s3_orf,
        args.swot_orf,
        str(land_mask),
        options,
        start=utc_time(args.start),
        end=utc_time(args.end),
        sample_seconds=args.sample_seconds,
        prefilter_seconds=args.prefilter_seconds,
    )
    matches = matches[
        (matches["ocean_percent"] >= args.min_ocean_percent)
        & (matches["area_km2"] >= args.min_area_km2)
    ].copy()
    matches["land_mask"] = Path(land_mask).name
    matches["sample_seconds"] = args.sample_seconds
    matches["prefilter_seconds"] = args.prefilter_seconds
    matches["min_latitude"] = args.min_latitude
    matches["max_latitude"] = args.max_latitude
    matches.insert(0, "s3_platform", s3_platform)
    matches.insert(
        0,
        "vignette_id",
        [f"{s3_platform}_SWOT_{idx:08d}" for idx in range(1, len(matches) + 1)],
    )
    swath_records: list[dict] = []
    for _, row in matches.iterrows():
        for side in ("left", "right"):
            swath_records.append(
                {
                    "vignette_id": row["vignette_id"],
                    "s3_platform": s3_platform,
                    "swot_side": side,
                    "area_km2": row[f"{side}_area_km2"],
                    "ocean_percent": row[f"{side}_ocean_percent"],
                    "dt_minutes": row[f"{side}_dt_minutes"],
                    "geometry": row[f"{side}_geometry"],
                }
            )
    swaths = gpd.GeoDataFrame(
        swath_records,
        columns=[
            "vignette_id",
            "s3_platform",
            "swot_side",
            "area_km2",
            "ocean_percent",
            "dt_minutes",
            "geometry",
        ],
        geometry="geometry",
        crs=4326,
    )
    public_matches = matches.drop(columns=["left_geometry", "right_geometry"])
    public_matches.to_file(output, layer="vignettes", driver="GPKG")
    swaths.to_file(output, layer="swaths", driver="GPKG", mode="a")
    public_matches.drop(columns="geometry").to_csv(
        output.with_suffix(".csv"), index=False
    )
    print(
        f"{len(public_matches)} SWOT scenes written to {output} with "
        f"{len(swaths)} side geometries and {output.with_suffix('.csv')}"
    )


if __name__ == "__main__":
    main()
