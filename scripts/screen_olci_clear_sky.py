from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import shutil

import eumdac
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import transform
import xarray as xr

from satmatch.credentials import load_api_credentials


INVALID_WQSF_FLAGS = (
    "INVALID",
    "LAND",
    "CLOUD",
    "CLOUD_AMBIGUOUS",
    "CLOUD_MARGIN",
    "SNOW_ICE",
    "WV_FAIL",
)
SWOT_SIDES = ("left", "right")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Associate OLCI WFR products with broad OLCI/SWOT intersections, "
            "screen quicklooks, and measure clear-sky coverage from WQSF."
        )
    )
    result.add_argument("--catalog", action="append", required=True)
    result.add_argument("--credentials", required=True)
    result.add_argument(
        "--temporary-directory",
        required=True,
        help="Temporary OLCI files are deleted product by product unless --keep-temporary is set",
    )
    result.add_argument("--output", required=True)
    result.add_argument("--collection", default="EO:EUM:DAT:0407")
    result.add_argument("--min-width-km", type=float, default=25.0)
    result.add_argument("--min-length-km", type=float, default=25.0)
    result.add_argument("--browse-clear-min", type=float, default=80.0)
    result.add_argument("--clear-sky-min", type=float, default=90.0)
    result.add_argument("--white-threshold", type=int, default=245)
    result.add_argument("--query-workers", type=int, default=8)
    result.add_argument("--download-workers", type=int, default=4)
    result.add_argument("--keep-temporary", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def load_scene_catalogs(
    paths: list[str],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load one-row-per-scene catalogues and their paired swath geometries."""
    scene_frames: list[gpd.GeoDataFrame] = []
    swath_frames: list[gpd.GeoDataFrame] = []
    for catalog_number, path in enumerate(paths):
        layers = set(gpd.list_layers(path)["name"])
        if "swaths" not in layers:
            raise ValueError(
                f"{path} has no 'swaths' layer; regenerate it with the paired-"
                "swath intersection search"
            )
        scenes = gpd.read_file(path, layer="vignettes").to_crs(4326)
        swaths = gpd.read_file(path, layer="swaths").to_crs(4326)
        required_scene = {"vignette_id", "geometry"}
        required_swath = {"vignette_id", "swot_side", "geometry"}
        if not required_scene.issubset(scenes.columns):
            raise ValueError(f"{path}: incomplete vignettes layer")
        if not required_swath.issubset(swaths.columns):
            raise ValueError(f"{path}: incomplete swaths layer")
        prefix = f"{catalog_number}:{Path(path).stem}:"
        scenes["_scene_key"] = prefix + scenes["vignette_id"].astype(str)
        swaths["_scene_key"] = prefix + swaths["vignette_id"].astype(str)
        scene_frames.append(scenes)
        swath_frames.append(swaths)

    scene_catalog = gpd.GeoDataFrame(
        pd.concat(scene_frames, ignore_index=True), crs=4326
    )
    swath_catalog = gpd.GeoDataFrame(
        pd.concat(swath_frames, ignore_index=True), crs=4326
    )
    duplicate_sides = swath_catalog.duplicated(["_scene_key", "swot_side"])
    if duplicate_sides.any():
        raise ValueError("A scene has duplicate left or right swath geometries")
    counts = swath_catalog.groupby("_scene_key")["swot_side"].agg(set)
    invalid = counts[counts.map(lambda sides: sides != {"left", "right"})]
    if not invalid.empty:
        raise ValueError("Every scene must contain one left and one right swath")
    return scene_catalog, swath_catalog


def scene_passes_clear_sky(
    left_percent: float,
    right_percent: float,
    threshold: float,
) -> bool:
    """Accept the complete scene when either SWOT swath passes."""
    return bool(
        (np.isfinite(left_percent) and left_percent >= threshold)
        or (np.isfinite(right_percent) and right_percent >= threshold)
    )


def rectangle_dimensions_km(frame: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    widths: list[float] = []
    lengths: list[float] = []
    for geometry in frame.geometry:
        center = geometry.centroid
        cosine = max(0.1, float(np.cos(np.deg2rad(center.y))))
        projected = transform(
            lambda x, y, z=None: (
                (((np.asarray(x) - center.x + 180.0) % 360.0) - 180.0)
                * 111.32
                * cosine,
                (np.asarray(y) - center.y) * 110.57,
            ),
            geometry,
        )
        coordinates = list(projected.minimum_rotated_rectangle.exterior.coords)
        edges = sorted(
            np.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(coordinates, coordinates[1:])
        )
        widths.append(float(np.mean(edges[:2])))
        lengths.append(float(np.mean(edges[2:])))
    return np.asarray(widths), np.asarray(lengths)


def download_entry(product, filename: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    if target.exists() and target.stat().st_size > 0:
        return target
    entries = {Path(str(entry)).name: str(entry) for entry in product.entries}
    if filename not in entries:
        raise RuntimeError(f"{product}: missing {filename}")
    partial = target.with_suffix(target.suffix + ".part")
    with product.open(entry=entries[filename]) as source, partial.open("wb") as sink:
        shutil.copyfileobj(source, sink, length=4 * 1024 * 1024)
    partial.replace(target)
    return target


def query_and_associate_products(
    collection, candidates: gpd.GeoDataFrame, workers: int
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    result = candidates.copy()
    result["olci_product"] = None
    result["olci_product_start"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    result["olci_product_end"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    best_distance: dict[int, float] = {}
    retained_products: dict[str, object] = {}
    candidates = candidates.copy()
    candidates["day"] = pd.to_datetime(candidates["s3_time"], utc=True).dt.date
    satellite_names = {"S3A": "Sentinel-3A", "S3B": "Sentinel-3B"}
    groups = list(
        candidates.groupby(
            ["s3_platform", "day", "s3_revolution"], sort=True
        )
    )

    def query_group(item):
        (platform, day, revolution), group = item
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        matches = collection.search(
            dtstart=start,
            dtend=end,
            sat=satellite_names[platform],
            timeliness="NT",
            orbit=int(revolution),
        )
        assignments: list[tuple[int, float, str, pd.Timestamp, pd.Timestamp]] = []
        products: dict[str, object] = {}
        for product in matches:
            footprint = shape(product.metadata["geometry"])
            product_start = pd.Timestamp(product.sensing_start)
            product_end = pd.Timestamp(product.sensing_end)
            if product_start.tzinfo is None:
                product_start = product_start.tz_localize("UTC")
                product_end = product_end.tz_localize("UTC")
            else:
                product_start = product_start.tz_convert("UTC")
                product_end = product_end.tz_convert("UTC")
            midpoint = product_start + (product_end - product_start) / 2
            for index, row in group.iterrows():
                timestamp = pd.Timestamp(row["s3_time"])
                if not (
                    product_start - timedelta(seconds=90)
                    <= timestamp
                    <= product_end + timedelta(seconds=90)
                ):
                    continue
                if not footprint.covers(row.geometry.centroid):
                    continue
                distance = abs((timestamp - midpoint).total_seconds())
                identifier = str(product)
                assignments.append(
                    (index, distance, identifier, product_start, product_end)
                )
                products[identifier] = product
        return platform, day, revolution, assignments, products

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(query_group, item) for item in groups]
        for future in as_completed(futures):
            platform, day, revolution, assignments, products = future.result()
            print(
                f"CATALOG {platform} {day} ORBIT={revolution} "
                f"MATCHED={len(assignments)}",
                flush=True,
            )
            for index, distance, identifier, product_start, product_end in assignments:
                if distance >= best_distance.get(index, float("inf")):
                    continue
                best_distance[index] = distance
                result.at[index, "olci_product"] = identifier
                result.at[index, "olci_product_start"] = product_start
                result.at[index, "olci_product_end"] = product_end
            retained_products.update(products)
    print(f"OLCI_PRODUCTS={len(retained_products)}", flush=True)
    return result, retained_products


def quicklook_coordinates(tie_path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    with xr.open_dataset(tie_path) as tie:
        row_indices = np.clip(
            np.rint(np.linspace(0, tie.sizes["tie_rows"] - 1, height)).astype(int),
            0,
            tie.sizes["tie_rows"] - 1,
        )
        across_factor = int(tie.attrs.get("ac_subsampling_factor", 64))
        tie_columns = np.arange(tie.sizes["tie_columns"]) * across_factor
        full_width = int(tie_columns[-1]) + 1
        output_columns = np.linspace(0, full_width - 1, width)
        source_lon = tie["longitude"].values[row_indices]
        source_lat = tie["latitude"].values[row_indices]
        longitude = np.empty((height, width), dtype=np.float32)
        latitude = np.empty((height, width), dtype=np.float32)
        for index in range(height):
            unwrapped = np.unwrap(np.deg2rad(source_lon[index]))
            longitude[index] = (
                np.rad2deg(np.interp(output_columns, tie_columns, unwrapped)) + 180.0
            ) % 360.0 - 180.0
            latitude[index] = np.interp(output_columns, tie_columns, source_lat[index])
    return longitude, latitude


def browse_clear_percent(
    geometry, browse_path: Path, tie_path: Path, white_threshold: int
) -> tuple[float, int]:
    image = np.asarray(Image.open(browse_path).convert("RGB"))
    longitude, latitude = quicklook_coordinates(tie_path, image.shape[:2])
    inside = contains_xy(geometry, longitude, latitude)
    count = int(inside.sum())
    if count == 0:
        return float("nan"), 0
    unavailable = np.min(image, axis=2) > white_threshold
    clear = inside & ~unavailable
    return float(100.0 * clear.sum() / count), count


def wqsf_clear_percent(geometry, wqsf_path: Path, tie_path: Path) -> tuple[float, int, int]:
    with xr.open_dataset(tie_path) as tie, xr.open_dataset(
        wqsf_path, mask_and_scale=False
    ) as quality_dataset:
        quality = quality_dataset["WQSF"]
        meanings = str(quality.attrs.get("flag_meanings", "")).split()
        masks = np.asarray(quality.attrs.get("flag_masks", []), dtype=np.uint64)
        by_name = dict(zip(meanings, masks, strict=True))
        missing = [name for name in INVALID_WQSF_FLAGS if name not in by_name]
        if missing:
            raise RuntimeError("WQSF missing flags: " + ", ".join(missing))

        minx, miny, maxx, maxy = geometry.bounds
        tie_lon = tie["longitude"].values
        tie_lat = tie["latitude"].values
        if maxx - minx > 180.0:
            longitude_window = (tie_lon >= minx) | (tie_lon <= maxx)
        else:
            longitude_window = (tie_lon >= minx - 1.0) & (tie_lon <= maxx + 1.0)
        nearby = longitude_window & (tie_lat >= miny - 1.0) & (tie_lat <= maxy + 1.0)
        rows, tie_columns_found = np.nonzero(nearby)
        if rows.size == 0:
            return float("nan"), 0, 0
        row_start = max(0, int(rows.min()) - 4)
        row_end = min(tie.sizes["tie_rows"], int(rows.max()) + 5)
        factor = int(tie.attrs.get("ac_subsampling_factor", 64))
        column_start = max(0, int(tie_columns_found.min()) * factor - factor)
        column_end = min(
            quality.sizes[quality.dims[1]],
            int(tie_columns_found.max()) * factor + factor + 1,
        )
        columns = np.arange(column_start, column_end)
        source_columns = np.arange(tie.sizes["tie_columns"]) * factor
        source_lon = tie["longitude"].values[row_start:row_end]
        source_lat = tie["latitude"].values[row_start:row_end]
        longitude = np.empty((row_end - row_start, len(columns)), dtype=np.float32)
        latitude = np.empty_like(longitude)
        for index in range(row_end - row_start):
            unwrapped = np.unwrap(np.deg2rad(source_lon[index]))
            longitude[index] = (
                np.rad2deg(np.interp(columns, source_columns, unwrapped)) + 180.0
            ) % 360.0 - 180.0
            latitude[index] = np.interp(columns, source_columns, source_lat[index])
        inside = contains_xy(geometry, longitude, latitude)
        inside_count = int(inside.sum())
        if inside_count == 0:
            return float("nan"), 0, 0
        values = np.asarray(
            quality.isel(
                {
                    quality.dims[0]: slice(row_start, row_end),
                    quality.dims[1]: slice(column_start, column_end),
                }
            ).values,
            dtype=np.uint64,
        )
        invalid = np.zeros(values.shape, dtype=bool)
        for name in INVALID_WQSF_FLAGS:
            invalid |= (values & by_name[name]) != 0
        clear_count = int((inside & ~invalid).sum())
        return float(100.0 * clear_count / inside_count), inside_count, clear_count


def main() -> None:
    args = parser().parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}; use --overwrite")

    try:
        candidates, swaths = load_scene_catalogs(args.catalog)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    widths, lengths = rectangle_dimensions_km(candidates)
    candidates["width_km"] = widths
    candidates["length_km"] = lengths
    candidates = candidates[
        (candidates["width_km"] >= args.min_width_km)
        & (candidates["length_km"] >= args.min_length_km)
    ].copy()
    swaths = swaths[swaths["_scene_key"].isin(candidates["_scene_key"])].copy()
    print(f"DIMENSION_SCENES={len(candidates)}", flush=True)

    credentials = load_api_credentials(args.credentials)
    token = eumdac.AccessToken((credentials.eumetsat_key, credentials.eumetsat_secret))
    collection = eumdac.DataStore(token).get_collection(args.collection)
    candidates, products = query_and_associate_products(
        collection, candidates, args.query_workers
    )
    candidates = candidates[candidates["olci_product"].notna()].copy()
    swaths = swaths[swaths["_scene_key"].isin(candidates["_scene_key"])].copy()
    swath_lookup = {
        scene_key: {
            str(row.swot_side): row.geometry
            for row in group.itertuples()
        }
        for scene_key, group in swaths.groupby("_scene_key", sort=False)
    }
    print(f"PRODUCT_SCENES={len(candidates)}", flush=True)

    cache = Path(args.temporary_directory)
    browse_values: dict[int, dict[str, tuple[float, int]]] = {}
    exact_values: dict[int, dict[str, tuple[float, int, int]]] = {}

    def screen_product(item):
        product_id, group = item
        local_browse: dict[int, dict[str, tuple[float, int]]] = {}
        local_exact: dict[int, dict[str, tuple[float, int, int]]] = {}
        product = products[product_id]
        folder = cache / product_id
        try:
            browse = download_entry(product, "browse.jpg", folder)
            tie = download_entry(product, "tie_geo_coordinates.nc", folder)
            browse_selected: list[int] = []
            for index, row in group.iterrows():
                geometries = swath_lookup[row["_scene_key"]]
                local_browse[index] = {
                    side: browse_clear_percent(
                        geometries[side],
                        browse,
                        tie,
                        args.white_threshold,
                    )
                    for side in SWOT_SIDES
                }
                if scene_passes_clear_sky(
                    local_browse[index]["left"][0],
                    local_browse[index]["right"][0],
                    args.browse_clear_min,
                ):
                    browse_selected.append(index)
            if browse_selected:
                wqsf = download_entry(product, "wqsf.nc", folder)
                for index in browse_selected:
                    scene_key = candidates.at[index, "_scene_key"]
                    geometries = swath_lookup[scene_key]
                    # Once the scene passes the inexpensive browse screen,
                    # measure both sides exactly and retain both diagnostics.
                    local_exact[index] = {
                        side: wqsf_clear_percent(
                            geometries[side], wqsf, tie
                        )
                        for side in SWOT_SIDES
                    }
        finally:
            if not args.keep_temporary and folder.exists():
                shutil.rmtree(folder)
        return product_id, len(group), local_browse, local_exact

    product_groups = list(candidates.groupby("olci_product", sort=True))
    with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as executor:
        futures = [executor.submit(screen_product, item) for item in product_groups]
        for future in as_completed(futures):
            product_id, count, local_browse, local_exact = future.result()
            browse_values.update(local_browse)
            exact_values.update(local_exact)
            print(
                f"SCREEN {product_id} SCENES={count} "
                f"WQSF={len(local_exact)}",
                flush=True,
            )

    for side in SWOT_SIDES:
        candidates[f"browse_clear_percent_{side}"] = [
            browse_values.get(i, {}).get(side, (float("nan"), 0))[0]
            for i in candidates.index
        ]
        candidates[f"browse_pixels_{side}"] = [
            browse_values.get(i, {}).get(side, (float("nan"), 0))[1]
            for i in candidates.index
        ]
    screened = candidates[candidates.index.isin(exact_values)].copy()
    print(f"BROWSE_SCREENED={len(screened)}", flush=True)
    for side in SWOT_SIDES:
        screened[f"clear_sky_percent_{side}"] = [
            exact_values[i][side][0] for i in screened.index
        ]
        screened[f"olci_pixels_{side}"] = [
            exact_values[i][side][1] for i in screened.index
        ]
        screened[f"clear_pixels_{side}"] = [
            exact_values[i][side][2] for i in screened.index
        ]
        screened[f"clear_sky_pass_{side}"] = (
            screened[f"clear_sky_percent_{side}"] >= args.clear_sky_min
        )
    screened["clear_sky_percent_best"] = np.fmax(
        screened["clear_sky_percent_left"],
        screened["clear_sky_percent_right"],
    )
    screened["selected_swaths"] = [
        ",".join(
            side
            for side in SWOT_SIDES
            if bool(row[f"clear_sky_pass_{side}"])
        )
        for _, row in screened.iterrows()
    ]
    selected = screened[
        screened["clear_sky_pass_left"] | screened["clear_sky_pass_right"]
    ].copy()
    selected = selected.sort_values(
        ["clear_sky_percent_best", "area_km2", "dt_minutes"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    selected["clear_sky_method"] = (
        "WQSF flags with OLCI tie-point geolocation, evaluated per SWOT swath"
    )
    selected["scene_selection_rule"] = (
        "complete scene retained when left OR right clear-sky percentage passes"
    )
    selected["search_collection"] = args.collection
    selected["search_min_clear_sky_percent"] = args.clear_sky_min
    selected["search_min_width_km"] = args.min_width_km
    selected["search_min_length_km"] = args.min_length_km
    selected.insert(
        0,
        "annual_case_id",
        [f"OLCI_SWOT_2026_{i:05d}" for i in range(1, len(selected) + 1)],
    )

    selected_swaths = swaths.merge(
        selected[["_scene_key", "annual_case_id"]],
        on="_scene_key",
        how="inner",
    )
    public_selected = selected.drop(columns=["_scene_key"])
    public_swaths = selected_swaths.drop(columns=["_scene_key"])

    output.parent.mkdir(parents=True, exist_ok=True)
    public_selected.to_file(output, layer="intersections", driver="GPKG")
    public_swaths.to_file(output, layer="swaths", driver="GPKG", mode="a")
    csv_output = output.with_suffix(".csv")
    public_selected.drop(columns="geometry").to_csv(csv_output, index=False)
    print(f"SELECTED_SCENES={len(public_selected)}", flush=True)
    print(f"OUTPUT={output}", flush=True)
    print(f"CSV={csv_output}", flush=True)


if __name__ == "__main__":
    main()
