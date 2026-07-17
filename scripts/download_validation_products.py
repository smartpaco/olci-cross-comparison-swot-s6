from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import earthaccess
import eumdac

from satmatch.credentials import load_api_credentials


def download_olci(credentials, destination: Path, collection_id: str, product_id: str, filenames: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    token = eumdac.AccessToken((credentials.eumetsat_key, credentials.eumetsat_secret))
    product = eumdac.DataStore(token).get_product(collection_id, product_id)
    entries = {Path(str(entry)).name: str(entry) for entry in product.entries}
    missing = sorted(set(filenames) - entries.keys())
    if missing:
        raise RuntimeError(f"Entrées EUMETSAT absentes: {', '.join(missing)}")
    for filename in filenames:
        target = destination / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"OLCI SKIP {filename} {target.stat().st_size / 1e6:.1f} MB", flush=True)
            continue
        print(f"OLCI START {filename}", flush=True)
        partial = target.with_suffix(target.suffix + ".part")
        with product.open(entry=entries[filename]) as source, partial.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=4 * 1024 * 1024)
        partial.replace(target)
        print(f"OLCI DONE {filename} {target.stat().st_size / 1e6:.1f} MB", flush=True)


def download_swot(credentials, destination: Path, collection: str, start: str, end: str, pass_number: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["EARTHDATA_TOKEN"] = credentials.earthdata_token
    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise RuntimeError("Échec de l'authentification Earthdata")
    matches = earthaccess.search_data(
        short_name=collection,
        temporal=(start, end),
        count=100,
    )
    marker = f"_{pass_number:03d}_"
    granule = next(item for item in matches if marker in item["umm"]["GranuleUR"])
    print(f"SWOT START {granule['umm']['GranuleUR']} ({granule.size():.1f} MB)", flush=True)
    paths = earthaccess.download(granule, destination, threads=1, show_progress=True)
    for path in paths:
        print(f"SWOT DONE {Path(path).name} {Path(path).stat().st_size / 1e6:.1f} MB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", choices=("all", "olci", "swot"), default="all")
    parser.add_argument("--olci-collection", default="EO:EUM:DAT:1121")
    parser.add_argument("--olci-product")
    parser.add_argument("--olci-files", nargs="+", default=["TCWV.nc", "xfdumanifest.xml"])
    parser.add_argument("--swot-collection", default="SWOT_L2_LR_SSH_UNSMOOTHED_D")
    parser.add_argument("--swot-start")
    parser.add_argument("--swot-end")
    parser.add_argument("--swot-pass", type=int)
    args = parser.parse_args()
    credentials = load_api_credentials(args.credentials)
    root = Path(args.output)
    if args.source in ("all", "olci"):
        if not args.olci_product:
            parser.error("--olci-product est requis pour OLCI")
        download_olci(credentials, root / "olci", args.olci_collection, args.olci_product, args.olci_files)
    if args.source in ("all", "swot"):
        if not (args.swot_start and args.swot_end and args.swot_pass is not None):
            parser.error("--swot-start, --swot-end et --swot-pass sont requis pour SWOT")
        download_swot(
            credentials,
            root / "swot",
            args.swot_collection,
            args.swot_start,
            args.swot_end,
            args.swot_pass,
        )


if __name__ == "__main__":
    main()
