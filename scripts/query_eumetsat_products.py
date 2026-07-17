from __future__ import annotations

import argparse
from datetime import datetime, timezone

import eumdac

from satmatch.credentials import load_api_credentials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--start", default="2023-07-27T18:35:00+00:00")
    parser.add_argument("--end", default="2023-07-27T18:55:00+00:00")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["EO:EUM:DAT:1121", "EO:EUM:DAT:0556", "EO:EUM:DAT:0407"],
    )
    args = parser.parse_args()
    credentials = load_api_credentials(args.credentials)
    token = eumdac.AccessToken((credentials.eumetsat_key, credentials.eumetsat_secret))
    store = eumdac.DataStore(token)

    start = datetime.fromisoformat(args.start).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end).astimezone(timezone.utc)
    for collection_id in args.collections:
        collection = store.get_collection(collection_id)
        products = list(collection.search(dtstart=start, dtend=end))
        print(f"COLLECTION {collection_id} MATCHES={len(products)}", flush=True)
        for product in products:
            print(
                f"PRODUCT {product} | {product.sensing_start.isoformat()} | "
                f"{product.sensing_end.isoformat()} | {product.satellite}",
                flush=True,
            )


if __name__ == "__main__":
    main()
