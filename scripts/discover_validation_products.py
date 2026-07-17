from __future__ import annotations

import argparse
import os

import earthaccess
import eumdac

from satmatch.credentials import load_api_credentials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--source", choices=("all", "nasa", "eumetsat"), default="all")
    args = parser.parse_args()
    credentials = load_api_credentials(args.credentials)

    if args.source in ("all", "nasa"):
        os.environ["EARTHDATA_TOKEN"] = credentials.earthdata_token
        auth = earthaccess.login(strategy="environment")
        if not auth.authenticated:
            raise RuntimeError("Échec de l'authentification Earthdata")

        swot = earthaccess.search_data(
            short_name="SWOT_L2_LR_SSH_UNSMOOTHED_D",
            temporal=("2023-07-24T09:45:00Z", "2023-07-24T10:00:00Z"),
            bounding_box=(17.9, 77.6, 20.4, 78.2),
            count=20,
        )
        print(f"SWOT_MATCHES={len(swot)}", flush=True)
        for granule in swot:
            print(f"SWOT {granule['umm']['GranuleUR']}", flush=True)

    if args.source in ("all", "eumetsat"):
        token = eumdac.AccessToken((credentials.eumetsat_key, credentials.eumetsat_secret))
        store = eumdac.DataStore(token)
        matches = []
        for collection in store.collections:
            title = collection.title or ""
            product_type = collection.product_type or ""
            text = f"{title} {product_type}".casefold()
            if "olci" in text and "level 2" in text and "full resolution" in text and "water" in text:
                matches.append((str(collection), title, product_type))
        print(f"EUMETSAT_COLLECTIONS={len(matches)}", flush=True)
        for identifier, title, product_type in matches:
            print(f"EUMETSAT {identifier} | {product_type} | {title}", flush=True)


if __name__ == "__main__":
    main()
