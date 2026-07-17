from __future__ import annotations

import argparse
import eumdac

from satmatch.credentials import load_api_credentials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--contains", nargs="*", default=[])
    args = parser.parse_args()
    credentials = load_api_credentials(args.credentials)
    token = eumdac.AccessToken((credentials.eumetsat_key, credentials.eumetsat_secret))
    product = eumdac.DataStore(token).get_product(args.collection, args.product)
    for entry in product.entries:
        if not args.contains or any(name in str(entry) for name in args.contains):
            print(entry)


if __name__ == "__main__":
    main()
