from __future__ import annotations

import argparse
import os

import earthaccess
import xarray as xr

from satmatch.credentials import load_api_credentials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True)
    args = parser.parse_args()
    credentials = load_api_credentials(args.credentials)
    os.environ["EARTHDATA_TOKEN"] = credentials.earthdata_token
    earthaccess.login(strategy="environment")
    matches = earthaccess.search_data(
        short_name="SWOT_L2_LR_SSH_UNSMOOTHED_D",
        temporal=("2023-07-27T18:25:00Z", "2023-07-27T19:25:00Z"),
        count=100,
    )
    granule = next(item for item in matches if "_184_" in item["umm"]["GranuleUR"])
    remote = earthaccess.open([granule], show_progress=False)[0]
    with xr.open_dataset(remote, group="left", engine="h5netcdf") as dataset:
        print(f"GRANULE={granule['umm']['GranuleUR']}")
        print(f"DIMS={dict(dataset.sizes)}")
        for name in ("latitude", "longitude", "ssh_karin", "ssh_karin_2", "sig0_karin", "sig0_karin_2"):
            variable = dataset[name]
            print(f"VAR={name} dims={variable.dims} dtype={variable.dtype} units={variable.attrs.get('units')}")


if __name__ == "__main__":
    main()

