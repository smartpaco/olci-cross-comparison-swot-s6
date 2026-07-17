# OLCI cross-comparison with SWOT and S6

Tools for finding spatio-temporal intersections between Sentinel-3 OLCI,
SWOT KaRIn, and, in a later phase, Sentinel-6 microwave-radiometer
observations. The project
uses compact Orbit Revolution Files (ORFs) for acquisition screening, so large
Level-2 science products are downloaded only after useful matchups have been
identified.

## Scientific goals

### Current work: OLCI and SWOT

The first component finds intersections between the Sentinel-3A OLCI field of
view and the two SWOT KaRIn swaths. Candidate intersections are constrained in
space, time, latitude, surface area, ocean fraction, and along-track length.

For selected matchups, the scientific comparison is visual:

- map OLCI Total Column Water Vapour (TCWV/COWa) at native OLCI resolution;
- map fully corrected SWOT KaRIn sea-surface height (`ssh_karin_2`);
- map SWOT KaRIn normalized radar cross section (`sig0_karin_2`), converted to
  decibels for display;
- inspect whether coherent structures in OLCI TCWV are still visible as wet
  tropospheric path-delay residuals in corrected SWOT SSH;
- use Sigma0 to identify structures more plausibly associated with surface,
  ice, rain, land contamination, or coastal effects.

No artificial resolution matching is performed. Each sensor is displayed on
its native grid and only the common geographic vignette is applied.

### Planned work: OLCI and Sentinel-6

The Sentinel-6 component will focus exclusively on the wet-tropospheric
correction derived from the onboard microwave radiometer, especially its
high-frequency spatial variability. The objective is to determine whether fine
atmospheric-water-vapour features, such as moist fronts visible in OLCI TCWV,
can be detected in or compared with the Sentinel-6 radiometer correction.

Sentinel-6 radiometer colocation and scientific analysis are not implemented
yet.

## Current field-of-view model

- **Sentinel-3A OLCI:** nominal 1,270 km swath, represented as 635 km on each
  side of nadir.
- **SWOT KaRIn:** two separate swaths, 10 to 60 km from nadir on the left and
  right sides.
- **Time separation:** 30 minutes by default.
- **Vignette length:** at most 50 km along the SWOT ground track.
- **Minimum vignette area:** 400 km² by default.
- **Minimum ocean fraction:** 50% by default.
- **Latitude range:** -66° to +66° by default, excluding the dense polar
  convergence region.

These values are command-line options and can be changed without modifying the
source code.

## Method

1. Read pole/equator/pole events from the mission ORFs.
2. Interpolate satellite sub-points on the unit sphere using cubic splines.
3. Select only half-orbits overlapping the requested dates.
4. Apply a conservative 60-second 3-D KD-tree space-time prefilter to reject
   clearly separated half-orbit pairs.
5. Apply a finer simultaneous distance and time test along the SWOT track.
6. Build OLCI and KaRIn polygons in local azimuthal-equidistant projections.
7. Split long intersections into along-track vignettes no longer than 50 km.
8. Calculate area and ocean fraction using Natural Earth land polygons.
9. Export a GeoPackage and a companion CSV catalogue.

The latitude constraint is applied in the coarse prefilter, the fine candidate
search, and the final vignette check. On a ten-day test window, this rejected
743 of 896 temporal half-orbit pairs before detailed geometry.

## Repository layout

```text
orbits/                 Included Sentinel-3A and SWOT ORFs
src/satmatch/           ORF parsing, geometry, ocean mask, and matchup engine
scripts/                Product discovery, download, subsetting, and plotting
tests/                  Unit tests for ORFs, geometry, and prefilters
pyproject.toml           Python package and dependency definition
```

Generated `data/` and `outputs/` directories are intentionally ignored by Git.
API keys, bearer tokens, and credential files must never be committed.

## Requirements

- Python 3.11 or newer
- Internet access for the initial Natural Earth mask download
- NASA Earthdata access for SWOT L2 products
- EUMETSAT Data Store access and the required Copernicus licence for OLCI TCWV

## Installation

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### Linux or macOS

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev]"
```

## Credentials

Orbit intersection searches require no API credentials. Product discovery and
download scripts read a local text file supplied with `--credentials`. Keep the
file outside the repository. It must contain labels for an EUMETSAT key,
EUMETSAT secret, and NASA Earthdata bearer token, followed by their local
values. Placeholder example:

```text
EUMETSAT
Key
<local consumer key>
Secret
<local consumer secret>

NASA Token
<local Earthdata bearer token>
```

The repository ignore rules exclude common credential filenames, but users are
still responsible for checking staged files before every commit.

## 1. Find OLCI–SWOT intersections

The ORFs required by this example are included in `orbits/`.

```powershell
.\.venv\Scripts\find-s3a-swot.exe `
  --s3-orf "orbits\S3A_ORF_AXXCNE20260717_075300_20160302_154759_20260801_081654" `
  --swot-orf "orbits\SWOT_ORF_AXXCNE20260717_103800_20230720_200750_20260801_081502" `
  --start 2026-06-16 `
  --end 2026-06-26 `
  --dt-minutes 30 `
  --sample-seconds 30 `
  --prefilter-seconds 60 `
  --max-along-track-km 50 `
  --min-area-km2 400 `
  --min-ocean-percent 50 `
  --min-latitude -66 `
  --max-latitude 66 `
  --land-resolution 50m `
  --output "outputs\s3a_swot_20260616_20260626.gpkg"
```

The command writes:

- a GeoPackage containing exact vignette polygons;
- a CSV containing the same attributes without geometry.

Important output fields include acquisition times, absolute time separation,
cycle/pass/revolution identifiers, KaRIn side, along-track limits, total area,
ocean area, ocean percentage, centre coordinates, and processing parameters.

For an exploratory search, use a 30-second orbit sampling interval and the 50m
land mask. Refine a selected acquisition with `--sample-seconds 5` and
`--land-resolution 10m`. To include polar regions explicitly, set
`--min-latitude -90 --max-latitude 90`.

Natural Earth land masks are downloaded automatically on first use.

## 2. Search the EUMETSAT TCWV catalogue

The current dedicated OLCI COWa/TCWV collection is
`EO:EUM:DAT:1121` (`OL_2_TCWVFR`).

```powershell
.\.venv\Scripts\python.exe scripts\query_eumetsat_products.py `
  --credentials "C:\path\outside\the\repository\credentials.txt" `
  --start 2026-06-17T15:47:00+00:00 `
  --end 2026-06-17T15:54:00+00:00 `
  --collections EO:EUM:DAT:1121
```

This step queries metadata only. It does not download the science product.

## 3. Download only selected validation products

After selecting a vignette, download the matching OLCI TCWV entry:

```powershell
.\.venv\Scripts\python.exe scripts\download_validation_products.py `
  --credentials "C:\path\outside\the\repository\credentials.txt" `
  --output "data\validation_case" `
  --source olci `
  --olci-collection EO:EUM:DAT:1121 `
  --olci-product "<exact OL_2_TCWVFR product identifier>" `
  --olci-files TCWV.nc xfdumanifest.xml
```

Download the matching SWOT L2 LR Unsmoothed granule from NASA Earthdata:

```powershell
.\.venv\Scripts\python.exe scripts\download_validation_products.py `
  --credentials "C:\path\outside\the\repository\credentials.txt" `
  --output "data\validation_case" `
  --source swot `
  --swot-start 2026-06-17T15:00:00Z `
  --swot-end 2026-06-17T15:40:00Z `
  --swot-pass 537
```

The download script skips existing non-empty files and writes via `.part`
files before atomically renaming completed downloads.

## 4. Extract the exact SWOT vignette

```powershell
.\.venv\Scripts\python.exe scripts\subset_swot_validation.py `
  --catalog "outputs\case_refined.gpkg" `
  --vignette-id S3A_SWOT_00000001 `
  --swot "data\validation_case\swot\<SWOT granule>.nc" `
  --output "data\validation_case\swot_subset.nc"
```

The subset contains native SWOT pixels, geolocation, time, corrected SSH,
SSHA, Sigma0, quality flags, surface classification, and an exact
`in_vignette` mask. A one-feature GeoPackage and CSV are written beside it.

## 5. Plot corrected SWOT SSH and Sigma0

```powershell
.\.venv\Scripts\python.exe scripts\plot_swot_validation.py `
  --subset "data\validation_case\swot_subset.nc" `
  --vignette "data\validation_case\swot_subset.gpkg" `
  --land "data\natural_earth\ne_10m_land.zip" `
  --title "coastal validation case" `
  --output "outputs\swot_ssh_sig0.png"
```

The plot uses native open-ocean SWOT pixels. `bad_not_usable` and
`bad_outside_of_range` pixels are excluded. Corrected SSH is displayed after
subtracting the vignette median, which changes only the reference level and
preserves spatial structures. Sigma0 is converted from linear units to dB.

## OLCI clear-sky coverage

The TCWV product is generated for cloud-free daytime pixels. Once TCWV access
is available, clear-sky coverage should be reported at least as:

```text
valid TCWV pixels inside the vignette / OLCI pixels inside the vignette
```

and separately over ocean. A product browse image can provide a rapid
qualitative indication, but an exact percentage requires TCWV validity and
quality information for the vignette.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover ORF parsing and interpolation, tangent geometry handling,
space-time prefilter behaviour, and polar-latitude rejection.

## Current limitations

- Only Sentinel-3A OLCI versus SWOT is implemented in the matchup engine.
- The OLCI swath is currently a symmetric nominal buffer rather than an exact
  instrument edge model.
- ORF interpolation is designed for acquisition screening, not precise pixel
  geolocation.
- Cloud-free OLCI coverage is not available from ORFs and must be determined
  from product information.
- TCWV-to-wet-path-delay conversion is not implemented yet and will require an
  explicit physical model and ancillary atmospheric information.
- Sentinel-6 radiometer colocation, product download, analysis, and comparison
  workflows remain on the roadmap.

## Roadmap

1. Add OLCI TCWV subsetting, quality masks, and clear-sky coverage metrics.
2. Add a combined TCWV / corrected SSH / Sigma0 comparison figure.
3. Quantify spatial correlations and scale-dependent coherence.
4. Add Sentinel-3B ORF support to the same catalogue workflow.
5. Add Sentinel-6 radiometer colocation and wet-troposphere analysis.
6. Add a documented TCWV-to-wet-path-delay model with uncertainty estimates.

## Data policy and attribution

The repository contains software and the user-supplied ORFs only. It does not
redistribute NASA or EUMETSAT Level-2 products. Users must obtain data through
the official services and comply with the corresponding licences and
attribution requirements.
