# OLCI cross-comparison with SWOT and S6

Tools for finding spatio-temporal intersections between Sentinel-3 OLCI,
SWOT KaRIn, and, in a later phase, Sentinel-6 microwave-radiometer
observations. The project
uses compact Orbit Revolution Files (ORFs) for acquisition screening, so large
Level-2 science products are downloaded only after useful matchups have been
identified.

## Scientific goals

### Current work: OLCI and SWOT

The first component finds intersections between the Sentinel-3A or Sentinel-3B
OLCI field of view and the two SWOT KaRIn swaths. The platform is selected for
each run from the command line. Candidate intersections are constrained in
space, time, latitude, surface area, ocean fraction, and along-track length.

For selected matchups, the scientific comparison is visual:

- map OLCI Total Column Water Vapour (TCWV/COWa) and its equivalent wet path
  delay at native OLCI resolution;
- map fully corrected SWOT KaRIn sea-surface height (`ssh_karin_2`);
- map SWOT KaRIn normalized radar cross section (`sig0_karin_2`), converted to
  decibels for display;
- map the positive equivalent wet path delay derived from the SWOT Expert
  `model_wet_tropo_cor` field;
- inspect whether coherent structures in OLCI TCWV are still visible as wet
  tropospheric path-delay residuals in corrected SWOT SSH;
- interpret Sigma0 jointly with TCWV and SSH. Sigma0 responds to sea-surface
  backscatter, but the Ka-band echo is also modified by atmospheric attenuation
  and precipitation. Small-scale atmospheric structures that are absent from
  the model-based correction can therefore appear in `sig0_karin_2`; they must
  not automatically be attributed to the surface. Ice, rain, land
  contamination, and coastal effects remain alternative explanations.

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

- **Sentinel-3A or Sentinel-3B OLCI:** nominal 1,270 km swath, represented as
  635 km on each side of nadir.
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
orbits/                 Included Sentinel-3A, Sentinel-3B, and SWOT ORFs
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

### Sentinel-3A

```powershell
.\.venv\Scripts\find-s3-swot.exe `
  --s3-platform S3A `
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

### Sentinel-3B

Use the same command with the S3B platform and ORF:

```powershell
.\.venv\Scripts\find-s3-swot.exe `
  --s3-platform S3B `
  --s3-orf "orbits\S3B_ORF_AXXCNE20260717_080700_20181123_213005_20260801_080330" `
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
  --output "outputs\s3b_swot_20260616_20260626.gpkg"
```

`--s3-platform` may be omitted when the ORF filename contains `S3A` or `S3B`.
The command validates an explicitly selected platform against the filename to
prevent accidental use of the wrong orbit table. Output vignette identifiers
are prefixed with `S3A_SWOT_` or `S3B_SWOT_`, and the catalogue contains an
explicit `s3_platform` field. The former `find-s3a-swot` command remains as a
backward-compatible alias, but `find-s3-swot` is recommended.

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

The SWOT model wet-troposphere field is stored in the matching Expert product.
Download that much smaller granule with the same cycle, pass, and time window:

```powershell
.\.venv\Scripts\python.exe scripts\download_validation_products.py `
  --credentials "C:\path\outside\the\repository\credentials.txt" `
  --output "data\validation_case" `
  --source swot `
  --swot-collection SWOT_L2_LR_SSH_EXPERT_D `
  --swot-start 2026-06-17T14:40:00Z `
  --swot-end 2026-06-17T15:40:00Z `
  --swot-pass 537
```

The download script skips existing non-empty files and writes via `.part`
files before atomically renaming completed downloads.

## 4. Convert OLCI TCWV to wet tropospheric path delay

`IWV_W` is the integrated water-vapour column in kg m⁻². The conversion
implemented here produces the positive, one-way zenith wet propagation delay
in metres:

```text
ZWD = (A + B / Tm) * TCWV
A = -2.95077e-5 m / (kg m-2)
B =  1.73276 m K / (kg m-2)
```

`Tm` is the water-vapour-weighted mean atmospheric temperature. This is
Eq. A15 of [Bennartz et al. (2017)](https://doi.org/10.5194/amt-10-1387-2017).
The OLCI `IWV_W` field and its kg m⁻² units are documented in the
[Copernicus Sentinel-3 OLCI L2 data description](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3OLCIL2.html).

Run the conversion after downloading the selected `TCWV.nc` file:

```powershell
.\.venv\Scripts\python.exe scripts\convert_olci_tcwv.py `
  --input "data\validation_case\olci\TCWV.nc" `
  --output "data\validation_case\olci\TCWV_with_wet_delay.nc" `
  --vignette "data\validation_case\swot_subset.gpkg" `
  --mean-temperature-k 270
```

The variable name is detected automatically among `IWV_W`, `TCWV`, `tcwv`, and
common case variants when `--tcwv-variable` is omitted. With `--vignette`, the
converter first locates the relevant OLCI rows and columns and loads only TCWV,
longitude, latitude, uncertainty, and quality information. This avoids loading
the full 4091 x 4865 product into memory. Pixels with `qi = 0` are excluded.
The compact output contains a float32 `wet_tropo_path_delay` variable in metres
and an exact `in_vignette` mask. Existing scale factors and fill values are
decoded by xarray before conversion.

With the default `Tm = 270 K`, the conversion factor is
6.388 mm of wet delay per kg m⁻² of TCWV, close to the commonly quoted
6.4 mm. This constant-temperature mode is intended for initial pattern and
amplitude comparisons. For a more accurate absolute amplitude, first add a
pixel-wise water-vapour-weighted `Tm` field from a meteorological profile, then
use `--tm-variable <variable-name>`. A fixed `Tm` changes the multiplicative
amplitude but does not create new small-scale spatial patterns.

The generated delay is positive excess path length, not a signed SSH
correction. Compare demeaned or detrended amplitudes in metres. If an excess
wet delay is left uncorrected in the radar range, its expected SSH residual has
the opposite sign; the exact sign comparison must follow the convention of the
SWOT correction variable being analysed.

## 5. Extract the exact SWOT vignette

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

## 6. Plot corrected SWOT SSH, Sigma0, OLCI delay, and SWOT model delay

```powershell
.\.venv\Scripts\python.exe scripts\plot_swot_validation.py `
  --subset "data\validation_case\swot_subset.nc" `
  --olci "data\validation_case\olci\TCWV_with_wet_delay.nc" `
  --swot-model "data\validation_case\swot\<SWOT Expert granule>.nc" `
  --vignette "data\validation_case\swot_subset.gpkg" `
  --land "data\natural_earth\ne_10m_land.zip" `
  --title "coastal validation case" `
  --scale-mode independent `
  --output "outputs\olci_swot_four_panel_patterns.png"
```

The figure contains four side-by-side panels in this order:

1. corrected SWOT `ssh_karin_2` minus its vignette median;
2. SWOT `sig0_karin_2` in decibels;
3. OLCI `wet_tropo_path_delay` minus its vignette median.
4. SWOT model wet path delay minus its vignette median.

The Expert variable `model_wet_tropo_cor` is a negative correction in metres.
The plotting code negates it to obtain a positive equivalent vertical wet path
delay before removing its median, making its sign convention comparable to the
OLCI-derived positive delay.

All figure titles, colour bars, annotations, and processing comments are in
English. Corrected SSH and both wet-delay anomalies are expressed in metres.
`--scale-mode shared` applies one symmetric colour scale to all three metre
panels for direct visual amplitude comparison. `--scale-mode independent`
applies an independent symmetric 98th-percentile scale to SSH, while OLCI and
the SWOT model wet-delay panels both use the OLCI 98th-percentile limit. This
keeps the two wet-delay amplitudes directly comparable while still revealing
their spatial patterns. The console field `WET_DELAY_PLOT_LIMIT_M` records the
limit used by both wet-delay colour bars.
Sigma0 always uses an independent percentile-based dB scale. Each instrument
remains on its native grid; no spatial resampling or resolution matching is
performed. The coarser SWOT Expert wet-delay field is rendered as filled native
curvilinear grid cells rather than point markers. This makes the panel read as
an image while preserving the model product's native spatial resolution and
without interpolating it onto the 250 m KaRIn grid.

The SWOT panels use native open-ocean pixels. `bad_not_usable` and
`bad_outside_of_range` pixels are excluded. The OLCI panel uses finite wet-delay
pixels over ocean whose centres fall inside the exact vignette polygon; land,
invalid, or cloudy TCWV pixels remain absent. OLCI longitude and latitude are
automatically detected for common 1-D or 2-D coordinate layouts. Non-standard
names can be provided with `--olci-longitude-variable` and
`--olci-latitude-variable`.

The figure header and the console output report OLCI clear-sky data coverage.
It is calculated as the percentage of ocean pixel centres inside the vignette
that have a finite converted OLCI wet-delay value. The machine-readable console
field is `OLCI_CLEAR_SKY_PERCENT`.

The plotted `sig0_karin_2` uses a model-based atmospheric attenuation
correction. Rain, cloud liquid water, and water-vapour-related attenuation can
leave atmospheric signatures when that model does not resolve the observed
feature. See the
[SWOT L2 LR SSH product description](https://www.aviso.altimetry.fr/fileadmin/documents/data/tools/D-56407_SWOT_Product_Description_L2_LR_SSH_20220902_RevA.pdf)
for the distinction between `sig0_karin` and `sig0_karin_2`.

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
space-time prefilter behaviour, polar-latitude rejection, the TCWV-to-wet-delay
physics, memory-efficient NetCDF conversion, OLCI coordinate broadcasting, and
generation of the four-panel comparison figure.

## Current limitations

- Sentinel-3A and Sentinel-3B use the same nominal OLCI field-of-view model;
  platform-specific differences beyond their ORFs are not represented.
- The OLCI swath is currently a symmetric nominal buffer rather than an exact
  instrument edge model.
- ORF interpolation is designed for acquisition screening, not precise pixel
  geolocation.
- Cloud-free OLCI coverage is not available from ORFs and must be determined
  from product information.
- The default TCWV-to-wet-delay conversion uses a constant `Tm = 270 K`.
  Pixel-wise meteorological profiles are required for the best absolute
  accuracy and for a spatially varying conversion factor.
- Sentinel-6 radiometer colocation, product download, analysis, and comparison
  workflows remain on the roadmap.

## Roadmap

1. Add OLCI TCWV subsetting, quality masks, and clear-sky coverage metrics.
2. Quantify spatial correlations and scale-dependent coherence.
3. Add Sentinel-6 radiometer colocation and wet-troposphere analysis.
4. Add pixel-wise meteorological `Tm` generation and propagated uncertainty
   estimates for the wet-delay conversion.

## Data policy and attribution

The repository contains software and the user-supplied ORFs only. It does not
redistribute NASA or EUMETSAT Level-2 products. Users must obtain data through
the official services and comply with the corresponding licences and
attribution requirements.
