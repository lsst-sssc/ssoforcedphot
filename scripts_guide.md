# Quick-look scripts: `plot_lightcurve` and `compare_rubin`

Temporary notes. Both scripts read the photometry results JSON written by the photometry
service (e.g. `output/2024_TN57_photometry_results.json`) and live in `src/forcedphot/`.

| | `plot_lightcurve.py` | `compare_rubin.py` |
|---|---|---|
| Purpose | Light curve of the app's own photometry | Check the app's photometry against the Rubin pipeline |
| Runs where | Anywhere (matplotlib only) | RSP only (TAP query); the logic itself is testable offline |
| Input | One or more result JSON files | One result JSON file |
| Output | `<input>_lightcurve.png` | `<input>_rubin.csv`, `<input>_rubin.png`, printed summary |
| Console command | `forcedphot-lightcurve` | `forcedphot-compare-rubin` |
| Tests | `tests/forcedphot/test_plot_lightcurve.py` | `tests/forcedphot/test_compare_rubin.py` |

The console commands are declared in `pyproject.toml` under `[project.scripts]` and become
available after `pip install -e .`. Without installing, run the files directly with `python`.

---

## Input: the photometry results JSON

A list of visit records (a single dict is also accepted). The fields the scripts use:

```jsonc
{
  "target_name": "2024 TN57",
  "band": "z",
  "obs_time": "60652.08246629621",      // MJD, as a string
  "visit_id": 2024120700152,            // compare_rubin only
  "detector_id": 3,                     // compare_rubin only (reported, not matched on)
  "forced_phot_on_target": {            // forced PSF photometry at the ephemeris position
    "ra": 37.28207, "dec": 6.62711,
    "flux": 6546.1, "flux_err": 730.9,  // nJy
    "mag": 21.86, "mag_err": 0.12       // AB
  },
  "phot_within_error_ellipse": [        // sources detected inside the ephemeris error ellipse
    { "ra": ..., "dec": ..., "flux": ..., "flux_err": ..., "mag": ..., "mag_err": ...,
      "separation": 0.33 }              // arcsec from the ephemeris position
  ]
}
```

The pipeline writes `0.0` where a measurement failed or was NaN; both scripts drop those visits.

### Choosing the measurement: `--source`

Both scripts share `select_photometry()` from `plot_lightcurve.py`:

- **`nearest`** (default): the detected source in `phot_within_error_ellipse` with the smallest
  `separation`. If it is further than `--max-sep` (default 1.0″) it is treated as an unrelated
  field source, and the forced measurement is used for that visit instead
  (`plot_lightcurve` prints how many visits fell back).
- **`forced`**: always `forced_phot_on_target`, i.e. PSF photometry at the ephemeris position.

---

## `plot_lightcurve.py`

Plots magnitude (or flux) against MJD with error bars, one colour and marker per band
(Rubin band order u g r i z y, colour-vision-safe palette).

### Usage

```bash
python src/forcedphot/plot_lightcurve.py output/2024_TN57_photometry_results.json

# several files (e.g. separate runs of the same target) in one plot
python src/forcedphot/plot_lightcurve.py run1.json run2.json -o combined.png

# forced photometry, flux instead of magnitude, one panel per band
python src/forcedphot/plot_lightcurve.py results.json --source forced --y flux --separate

# only g and r, open a window too
python src/forcedphot/plot_lightcurve.py results.json --bands g r --show
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `json_files` | — | One or more result JSON files |
| `--bands B [B ...]` | every band present | Bands to plot (`u g r i z y`) |
| `--y mag\|flux` | `mag` | Plot AB magnitude (inverted axis) or flux in nJy |
| `--source nearest\|forced` | `nearest` | See above |
| `--max-sep ARCSEC` | `1.0` | Match radius for `nearest` |
| `--separate` | off | One stacked subplot per band |
| `-o, --output PNG` | `<first input>_lightcurve.png` | Output file |
| `--show` | off | Also open the figure in a window |

MJD tick labels are written in full (no `+6.0652e4` offset notation).

---

## `compare_rubin.py`

Answers: *did the Rubin pipeline detect this object in the same visits, and does its
photometry agree with ours?*

### How it works

1. **Load** the app rows from the JSON (one per usable visit, using `--source`). The position
   used for matching is the ra/dec of the selected measurement.
2. **Query** the RSP TAP service once:
   ```sql
   SELECT diaSourceId, ssObjectId, visit, detector, ra, dec,
          psfFlux, psfFluxErr, scienceFlux, scienceFluxErr, isDipole
   FROM dp1.DiaSource
   WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', <track centre>, <radius>)) = 1
     AND visit IN (<the JSON's visits>)
   ```
   The cone covers the whole track plus a 10″ margin, so Qserv can use its spatial index;
   the visit list does the rest.
3. **Match** every app visit to the closest DiaSource *in the same visit* within
   `--match-radius` (default 2.0″). No designation lookup is needed.
4. **Label** the match: DP1's `ssObjectId` is the MPC packed designation stored as ASCII bytes,
   so it is unpacked directly (`21165823585761102` → `K24T57N` → `2024 TN57`).
   `ssObjectId = 0` means the pipeline detected something there but did not link it to any
   known object.
5. **Write** the CSV and plot, and print a summary.

### Which Rubin flux, and why

`DiaSource` carries several fluxes (see the DP1 tutorial `docs/notebooks/dp1/308_4_Photometry.ipynb`):

| Column | Measured on | Used here |
|---|---|---|
| `scienceFlux` | visit ("science") image, PSF fit at the DiaSource position | **Primary comparison.** Same pixels and calibration as the app. |
| `psfFlux` | difference image, PSF fit | Secondary column. In DP1 the templates often contain the moving object, which biases this flux, especially for `isDipole` detections. |
| `apFlux` | difference image, 12 px (≈2.4″) aperture | Not used: aperture differs from the app's 3/5/7″ apertures. |
| `trailFlux` | difference image, trailed PSF | Not used: no uncertainty in DP1. |

Only DiaSources with SNR > 5 exist, so faint visits simply have no Rubin counterpart.

### Usage (on the RSP)

```bash
python src/forcedphot/compare_rubin.py output/2024_TN57_photometry_results.json

# compare the forced (ephemeris-position) photometry instead of the nearest detection
python src/forcedphot/compare_rubin.py results.json --source forced

# looser match, custom output prefix, open the plot
python src/forcedphot/compare_rubin.py results.json --match-radius 3 -o output/tn57_check --show
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `json_file` | — | One result JSON file |
| `--source nearest\|forced` | `nearest` | App measurement to compare. `nearest` (PSF fit at the detected position) is the closest analogue of `scienceFlux`. |
| `--max-sep ARCSEC` | `1.0` | App match radius for `nearest` |
| `--match-radius ARCSEC` | `2.0` | Largest accepted app ↔ DiaSource separation |
| `--dr SCHEMA` | `dp1` | TAP schema to query. Other releases may use a different schema (e.g. DP2 links through `SSSource`), so check before relying on it. |
| `-o, --output PREFIX` | `<input>_rubin` | Writes `PREFIX.csv` and `PREFIX.png` |
| `--show` | off | Also open the figure in a window |

### Output CSV columns

One row per app visit; Rubin columns are empty/NaN when there is no match.

| Column | Meaning |
|---|---|
| `visit`, `detector`, `band`, `mjd` | From the JSON |
| `app_mag`, `app_mag_err` | App measurement (AB) |
| `sci_mag`, `sci_mag_err` | Rubin `scienceFlux` converted to AB (`31.4 − 2.5 log10(flux_nJy)`) |
| `dmag_sci` | `app_mag − sci_mag` |
| `pull_sci` | `(app_flux − scienceFlux) / hypot(errors)` |
| `psf_mag`, `psf_mag_err`, `dmag_psf` | Same for the difference-image `psfFlux` |
| `sep_arcsec` | App position ↔ DiaSource separation |
| `diaSourceId`, `ssObjectId`, `designation` | Rubin identifiers; designation unpacked from `ssObjectId` |
| `isDipole` | DiaSource dipole flag |

### Output plot

- **Left:** `dmag_sci` against MJD, error bars are the two errors in quadrature.
- **Right:** app mag against Rubin `scienceFlux` mag with the 1:1 line.
- Colour/marker per band as in `plot_lightcurve`; **hollow markers are `isDipole` detections.**

### Reading the results

- **`dmag_sci` near 0** across visits → the app agrees with the pipeline.
- **A consistent offset in `dmag_sci`** → look for a calibration or photometry bug in the app.
- **Offset only in `dmag_psf`** → expected in DP1 (template contamination), not the app's fault.
- **Pull scatter below 1** is normal: both measure the same pixels, so errors are correlated.
- **Unmatched visits** → target below SNR 5, a missed detection, or the app measured the wrong
  position (check `sep_arcsec` on neighbouring visits and the ephemeris uncertainty).
- **`ssObjectId = 0` matches** → the pipeline saw the source but did not link it.

### Limitations

- Designation unpacking handles provisional designations only (`2024 TN57`, `2007 TN102`);
  numbered objects and comets are shown packed.
- Only `isDipole` is reported, not the full set of DiaSource flags.
- Verified offline (tests plus a run on the real 2024 TN57 JSON with synthetic DiaSources);
  the TAP query has not yet been run end to end on the RSP.

### Tests

```bash
pytest tests/forcedphot/test_plot_lightcurve.py tests/forcedphot/test_compare_rubin.py
```

Both run without the LSST stack; `compare_rubin` imports `lsst.rsp` only inside the query
function.
