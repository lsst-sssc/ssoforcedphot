"""
Compare photometry result JSON files against the Rubin pipeline's own measurements.

For every visit in the JSON, the closest DiaSource in the same visit is looked up on
the RSP TAP service. The script writes a per-visit CSV and a comparison plot of the
app's PSF photometry against the pipeline's ``scienceFlux`` (PSF photometry on the
same visit image) and ``psfFlux`` (PSF photometry on the difference image).

For DP1 the science-image flux is the fair comparison: DP1 templates often contain
the moving object itself, which biases the difference-image flux (see the
308.4 Solar System object photometry tutorial).
"""

import argparse
import csv
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from plot_lightcurve import BAND_COLOR, BAND_MARKER, DEFAULT_MAX_SEP, select_photometry

# DiaSource match radius in arcseconds. The app and the pipeline measure the same
# pixels, so a genuine counterpart sits well inside this.
DEFAULT_MATCH_RADIUS = 2.0
AB_ZEROPOINT_NJY = 31.4

DIA_COLUMNS = (
    "diaSourceId, ssObjectId, visit, detector, ra, dec, "
    "psfFlux, psfFluxErr, scienceFlux, scienceFluxErr, isDipole"
)

CSV_FIELDS = [
    "visit", "detector", "band", "mjd", "app_mag", "app_mag_err",
    "sci_mag", "sci_mag_err", "dmag_sci", "pull_sci", "psf_mag", "psf_mag_err", "dmag_psf",
    "sep_arcsec", "diaSourceId", "ssObjectId", "designation", "isDipole",
]  # fmt: skip


def unpack_designation(ss_object_id):
    """
    Turn a DP1 ssObjectId into a readable MPC designation.

    DP1 builds ssObjectId from the ASCII bytes of the packed MPC designation
    (e.g. 21165823585761102 -> "K24T57N" -> "2024 TN57").

    Parameters
    ----------
    ss_object_id : int
        The ssObjectId; 0 means the detection is not linked to any object.

    Returns
    -------
    str
        The unpacked provisional designation, the packed string when it is not a
        provisional designation, or "" for 0.
    """
    if not ss_object_id:
        return ""
    packed = int(ss_object_id).to_bytes(8, "big").lstrip(b"\x00").decode("ascii", "replace")
    # ponytail: provisional designations only; numbered objects and comets come back packed.
    m = re.fullmatch(r"([IJK])(\d\d)([A-Z])([0-9A-Za-z])(\d)([A-Z])", packed)
    if not m:
        return packed
    century, year, half_month, cycle_hi, cycle_lo, letter = m.groups()
    cycle_hi = int(cycle_hi) if cycle_hi.isdigit() else ord(cycle_hi) - (55 if cycle_hi.isupper() else 61)
    cycle = cycle_hi * 10 + int(cycle_lo)
    return f"{ord(century) - 55}{year} {half_month}{letter}{cycle or ''}"


def flux_to_mag(flux, flux_err):
    """Convert a flux in nJy to an AB magnitude and error; NaN for non-positive flux."""
    if not flux or flux <= 0:
        return np.nan, np.nan
    return AB_ZEROPOINT_NJY - 2.5 * np.log10(flux), 1.0857 * flux_err / flux


def load_app_rows(path, source, max_sep=DEFAULT_MAX_SEP):
    """
    Read one photometry results JSON file into one row per usable visit.

    Parameters
    ----------
    path : str
        Photometry results JSON file.
    source : str
        "nearest" or "forced", see :func:`plot_lightcurve.select_photometry`.
    max_sep : float, optional
        Match radius in arcseconds for the "nearest" source.

    Returns
    -------
    list of dict
        Keys: visit, detector, band, mjd, ra, dec, flux, flux_err, mag, mag_err, target.
    """
    with open(path) as f:
        entries = json.load(f)
    if isinstance(entries, dict):
        entries = [entries]

    rows = []
    for entry in entries:
        phot = select_photometry(entry, source, max_sep)
        # The pipeline writes 0.0 where a measurement failed or was NaN.
        if not phot or not phot.get("flux") or "visit_id" not in entry:
            continue
        rows.append(
            {
                "visit": int(entry["visit_id"]),
                "detector": entry.get("detector_id"),
                "band": entry.get("band"),
                "mjd": float(entry["obs_time"]),
                "ra": phot["ra"],
                "dec": phot["dec"],
                "flux": phot["flux"],
                "flux_err": phot.get("flux_err") or 0.0,
                "mag": phot.get("mag"),
                "mag_err": phot.get("mag_err") or 0.0,
                "target": entry.get("target_name", "unknown"),
            }
        )
    return rows


def build_query(rows, dr="dp1", margin_arcsec=10.0):
    """
    Build the ADQL query for all DiaSources near the app positions in the app's visits.

    A single cone around the whole track keeps Qserv on its spatial index; the
    visit list does the rest.
    """
    coords = SkyCoord([r["ra"] for r in rows], [r["dec"] for r in rows], unit="deg")
    xyz = coords.cartesian.xyz.mean(axis=1)
    center = SkyCoord(*xyz, representation_type="cartesian").spherical
    center = SkyCoord(center.lon, center.lat)
    radius = coords.separation(center).max().deg + margin_arcsec / 3600
    visits = ", ".join(str(v) for v in sorted({r["visit"] for r in rows}))
    return (
        f"SELECT {DIA_COLUMNS} FROM {dr}.DiaSource "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {center.ra.deg:.7f}, {center.dec.deg:.7f}, {radius:.7f})) = 1 "
        f"AND visit IN ({visits})"
    )


def fetch_dia_sources(query):
    """Run the query on the RSP TAP service and return the rows as a list of dicts."""
    from lsst.rsp import get_tap_service

    service = get_tap_service("tap")
    job = service.submit_job(query)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR"])
    if job.phase == "ERROR":
        job.raise_if_error()
    table = job.fetch_result().to_table()
    job.delete()
    return [dict(zip(table.colnames, row)) for row in table]


def compare(app_rows, dia_rows, match_radius=DEFAULT_MATCH_RADIUS):
    """
    Pair every app visit with the closest DiaSource in the same visit.

    Parameters
    ----------
    app_rows : list of dict
        Rows from :func:`load_app_rows`.
    dia_rows : list of dict
        DiaSource rows with at least the columns in ``DIA_COLUMNS``.
    match_radius : float, optional
        Largest accepted separation in arcseconds.

    Returns
    -------
    list of dict
        One row per app visit with the ``CSV_FIELDS`` keys. Rubin columns are
        NaN / "" when the pipeline has no DiaSource there.
    """
    out = []
    for app in app_rows:
        row = dict.fromkeys(CSV_FIELDS, np.nan)
        row.update(
            visit=app["visit"],
            detector=app["detector"],
            band=app["band"],
            mjd=app["mjd"],
            app_mag=app["mag"],
            app_mag_err=app["mag_err"],
            diaSourceId="",
            ssObjectId="",
            designation="",
            isDipole="",
        )

        candidates = [d for d in dia_rows if int(d["visit"]) == app["visit"]]
        if candidates:
            target = SkyCoord(app["ra"], app["dec"], unit="deg")
            seps = target.separation(
                SkyCoord([d["ra"] for d in candidates], [d["dec"] for d in candidates], unit="deg")
            ).arcsec
            best = int(np.argmin(seps))
            if seps[best] <= match_radius:
                dia = candidates[best]
                sci_mag, sci_err = flux_to_mag(dia["scienceFlux"], dia["scienceFluxErr"])
                psf_mag, psf_err = flux_to_mag(dia["psfFlux"], dia["psfFluxErr"])
                # The app and the pipeline measure the same pixels, so their errors are
                # correlated and the pull is expected to scatter by less than 1.
                pull = (app["flux"] - dia["scienceFlux"]) / np.hypot(app["flux_err"], dia["scienceFluxErr"])
                row.update(
                    sci_mag=sci_mag,
                    sci_mag_err=sci_err,
                    dmag_sci=app["mag"] - sci_mag,
                    pull_sci=pull,
                    psf_mag=psf_mag,
                    psf_mag_err=psf_err,
                    dmag_psf=app["mag"] - psf_mag,
                    sep_arcsec=seps[best],
                    diaSourceId=int(dia["diaSourceId"]),
                    ssObjectId=int(dia["ssObjectId"]),
                    designation=unpack_designation(dia["ssObjectId"]),
                    isDipole=bool(dia["isDipole"]),
                )
        out.append(row)
    return out


def write_csv(rows, path):
    """Write the comparison rows to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def plot(rows, title, output, show=False):
    """
    Draw the comparison: residuals against time, and app against Rubin magnitudes.

    Hollow markers are DiaSources flagged ``isDipole``.
    """
    matched = [r for r in rows if np.isfinite(r["dmag_sci"])]
    fig, (ax_t, ax_1) = plt.subplots(1, 2, figsize=(12, 4.8))

    for band in dict.fromkeys(r["band"] for r in matched):
        for dipole in (False, True):
            sel = [r for r in matched if r["band"] == band and r["isDipole"] is dipole]
            if not sel:
                continue
            style = dict(
                linestyle="none",
                marker=BAND_MARKER.get(band, "o"),
                color=BAND_COLOR.get(band, "grey"),
                markerfacecolor="none" if dipole else BAND_COLOR.get(band, "grey"),
                markersize=7,
                elinewidth=1.2,
                capsize=0,
                label=f"{band} band" + (" (dipole)" if dipole else ""),
            )
            ax_t.errorbar(
                [r["mjd"] for r in sel],
                [r["dmag_sci"] for r in sel],
                yerr=[np.hypot(r["app_mag_err"], r["sci_mag_err"]) for r in sel],
                **style,
            )
            style.pop("label")
            ax_1.errorbar(
                [r["sci_mag"] for r in sel],
                [r["app_mag"] for r in sel],
                xerr=[r["sci_mag_err"] for r in sel],
                yerr=[r["app_mag_err"] for r in sel],
                **style,
            )

    ax_t.axhline(0, color="black", linewidth=0.8)
    ax_t.set_xlabel("MJD")
    ax_t.set_ylabel("app mag − Rubin scienceFlux mag")
    ax_t.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax_t.tick_params(axis="x", labelrotation=30)
    ax_t.legend(frameon=False, fontsize="small")

    if matched:
        lo = min(min(r["sci_mag"], r["app_mag"]) for r in matched) - 0.2
        hi = max(max(r["sci_mag"], r["app_mag"]) for r in matched) + 0.2
        ax_1.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
        ax_1.set_xlim(hi, lo)
        ax_1.set_ylim(hi, lo)
    ax_1.set_xlabel("Rubin scienceFlux (AB mag)")
    ax_1.set_ylabel("app (AB mag)")

    for ax in (ax_t, ax_1):
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


def summarize(rows):
    """Print how many visits matched and the median offsets."""
    matched = [r for r in rows if np.isfinite(r["dmag_sci"])]
    names = sorted({r["designation"] or "unlinked (ssObjectId 0)" for r in matched})
    print(f"{len(matched)}/{len(rows)} visits have a Rubin DiaSource within the match radius.")
    if matched:
        print(f"Linked to: {', '.join(names)}")
        print(f"Median app − scienceFlux: {np.nanmedian([r['dmag_sci'] for r in matched]):+.3f} mag")
        print(f"Median app − psfFlux:     {np.nanmedian([r['dmag_psf'] for r in matched]):+.3f} mag")
        print(f"Median pull (science):    {np.nanmedian([r['pull_sci'] for r in matched]):+.2f}")
        print(f"Dipole-flagged matches:   {sum(r['isDipole'] is True for r in matched)}")


def main():
    """Parse command line arguments, query the RSP and write the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare photometry result JSON against Rubin DiaSource measurements (RSP only)."
    )
    parser.add_argument("json_file", help="photometry results JSON file")
    parser.add_argument(
        "--source",
        choices=("nearest", "forced"),
        default="nearest",
        help="app photometry to compare: nearest detected source, or forced at the ephemeris position",
    )
    parser.add_argument(
        "--max-sep",
        type=float,
        default=DEFAULT_MAX_SEP,
        help="app match radius in arcsec for --source nearest (default: %(default)s)",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=DEFAULT_MATCH_RADIUS,
        help="DiaSource match radius in arcsec (default: %(default)s)",
    )
    parser.add_argument("--dr", default="dp1", help="data release schema (default: %(default)s)")
    parser.add_argument("-o", "--output", help="output prefix (default: <input>_rubin)")
    parser.add_argument("--show", action="store_true", help="also open the figure in a window")
    args = parser.parse_args()

    app_rows = load_app_rows(args.json_file, args.source, args.max_sep)
    if not app_rows:
        parser.error("no usable photometry found in the given file")

    dia_rows = fetch_dia_sources(build_query(app_rows, args.dr))
    rows = compare(app_rows, dia_rows, args.match_radius)

    prefix = args.output or os.path.splitext(args.json_file)[0] + "_rubin"
    write_csv(rows, prefix + ".csv")
    summarize(rows)
    plot(
        rows, f"{app_rows[0]['target']} — app ({args.source}) vs Rubin {args.dr}", prefix + ".png", args.show
    )


if __name__ == "__main__":
    main()
