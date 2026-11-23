"""
Quick-look light curve plots from photometry result JSON files.

Reads one or more JSON files written by the photometry service and plots
brightness (AB magnitude or flux) against observation time (MJD), with error
bars and one colour per band.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt

# Rubin band order. The colours are the Rubin filter hues, adjusted so the set
# stays distinguishable under colour-vision deficiency on a light background;
# the markers give a second, colour-independent encoding.
BAND_ORDER = ("u", "g", "r", "i", "z", "y")
BAND_COLOR = {
    "u": "#1600EA",
    "g": "#31DE1F",
    "r": "#B52626",
    "i": "#370201",
    "z": "#BA52FF",
    "y": "#61A2B3",
}
BAND_MARKER = {"u": "o", "g": "s", "r": "^", "i": "D", "z": "v", "y": "P"}

Y_LABEL = {"mag": "AB magnitude", "flux": "Flux (nJy)"}

# Match radius in arcseconds. Real counterparts land well inside an arcsecond of
# the ephemeris position; anything further out is a field source, not the target.
DEFAULT_MAX_SEP = 1.0


def select_photometry(entry, source, max_sep=DEFAULT_MAX_SEP):
    """
    Pick the photometry measurement to plot for a single visit.

    Parameters
    ----------
    entry : dict
        One visit record from a photometry results JSON file.
    source : str
        Either "nearest" (the detected source closest to the ephemeris
        position) or "forced" (forced photometry at the ephemeris position).
    max_sep : float, optional
        Match radius in arcseconds. In "nearest" mode a detection further away
        than this is treated as an unrelated field source, and the forced
        measurement is used instead.

    Returns
    -------
    dict or None
        The selected photometry dictionary, or None if the visit holds none.
    """
    if source == "nearest":
        detections = entry.get("phot_within_error_ellipse") or []
        if detections:
            closest = min(detections, key=lambda d: d.get("separation", float("inf")))
            if closest.get("separation", float("inf")) <= max_sep:
                return closest
    return entry.get("forced_phot_on_target")


def load_points(paths, source, bands, quantity, max_sep=DEFAULT_MAX_SEP):
    """
    Collect the plottable measurements from a set of result files.

    Parameters
    ----------
    paths : list of str
        Paths to photometry results JSON files.
    source : str
        Photometry to use, "nearest" or "forced".
    bands : list of str or None
        Bands to keep. None keeps every band found.
    quantity : str
        Either "mag" or "flux".
    max_sep : float, optional
        Match radius in arcseconds, passed to :func:`select_photometry`.

    Returns
    -------
    list of dict
        One dict per usable visit with keys "mjd", "value", "error", "band"
        and "target". Visits whose measurement failed (value 0 or missing) are
        dropped.
    """
    points = []
    fallbacks = 0

    for path in paths:
        with open(path) as f:
            entries = json.load(f)
        if isinstance(entries, dict):
            entries = [entries]

        for entry in entries:
            band = entry.get("band")
            if bands and band not in bands:
                continue

            phot = select_photometry(entry, source, max_sep)
            if not phot:
                continue

            # The pipeline writes 0.0 where a measurement failed or was NaN.
            value = phot.get(quantity)
            if not value:
                continue

            if source == "nearest" and phot is entry.get("forced_phot_on_target"):
                fallbacks += 1

            points.append(
                {
                    "mjd": float(entry["obs_time"]),
                    "value": value,
                    "error": phot.get(f"{quantity}_err") or 0.0,
                    "band": band,
                    "target": entry.get("target_name", "unknown"),
                }
            )

    if fallbacks:
        print(
            f'Note: {fallbacks} visit(s) had no detection within {max_sep}" of the ephemeris '
            "position; used forced photometry there."
        )

    return points


def plot(points, quantity, source, separate, output, show=False):
    """
    Draw and save the light curve.

    Parameters
    ----------
    points : list of dict
        Measurements from :func:`load_points`.
    quantity : str
        Either "mag" or "flux". Magnitude axes are inverted.
    source : str
        Photometry used, shown in the plot title.
    separate : bool
        If True, give each band its own stacked subplot.
    output : str
        Path of the PNG file to write.
    show : bool, optional
        Also open the figure in a window.
    """
    bands = [b for b in BAND_ORDER if any(p["band"] == b for p in points)]
    targets = sorted({p["target"] for p in points})

    rows = len(bands) if separate else 1
    height = 2.6 * rows if separate else 5
    fig, axes = plt.subplots(rows, 1, sharex=True, squeeze=False, figsize=(9, height))
    axes = list(axes[:, 0])

    for i, band in enumerate(bands):
        ax = axes[i] if separate else axes[0]
        selected = [p for p in points if p["band"] == band]
        ax.errorbar(
            [p["mjd"] for p in selected],
            [p["value"] for p in selected],
            yerr=[p["error"] for p in selected],
            linestyle="none",
            marker=BAND_MARKER[band],
            color=BAND_COLOR[band],
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            elinewidth=1.5,
            capsize=0,
            label=f"{band} band",
        )

    for i, ax in enumerate(axes):
        if quantity == "mag":
            ax.invert_yaxis()
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(Y_LABEL[quantity])
        if separate:
            ax.set_title(f"{bands[i]} band", loc="left", fontsize=10)
        else:
            ax.legend(frameon=False)
    # Show full MJDs rather than matplotlib's "+6.0652e4" offset notation.
    axes[-1].ticklabel_format(axis="x", useOffset=False, style="plain")
    axes[-1].set_xlabel("MJD")

    fig.suptitle(f"{', '.join(targets)} — {source} photometry")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    """Parse command line arguments and write a light curve plot."""
    parser = argparse.ArgumentParser(description="Plot a light curve from photometry result JSON files.")
    parser.add_argument("json_files", nargs="+", help="photometry results JSON file(s)")
    parser.add_argument(
        "--bands", nargs="+", choices=BAND_ORDER, help="bands to plot (default: every band present)"
    )
    parser.add_argument(
        "--y", dest="quantity", choices=("mag", "flux"), default="mag", help="quantity to plot"
    )
    parser.add_argument(
        "--source",
        choices=("nearest", "forced"),
        default="nearest",
        help="nearest detected source in the error ellipse, or forced photometry at the ephemeris position",
    )
    parser.add_argument(
        "--max-sep",
        type=float,
        default=DEFAULT_MAX_SEP,
        help="match radius in arcsec for --source nearest (default: %(default)s)",
    )
    parser.add_argument("--separate", action="store_true", help="give each band its own subplot")
    parser.add_argument("-o", "--output", help="output PNG (default: <first input>_lightcurve.png)")
    parser.add_argument("--show", action="store_true", help="also open the figure in a window")
    args = parser.parse_args()

    points = load_points(args.json_files, args.source, args.bands, args.quantity, args.max_sep)
    if not points:
        parser.error("no usable photometry found in the given file(s)")

    output = args.output or os.path.splitext(args.json_files[0])[0] + "_lightcurve.png"
    plot(points, args.quantity, args.source, args.separate, output, args.show)


if __name__ == "__main__":
    main()