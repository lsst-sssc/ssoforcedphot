"""Help text for the Panel control panel UI.

Pure data plus a tiny formatter — no Panel or LSST imports — so it can be
unit-tested in CI without the Rubin Science Platform. ``control_panel.py``
wires these strings into widget ``description=`` tooltips and per-tab tip cards.
"""

# One short line per non-obvious widget (rendered as a hover (?) tooltip).
TOOLTIPS = {
    # Ephemeris
    "ephemeris_source": (
        "'Use Existing Data' runs a live query to the selected service; "
        "'Upload ECSV' loads a pre-computed ephemeris file instead."
    ),
    "service": "Ephemeris provider for live queries — JPL Horizons or IMCCE Miriade.",
    "target_name": "Object identifier, e.g. '2024 TN57', 'Ceres', or 'C/2020 F3 (NEOWISE)'.",
    "target_type": "How the name is resolved: smallbody, asteroid_name, comet_name, or designation.",
    "time_spec": "Define the span by an explicit End Time, or a Day Range (N days from start).",
    "day_range": "Number of days forward from the start time.",
    "step_value": (
        "Time between ephemeris points. Smaller steps = more rows; over a long range this "
        "creates very large tables that slow or break the image search — prefer a larger step "
        "plus 'Refine Ephemeris'."
    ),
    "step_unit": "Step unit: d (days), h (hours), m (minutes).",
    "save_ephem_data": "Write the queried ephemeris table to the output folder as ECSV.",
    "output_folder": "Directory for saved files. Must exist and be writable.",
    # Image search
    "image_search_method": (
        "Point checks each ephemeris row (slow for many rows); Polygon builds one sky polygon "
        "over the track (faster for long/dense ephemerides)."
    ),
    "filters": "Photometric bands to search (u g r i z y). Select only the bands you need.",
    "widening": "Expand the polygon search area around the track, in arcsec.",
    "time_interval": "Group ephemeris points into segments of this many days when building polygons.",
    # Photometry (shared across Photometry / Complete Run / Standalone)
    "image_type": (
        "visit_image = calibrated single-epoch image; difference_image = template-subtracted "
        "(target may appear as a dipole/negative)."
    ),
    "detection_threshold": "Minimum signal-to-noise for a pixel cluster to count as a detected source.",
    "cutout_provider": (
        "Butler = load full exposure and slice locally; SODA = request a server-side cutout "
        "(lighter for large images)."
    ),
    "cutout_size": "Square cutout side in pixels (Butler). 0 = use the whole image (slow, memory-heavy).",
    "cutout_radius": "Cutout radius in arcsec for the SODA server-side request.",
    "override_error": (
        "Replace the ephemeris error ellipse with a circular search radius (arcsec). "
        '0 = use the ephemeris uncertainty. Try 3-20" if the target falls outside the ellipse.'
    ),
    "refine_ephemeris": (
        "Re-query the ephemeris at each exact observation time instead of interpolating. "
        "Recommended for fast movers (NEOs, comets); adds query time."
    ),
    "run_aperture": "Also measure flux in fixed circular apertures, in addition to PSF forced photometry.",
    "aperture_radii": (
        "Circular aperture radii in arcsec, e.g. [3.0, 5.0, 7.0]. Used when Aperture Photometry is on."
    ),
    "save_csv_all_sources": "Write one CSV row per source in the error ellipse, not just the target row.",
    # Standalone
    "input_mode": "Single Coordinate, Batch CSV (uploaded file), or Multiple in one image.",
    "error_radius": "Search radius (arcsec) around each coordinate for nearby-source detection.",
}

# Gotchas / performance heuristics, shown in a collapsible card per tab.
TAB_TIPS = {
    "ephemeris": [
        "**Step size matters.** A long time range with a small step produces thousands of rows, "
        "which makes the image search slow and can fail. Prefer a coarser step (e.g. >= 1h) and "
        "turn on **Refine Ephemeris** in the Photometry step to keep accuracy.",
        "**Name & type must match.** e.g. 'C/2020 F3 (NEOWISE)' with type *comet_name*, "
        "'2024 TN57' with *smallbody*. A mismatched type is the most common query failure.",
        "Times use the format `YYYY-MM-DD HH:MM:SS`.",
    ],
    "image": [
        "**Use Polygon for long or dense tracks.** Point search checks every ephemeris row and "
        "gets slow (or fails) when there are many; Polygon searches one widened track polygon and "
        "is much faster.",
        "If a search is slow or fails, go back and **coarsen the ephemeris step**, then use "
        "Refine Ephemeris later.",
        "Select only the bands you actually need — each adds queries.",
    ],
    "photometry": [
        "**Fast movers:** enable **Refine Ephemeris** for NEOs/comets. Without it, interpolation "
        "can place the target outside the error ellipse and it is missed.",
        "**Can't find the target?** Raise **Override Error Ellipse** (start 3-20\") to widen the "
        "search, or lower the **Detection Threshold**.",
        "**Cutouts:** Butler size is in *pixels* (0 = full image, heavy); SODA radius is in "
        "*arcsec* and is lighter for big images.",
        "**difference_image** subtracts a template — a real moving source can appear as a "
        "positive/negative dipole.",
    ],
    "complete_run": [
        "Runs everything end-to-end with no stops to inspect intermediate tables. For large jobs, "
        "set a **coarser step**, use **Polygon** search and **Refine Ephemeris**.",
        "All the Photometry-tab tips apply here too.",
    ],
    "standalone": [
        "No ephemeris needed — measure flux at coordinates you already have.",
        "**Batch CSV** columns: `visit_id, detector, band, ra, dec` "
        "(optional `error_radius, target_name, aperture_radii`). A per-row `aperture_radii` "
        "overrides the global setting.",
        "**Error Radius** sets how far around each coordinate to look for nearby sources.",
    ],
}


def tips_markdown(tab):
    """Return a tab's tips as a markdown bullet list, or '' if the tab is unknown."""
    bullets = TAB_TIPS.get(tab)
    if not bullets:
        return ""
    return "\n".join(f"- {bullet}" for bullet in bullets)


# Warn when an ephemeris has more rows than this (image search gets slow / fails).
ROW_COUNT_WARN_THRESHOLD = 500


def row_count_warning(n, threshold=ROW_COUNT_WARN_THRESHOLD):
    """Return a warning string when n exceeds threshold, else None."""
    if n > threshold:
        return (
            f"⚠ {n:,} ephemeris rows — image search may be slow or fail. "
            "Consider a coarser step size, the Polygon search method, and Refine Ephemeris."
        )
    return None
