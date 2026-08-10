"""Tests for the Panel UI help-text content module (ui_help)."""

from ui_help import (
    ROW_COUNT_WARN_THRESHOLD,
    TAB_TIPS,
    TOOLTIPS,
    row_count_warning,
    tips_markdown,
)

EXPECTED_TABS = {"ephemeris", "image", "photometry", "complete_run", "standalone"}

# Every TOOLTIPS key referenced by control_panel.py must exist.
EXPECTED_TOOLTIP_KEYS = {
    "ephemeris_source",
    "service",
    "target_name",
    "target_type",
    "time_spec",
    "day_range",
    "step_value",
    "step_unit",
    "save_ephem_data",
    "output_folder",
    "image_search_method",
    "filters",
    "widening",
    "time_interval",
    "image_type",
    "detection_threshold",
    "cutout_provider",
    "cutout_size",
    "cutout_radius",
    "override_error",
    "refine_ephemeris",
    "run_aperture",
    "aperture_radii",
    "save_csv_all_sources",
    "input_mode",
    "error_radius",
}


def test_tooltips_present_and_short():
    """All expected tooltip keys are present, non-empty strings within the 200-char limit."""
    assert EXPECTED_TOOLTIP_KEYS.issubset(
        TOOLTIPS
    ), f"missing tooltip keys: {EXPECTED_TOOLTIP_KEYS - set(TOOLTIPS)}"
    for key, text in TOOLTIPS.items():
        assert isinstance(text, str), f"tooltip {key} must be a string"
        assert text.strip() == text and text, f"tooltip {key} must be trimmed and non-empty"
        assert len(text) <= 200, f"tooltip {key} too long ({len(text)}) for a hover tooltip"


def test_tab_tips_structure():
    """TAB_TIPS has exactly the expected tabs, each with a non-empty list of non-empty strings."""
    assert set(TAB_TIPS) == EXPECTED_TABS
    for tab, bullets in TAB_TIPS.items():
        assert isinstance(bullets, list) and bullets, f"{tab} must have a non-empty list"
        for bullet in bullets:
            assert isinstance(bullet, str) and bullet.strip(), f"empty tip in {tab}"


def test_tips_markdown_joins_each_bullet():
    """tips_markdown returns a string containing every bullet for the tab."""
    md = tips_markdown("photometry")
    for bullet in TAB_TIPS["photometry"]:
        assert bullet in md
    assert "Refine Ephemeris" in md


def test_tips_markdown_image_mentions_polygon():
    """The image-tab tips mention Polygon search."""
    assert "Polygon" in tips_markdown("image")


def test_tips_markdown_unknown_tab_returns_empty():
    """tips_markdown returns an empty string for an unrecognised tab name."""
    assert tips_markdown("does-not-exist") == ""


def test_row_count_threshold_is_500():
    """ROW_COUNT_WARN_THRESHOLD is exactly 500."""
    assert ROW_COUNT_WARN_THRESHOLD == 500


def test_row_count_warning_none_at_or_below_threshold():
    """row_count_warning returns None when n is at or below the threshold."""
    assert row_count_warning(0) is None
    assert row_count_warning(500) is None


def test_row_count_warning_above_threshold():
    """row_count_warning returns a non-None string mentioning the count and Polygon."""
    msg = row_count_warning(501)
    assert msg is not None
    assert "501" in msg
    assert "Polygon" in msg


def test_row_count_warning_formats_thousands_and_names_remedies():
    """Large counts are formatted with commas and the message names Refine Ephemeris."""
    msg = row_count_warning(3200)
    assert "3,200" in msg
    assert "Refine Ephemeris" in msg


def test_row_count_warning_honors_custom_threshold():
    """row_count_warning respects an explicit threshold argument."""
    assert row_count_warning(100, threshold=50) is not None
    assert row_count_warning(40, threshold=50) is None
