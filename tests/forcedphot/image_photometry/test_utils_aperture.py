"""Tests for AperturePhotometryResult and PhotometryResult.aperture field."""

from astropy.time import Time
from image_photometry.utils import (
    AperturePhotometryResult,
    EndResult,
    PhotometryResult,
)


def _make_phot_result(**kwargs):
    defaults = dict(
        ra=53.0,
        dec=-28.0,
        ra_err=0.0,
        dec_err=0.0,
        x=100.0,
        y=100.0,
        x_err=0.0,
        y_err=0.0,
        snr=10.0,
        flux=500.0,
        flux_err=50.0,
        mag=22.5,
        mag_err=0.1,
        separation=0.0,
        sigma=0.0,
        flags={},
    )
    defaults.update(kwargs)
    return PhotometryResult(**defaults)


def test_aperture_result_fields():
    """Test AperturePhotometryResult dataclass fields."""
    r = AperturePhotometryResult(
        radius_arcsec=3.0,
        flux=100.0,
        flux_err=5.0,
        snr=20.0,
        mag=22.5,
        mag_err=0.05,
        flag=False,
    )
    assert r.radius_arcsec == 3.0
    assert r.flux == 100.0
    assert r.flux_err == 5.0
    assert r.snr == 20.0
    assert r.mag == 22.5
    assert r.mag_err == 0.05
    assert r.flag is False


def test_aperture_result_flag_true():
    """Test AperturePhotometryResult with flag=True."""
    r = AperturePhotometryResult(
        radius_arcsec=5.0,
        flux=0.0,
        flux_err=0.0,
        snr=0.0,
        mag=0.0,
        mag_err=0.0,
        flag=True,
    )
    assert r.flag is True


def test_photometry_result_aperture_defaults_to_none():
    """Test that aperture field defaults to None in PhotometryResult."""
    result = _make_phot_result()
    assert result.aperture is None


def test_photometry_result_aperture_can_hold_list():
    """Test that aperture field can hold a list of AperturePhotometryResult."""
    ap1 = AperturePhotometryResult(3.0, 100.0, 5.0, 20.0, 22.5, 0.05, False)
    ap2 = AperturePhotometryResult(5.0, 200.0, 8.0, 25.0, 22.0, 0.04, False)
    result = _make_phot_result(aperture=[ap1, ap2])
    assert len(result.aperture) == 2
    assert result.aperture[0].radius_arcsec == 3.0
    assert result.aperture[1].radius_arcsec == 5.0


def _make_end_result(target_phot=None, ellipse_sources=None, aperture=None):
    if target_phot is None:
        ap_list = aperture  # may be None
        target_phot = _make_phot_result(aperture=ap_list)
    return EndResult(
        target_name="test",
        target_type="asteroid",
        image_type="visit_image",
        ephemeris_service="Horizons",
        visit_id=12345,
        detector_id=1,
        band="r",
        coordinates_central=(53.0, -28.0),
        obs_time=Time("2024-01-01", scale="utc"),
        cutout_size=800,
        saved_image_name="",
        uncertainty={"rss": 1.0, "smaa": 1.0, "smia": 0.5, "theta": 0.0},
        forced_phot_on_target=target_phot,
        phot_within_error_ellipse=ellipse_sources or [],
    )


def test_csv_row_no_aperture():
    """Test CSV row with no aperture photometry results."""
    result = _make_end_result()
    row = result.to_csv_row()
    # No aperture columns present when aperture is None
    assert not any(k.startswith("aperture_") for k in row)


# def test_csv_row_with_aperture_on_target():
#     """Test CSV row serialization of aperture photometry on target."""
#     ap = AperturePhotometryResult(3.0, 100.0, 5.0, 20.0, 22.5, 0.05, False)
#     result = _make_end_result(aperture=[ap])
#     row = result.to_csv_row()
#     assert row["aperture_3_0arcsec_flux"] == 100.0
#     assert row["aperture_3_0arcsec_flux_err"] == 5.0
#     assert row["aperture_3_0arcsec_snr"] == 20.0
#     assert row["aperture_3_0arcsec_mag"] == 22.5
#     assert row["aperture_3_0arcsec_mag_err"] == 0.05
#     assert row["aperture_3_0arcsec_flag"] is False


# def test_csv_row_multiple_aperture_radii():
#     """Test CSV row serialization with multiple aperture radii."""
#     ap1 = AperturePhotometryResult(3.0, 100.0, 5.0, 20.0, 22.5, 0.05, False)
#     ap2 = AperturePhotometryResult(5.0, 200.0, 8.0, 25.0, 22.0, 0.04, False)
#     result = _make_end_result(aperture=[ap1, ap2])
#     row = result.to_csv_row()
#     assert "aperture_3_0arcsec_flux" in row
#     assert "aperture_5_0arcsec_flux" in row
#     assert row["aperture_5_0arcsec_flux"] == 200.0


def test_csv_row_aperture_on_ellipse_source():
    """Test CSV row serialization of aperture photometry on ellipse source."""
    ap = AperturePhotometryResult(3.0, 80.0, 4.0, 20.0, 22.8, 0.05, False)
    ellipse_src = _make_phot_result(aperture=[ap])
    result = _make_end_result(ellipse_sources=[ellipse_src])
    row = result.to_csv_row()
    assert row["ellipse_source_aperture_3_0arcsec_flux"] == 80.0
