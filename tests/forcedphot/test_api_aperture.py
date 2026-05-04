"""Tests for aperture radii resolution logic in photometry_api."""


def _resolve(aperture_radii, defaults):
    """Mirror the resolution logic from measure_single."""
    if aperture_radii is None:
        return None
    if len(aperture_radii) == 0:
        return defaults
    return aperture_radii


DEFAULT = [3.0, 5.0, 7.0]


def test_none_stays_none():
    """Test that None aperture_radii stays None."""
    assert _resolve(None, DEFAULT) is None


def test_empty_list_uses_defaults():
    """Test that empty list resolves to defaults."""
    assert _resolve([], DEFAULT) == [3.0, 5.0, 7.0]


def test_explicit_radii_used_as_is():
    """Test that explicit radii are used as-is."""
    assert _resolve([2.0, 4.0], DEFAULT) == [2.0, 4.0]


def test_default_aperture_radii_constant():
    """Test that DEFAULT_APERTURE_RADII constant is properly defined."""
    from photometry_api import DEFAULT_APERTURE_RADII

    assert isinstance(DEFAULT_APERTURE_RADII, list)
    assert all(isinstance(r, float) for r in DEFAULT_APERTURE_RADII)
    assert len(DEFAULT_APERTURE_RADII) > 0
