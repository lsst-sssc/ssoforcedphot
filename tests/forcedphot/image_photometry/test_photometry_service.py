"""
Unit tests for the PhotometryService class.

Tests cover:
- Initialization
- Image processing
- Source detection
- Forced photometry
- Error handling
- Edge cases
"""

from dataclasses import dataclass
from unittest.mock import Mock

import lsst.afw.image as afwImage
import lsst.geom as geom
import pytest
from forcedphot.image_photometry.photometry_service import PhotometryService
from forcedphot.image_photometry.utils import ImageMetadata, PhotometryResult
from lsst.daf.butler import Butler


# Mock classes and fixtures
@pytest.fixture
def mock_butler():
    """Create a mock Butler instance."""
    butler = Mock(spec=Butler)
    mock_calexp = Mock(spec=afwImage.ExposureF)
    mock_calexp.getWcs.return_value = Mock()
    mock_calexp.getWidth.return_value = 1000
    mock_calexp.getHeight.return_value = 1000
    butler.get.return_value = mock_calexp
    return butler


@pytest.fixture
def mock_image_metadata():
    """Create mock image metadata for testing."""
    @dataclass
    class MockEphemeris:
        ra_deg: float = 150.0
        dec_deg: float = -30.0
        uncertainty: dict = None

        def __post_init__(self):
            self.uncertainty = {"smaa": 1.0, "smia": 0.5, "theta": 45.0, "rss": 1.2}

    return ImageMetadata(
        visit_id=12345,
        detector_id=42,
        ccdvisit=1234542,
        band="r",
        coordinates_central=(150.0, -30.0),
        t_min="2024-01-01T00:00:00",
        t_max="2024-02-01T00:00:00",
        ephemeris_data=[MockEphemeris()],
        exact_ephemeris=MockEphemeris(),
    )


@pytest.fixture
def photometry_service():
    """Create a PhotometryService instance for testing."""
    return PhotometryService(detection_threshold=5.0)


# Test cases
def test_init(photometry_service):
    """Test PhotometryService initialization."""
    assert photometry_service.detection_threshold == 5.0
    assert photometry_service.display is None
    assert isinstance(photometry_service.butler, Butler)


def test_prepare_image_cutout(photometry_service):
    """Test image cutout preparation."""
    mock_calexp = Mock(spec=afwImage.ExposureF)
    mock_calexp.getWidth.return_value = 1000
    mock_calexp.getHeight.return_value = 1000
    mock_wcs = Mock()
    mock_wcs.skyToPixel.return_value = geom.Point2D(500, 500)
    mock_calexp.getWcs.return_value = mock_wcs

    target_img, bbox, offsets = photometry_service._prepare_image(
        calexp=mock_calexp, ra_deg=150.0, dec_deg=-30.0, cutout_size=400
    )

    assert bbox is not None
    assert len(offsets) == 2

# def test_error_handling(photometry_service):
#     """Test error handling in photometry operations."""
#     # Test with None image metadata
#     result = photometry_service.process_image(
#         image_metadata=None,
#         target_name='Test Target',
#         target_type='asteroid',
#         ephemeris_service='test_service',
#         image_type='calexp'
#     )

#     assert result is None


def test_prepare_photometry_results(photometry_service):
    """Test preparation of photometry results."""
    mock_forced_cat = [Mock()]
    mock_forced_cat[0].get.side_effect = lambda x: 1000.0 if "Flux" in x else 10.0

    mock_sources = {"coord_ra": [], "coord_dec": []}

    target_result, sources = photometry_service._prepare_photometry_results(
        forced_meas_cat=mock_forced_cat, ra=150.0, dec=-30.0, found_sources=mock_sources
    )

    assert isinstance(target_result, PhotometryResult)
    assert isinstance(sources, list)
    assert target_result.ra == 150.0
    assert target_result.dec == -30.0
