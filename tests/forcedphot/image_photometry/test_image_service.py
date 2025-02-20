from unittest.mock import patch

import pandas as pd
import pytest
from astropy.time import Time
from forcedphot.image_photometry.image_service import ImageService
from forcedphot.image_photometry.utils import EphemerisDataCompressed, ImageMetadata


@pytest.fixture
def image_service():
    """Image service"""
    return ImageService()


@pytest.fixture
def mock_tap_service():
    """Mock TAP service"""
    with patch("forcedphot.image_photometry.image_service.get_tap_service") as mock:
        yield mock


@pytest.fixture
def sample_ephemeris_data():
    """Sample ephemeris data as fixture"""
    return [
        EphemerisDataCompressed(
            datetime=Time("2023-01-01 00:00:00"),
            ra_deg=10.0,
            dec_deg=20.0,
            ra_rate=0.1,
            dec_rate=0.1,
            uncertainty={"rss": 0.1, "smaa": 0.1, "smia": 0.1, "theta": 0.1},
        ),
        EphemerisDataCompressed(
            datetime=Time("2023-01-02 00:00:00"),
            ra_deg=10.1,
            dec_deg=20.1,
            ra_rate=0.1,
            dec_rate=0.1,
            uncertainty={"rss": 0.1, "smaa": 0.1, "smia": 0.1, "theta": 0.1},
        ),
    ]


def test_search_images_no_ephemeris(image_service):
    """Test search_images with empty ephemeris data"""
    with patch("forcedphot.image_photometry.utils.EphemerisDataCompressed.load_ephemeris", return_value=[]):
        result = image_service.search_images(bands={"g"}, ephemeris_data="nonexistent.ecsv")
        assert result is None


def test_search_images_with_results(image_service, sample_ephemeris_data):
    """Test search_images with valid results"""

    # Mock dependencies
    with patch(
        "forcedphot.image_photometry.utils.EphemerisDataCompressed.load_ephemeris",
        return_value=sample_ephemeris_data,
    ):
        with patch.object(image_service, "_execute_query") as mock_execute:
            # Create sample DataFrame result
            df = pd.DataFrame(
                {
                    "lsst_visit": [1],
                    "lsst_detector": [2],
                    "lsst_ccdvisitid": [3],
                    "lsst_band": ["g"],
                    "s_ra": [10.0],
                    "s_dec": [20.0],
                    "t_min": [59580.0],  # MJD for 2023-01-01
                    "t_max": [59581.0],  # MJD for 2023-01-02
                }
            )
            mock_execute.return_value = df

            result = image_service.search_images(
                bands={"g"}, ephemeris_data="../ephemeris/data/template_ephemeris.ecsv"
            )

            assert result is not None
            assert len(result) == 1
            assert isinstance(result[0], ImageMetadata)
            assert result[0].band == "g"
