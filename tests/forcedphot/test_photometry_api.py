"""
Unit tests for standalone photometry API module.

Tests the PhotometryRequest dataclass and StandalonePhotometryService class.
"""

import os
import tempfile

import pandas as pd
import pytest

try:
    from photometry_api import PhotometryRequest, StandalonePhotometryService

    LSST_AVAILABLE = True
except ImportError:
    LSST_AVAILABLE = False
    ImageType = None
    PhotometryRequest = None
    StandalonePhotometryService = None


@pytest.mark.skipif(not LSST_AVAILABLE, reason="Requires LSST environment")
class TestPhotometryRequest:
    """Tests for PhotometryRequest dataclass."""

    def test_photometry_request_creation(self):
        """Test creation of PhotometryRequest with required fields."""
        request = PhotometryRequest(
            visit_id=512055,
            detector=75,
            band="g",
            ra=150.123,
            dec=-23.456,
        )

        assert request.visit_id == 512055
        assert request.detector == 75
        assert request.band == "g"
        assert request.ra == 150.123
        assert request.dec == -23.456
        assert request.error_radius == 3.0  # default
        assert request.detection_threshold == 5.0  # default
        assert request.image_type == "visit_image"  # default
        assert request.aperture_radii is None  # default
        assert request.target_name == "standalone_target"  # default

    def test_photometry_request_with_optional_params(self):
        """Test PhotometryRequest with all optional parameters."""
        request = PhotometryRequest(
            visit_id=512055,
            detector=75,
            band="g",
            ra=150.123,
            dec=-23.456,
            error_radius=5.0,
            detection_threshold=10.0,
            image_type="difference_image",
            aperture_radii=[3.0, 5.0, 7.0],
            target_name="my_target",
        )

        assert request.error_radius == 5.0
        assert request.detection_threshold == 10.0
        assert request.image_type == "difference_image"
        assert request.aperture_radii == [3.0, 5.0, 7.0]
        assert request.target_name == "my_target"


@pytest.mark.skipif(not LSST_AVAILABLE, reason="Requires LSST Butler/RSP environment")
class TestStandalonePhotometryService:
    """Tests for StandalonePhotometryService class."""

    @pytest.fixture
    def service(self):
        """Create service instance for testing (without Butler connection)."""
        # Note: This will fail without LSST environment, so we skip it in conftest.py
        return StandalonePhotometryService(
            dr="dp1",
            collection="LSSTComCam/DP1",
            output_folder="./test_output",
        )

    def test_validate_coordinates_valid(self, service):
        """Test coordinate validation with valid coordinates."""
        # Valid coordinates should not raise
        assert service._validate_coordinates(150.0, -23.0) is True
        assert service._validate_coordinates(0.0, 0.0) is True
        assert service._validate_coordinates(359.999, 90.0) is True
        assert service._validate_coordinates(180.0, -90.0) is True

    def test_validate_coordinates_invalid_ra(self, service):
        """Test coordinate validation with invalid RA."""
        with pytest.raises(ValueError, match="RA must be in range"):
            service._validate_coordinates(-10.0, -23.0)

        with pytest.raises(ValueError, match="RA must be in range"):
            service._validate_coordinates(400.0, -23.0)

    def test_validate_coordinates_invalid_dec(self, service):
        """Test coordinate validation with invalid Dec."""
        with pytest.raises(ValueError, match="Dec must be in range"):
            service._validate_coordinates(150.0, -100.0)

        with pytest.raises(ValueError, match="Dec must be in range"):
            service._validate_coordinates(150.0, 100.0)

    def test_results_to_dataframe_empty(self, service):
        """Test DataFrame conversion with empty results."""
        requests = []
        results = []
        df = service._results_to_dataframe(results, requests)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_results_to_dataframe_failed_result(self, service):
        """Test DataFrame conversion with failed measurement."""
        request = PhotometryRequest(
            visit_id=512055,
            detector=75,
            band="g",
            ra=150.123,
            dec=-23.456,
            target_name="test_target",
        )
        results = [None]
        requests = [request]

        df = service._results_to_dataframe(results, requests)

        assert len(df) == 1
        assert not df.iloc[0]["success"]
        assert df.iloc[0]["visit_id"] == 512055
        assert df.iloc[0]["target_name"] == "test_target"

    def test_csv_parsing(self):
        """Test CSV file parsing without processing."""
        # Create test CSV
        csv_content = """visit_id,detector,band,ra,dec,error_radius,target_name
512055,75,g,150.123,-23.456,5.0,target1
512060,75,r,150.234,-23.567,3.0,target2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_file = f.name

        try:
            # Read and parse (without executing photometry)
            df = pd.read_csv(csv_file)

            assert len(df) == 2
            assert list(df["visit_id"]) == [512055, 512060]
            assert list(df["band"]) == ["g", "r"]
            assert list(df["ra"]) == [150.123, 150.234]
            assert list(df["target_name"]) == ["target1", "target2"]
        finally:
            os.unlink(csv_file)

    def test_csv_parsing_with_aperture_radii(self):
        """Test CSV parsing with comma-separated aperture radii."""
        csv_content = """visit_id,detector,band,ra,dec,aperture_radii
512055,75,g,150.123,-23.456,"3.0, 5.0, 7.0"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_file = f.name

        try:
            df = pd.read_csv(csv_file)
            aperture_radii_str = df.iloc[0]["aperture_radii"]

            # Parse aperture radii
            aperture_radii = [float(x.strip()) for x in str(aperture_radii_str).split(",")]

            assert aperture_radii == [3.0, 5.0, 7.0]
        finally:
            os.unlink(csv_file)

    @pytest.mark.skipif(not LSST_AVAILABLE, reason="Requires LSST Butler/RSP environment")
    def test_measure_single_real(self, service):
        """Integration test with real Butler data (requires RSP)."""
        request = PhotometryRequest(
            visit_id=2024112300235,
            detector=2,
            band="i",
            ra=38.6151529929,
            dec=7.424556805,
        )

        result = service.measure_single(request=request)

        assert result is not None
        assert result.visit_id == 2024112300235
        assert result.forced_phot_on_target is not None

    @pytest.mark.skipif(not LSST_AVAILABLE, reason="Requires LSST Butler/RSP environment")
    def test_get_image_time_info_real(self, service):
        """Test timing information retrieval with real Butler (requires RSP)."""
        info = service._get_image_time_info(2024112300235, 2)

        assert "begin_time" in info
        assert "end_time" in info
        assert "mid_time" in info
        assert info["mid_time"].scale == "tai"

    @pytest.mark.skipif(not LSST_AVAILABLE, reason="Requires LSST Butler/RSP environment")
    def test_measure_from_csv_real(self, service):
        """Integration test for CSV batch processing (requires RSP)."""
        csv_content = """visit_id,detector,band,ra,dec,error_radius,taget_name
        2024112300235,2,i,38.6151529929,7.424556805,15.0,target1
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_file = f.name

        try:
            results_df = service.measure_from_csv(csv_path=csv_file, output_folder="./test_output")

            assert len(results_df) == 1
            assert "forced_flux" in results_df.columns
            assert "success" in results_df.columns
        finally:
            os.unlink(csv_file)
