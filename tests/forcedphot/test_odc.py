from unittest.mock import MagicMock, patch

import astropy.units as u
import pytest
from astropy.time import Time
from forcedphot.ephemeris.data_model import QueryResult
from forcedphot.image_photometry.utils import ImageMetadata
from forcedphot.odc import ObjectDetectionController


@pytest.fixture
def odc_instance():
    """Fixture to create an instance of ObjectDetectionController with mocked dependencies."""
    controller = ObjectDetectionController()
    controller.ephemeris_client = MagicMock()
    controller.imphot_controller = MagicMock()
    controller.image_service = MagicMock()
    # Initialize args with default values
    controller.parse_args([])
    return controller

def test_parse_args_with_day_range(odc_instance):
    """Test parsing arguments with --start-time and --day-range."""
    args = [
        "--start-time", "2023-01-01 00:00:00",
        "--day-range", "5"
    ]
    parsed = odc_instance.parse_args(args)
    assert parsed.start_time == Time("2023-01-01 00:00:00", scale="utc")
    assert parsed.end_time == parsed.start_time + 5 * u.day

def test_parse_args_invalid_input(odc_instance):
    """Test error raised when required ephemeris args are missing."""
    odc_instance.parse_args(["--service-selection", "ephemeris"])
    with pytest.raises(SystemExit):
        odc_instance.run_ephemeris_query()

@patch("forcedphot.odc.EphemerisClient")
def test_run_ephemeris_query_ecsv(mock_load, odc_instance):
    """Test ephemeris query with --ephem-ecsv."""
    with patch("forcedphot.odc.DataLoader.load_ephemeris_from_ecsv") as mock_load:
        odc_instance.args.ephem_ecsv = "dummy.ecsv"
        mock_load.return_value = ["mock_data"]
        result = odc_instance.run_ephemeris_query()
        assert result == QueryResult(target=None, start=None, end=None, ephemeris=['mock_data'])
        mock_load.assert_called_once_with("dummy.ecsv")

def test_run_ephemeris_query_csv(odc_instance):
    """Test CSV batch processing."""
    odc_instance.args.csv = "input.csv"
    odc_instance.ephemeris_client.query_from_csv.return_value = "mock_result"
    result = odc_instance.run_ephemeris_query()
    odc_instance.ephemeris_client.query_from_csv.assert_called_once_with(
        odc_instance.args.ephemeris_service, "input.csv", odc_instance.args.location, False
    )
    assert result == "mock_result"

# @patch('forcedphot.odc.Time')
# def test_run_ephemeris_query_single(mock_time, odc_instance):
#     """Test single target query."""
#     odc_instance.args.target_name = "2023 ABC"
#     odc_instance.args.target_type = "asteroid"
#     odc_instance.args.start_time = Time("2023-01-01")
#     odc_instance.args.end_time = Time("2023-01-02")
#     odc_instance.args.step = "1h"
#     odc_instance.ephemeris_client.query_single.return_value = "mock_single_result"
#     result = odc_instance.run_ephemeris_query()
#     odc_instance.ephemeris_client.query_single.assert_called_once_with(
#         "Horizons", "2023 ABC", "asteroid", odc_instance.args.start_time.iso,
#         odc_instance.args.end_time.iso, "1h", "X05", False
#     )
#     assert result == "mock_single_result"

@patch("forcedphot.odc.DataLoader.load_ephemeris_from_ecsv")
def test_run_image_query_with_ephem_ecsv(mock_load, odc_instance):
    """Test image query with ecsv"""
    odc_instance.args.ephem_ecsv = "dummy.ecsv"
    mock_ephem = MagicMock()
    mock_load.return_value = mock_ephem
    odc_instance.imphot_controller.search_images.return_value = ["img1", "img2"]
    result = odc_instance.run_image_query()
    assert result == ["img1", "img2"]
    mock_load.assert_called_once_with("dummy.ecsv")
    odc_instance.imphot_controller.configure_search.assert_called_once_with(
        bands=set(odc_instance.args.filters), ephemeris_data=odc_instance.ephemeris_results
    )

def test_run_photometry(odc_instance):
    """Test photometry processing with image results."""
    mock_images = [MagicMock(spec=ImageMetadata)]
    odc_instance.args.threshold = 5
    odc_instance.args.save_cutouts = True
    odc_instance.run_photometry(mock_images)
    odc_instance.imphot_controller.process_images.assert_called_once_with(
        target_name=odc_instance.args.target,
        target_type=odc_instance.args.target_type,
        image_type=odc_instance.args.image_type,
        ephemeris_service=odc_instance.args.ephemeris_service,
        image_metadata=mock_images,
        save_cutout=True,
        cutout_size=odc_instance.args.min_cutout_size,
        display=False,
    )

def test_api_connection_ephemeris(odc_instance):
    """Test API connection handling ephemeris input."""
    input_data = {
        "ephemeris": {
            "service": "Horizons",
            "target": "2023 XYZ",
            "target_type": "comet",
            "start": "2023-01-01",
            "end": "2023-01-02",
            "step": "2h"
        }
    }
    mock_result = MagicMock(spec=QueryResult)
    odc_instance.ephemeris_client.query_single.return_value = mock_result
    result = odc_instance.api_connection(input_data)
    assert "ephemeris" in result
    odc_instance.ephemeris_client.query_single.assert_called_once()

# def test_api_connection_image_photometry(odc_instance):
#     """Test API connection with image and photometry processing."""
#     input_data = {
#         "ephemeris": {"service": "Horizons",
#                       "target": "asteroid1",
#                       "target_type": "smallbody",
#                       "start": "2023-01-01",
#                       "end": "2023-01-02",
#                       "step": "2h"},
#         "image": {"filters": ["g", "r"]},
#         "photometry": {"image_type": "calexp", "threshold": 10}
#     }
#     odc_instance.ephemeris_client.query_single.return_value = MagicMock()
#     odc_instance.imphot_controller.search_images.return_value = [MagicMock()]
#     result = odc_instance.api_connection(input_data)
#     assert "image" in result and "photometry" in result
#     odc_instance.imphot_controller.configure_search.assert_called_with(
#         bands={"g", "r"}, ephemeris_data=odc_instance.ephemeris_results
#     )

def test_run_service_selection_all(odc_instance):
    """Test run method with service_selection='all'."""
    odc_instance.parse_args = MagicMock()
    odc_instance.args.service_selection = "all"
    odc_instance.run_ephemeris_query = MagicMock(return_value=MagicMock())
    odc_instance.run_image_query = MagicMock(return_value=["img"])
    odc_instance.run_photometry = MagicMock()
    odc_instance.run()
    odc_instance.run_ephemeris_query.assert_called_once()
    odc_instance.run_photometry.assert_called_once_with(["img"])
