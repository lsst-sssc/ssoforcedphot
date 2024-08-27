from unittest.mock import MagicMock, patch

import pytest
from astropy.time import Time
from forcedphot.odc import ObjectDetectionController


@pytest.fixture
def controller():
    """Fixture for ObjectDetectionController."""
    return ObjectDetectionController()


def test_create_parser(controller):
    """Test if the parser is created with correct description and prefix chars."""
    parser = controller.create_parser()
    assert parser.description == "Object Detection Controller"
    assert parser.fromfile_prefix_chars == "@"


def test_parse_args_default(controller):
    """Test if default arguments are parsed correctly."""
    args = controller.parse_args([])
    assert args.ephemeris_service == "Horizons"
    assert args.service_selection == "ephemeris"
    assert args.location == "X05"
    assert args.threshold == 3


def test_parse_args_custom(controller):
    """Test if custom arguments are parsed correctly."""
    args = controller.parse_args([
        "--ephemeris-service", "Miriade",
        "--target", "Ceres",
        "--target-type", "smallbody",
        "--start-time", "2023-01-01 00:00:00",
        "--day-range", "30",
        "--step", "2h"
    ])
    assert args.ephemeris_service == "Miriade"
    assert args.target == "Ceres"
    assert args.target_type == "smallbody"
    assert isinstance(args.start_time, Time)
    assert isinstance(args.end_time, Time)
    assert args.step == "2h"


def test_parse_args_time_handling(controller):
    """Test if time-related arguments are parsed and converted correctly."""
    args = controller.parse_args(["--start-time", "2023-01-01 00:00:00", "--day-range", "30"])
    assert args.start_time == Time("2023-01-01 00:00:00", scale="utc")
    assert args.end_time == Time("2023-01-31 00:00:00", scale="utc")


@patch("forcedphot.odc.EphemerisClient")
def test_run_ephemeris_query_ecsv(mock_client, controller):
    """Test ephemeris query with ECSV file input."""
    with patch("forcedphot.odc.DataLoader.load_ephemeris_from_ecsv") as mock_load:
        mock_load.return_value = ["mock_data"]
        controller.args = MagicMock(ecsv="test.ecsv")
        result = controller.run_ephemeris_query()
        assert result == ["mock_data"]
        mock_load.assert_called_once_with("test.ecsv")


@patch("forcedphot.odc.EphemerisClient")
def test_run_ephemeris_query_csv(mock_client, controller):
    """Test ephemeris query with CSV file input."""
    mock_client_instance = mock_client.return_value
    mock_client_instance.query_from_csv.return_value = ["mock_data"]
    controller.args = MagicMock(
        ecsv=None, csv="test.csv", ephemeris_service="Horizons", location="X05", save_data=True
        )
    result = controller.run_ephemeris_query()
    assert result == ["mock_data"]
    mock_client_instance.query_from_csv.assert_called_once_with("Horizons", "test.csv", "X05", True)


@patch("forcedphot.odc.EphemerisClient")
def test_run_ephemeris_query_single(mock_client, controller):
    """Test single ephemeris query with directly provided parameters."""
    mock_client_instance = mock_client.return_value
    mock_client_instance.query_single.return_value = "mock_data"
    controller.args = MagicMock(
        ecsv=None,
        csv=None,
        ephemeris_service="Horizons",
        target="Ceres",
        target_type="smallbody",
        start_time=Time("2023-01-01 00:00:00", scale="utc"),
        end_time=Time("2023-01-31 00:00:00", scale="utc"),
        step="1h",
        location="X05",
        save_data=True,
    )
    result = controller.run_ephemeris_query()
    assert result == "mock_data"
    mock_client_instance.query_single.assert_called_once_with(
        "Horizons", "Ceres", "smallbody", "2023-01-01 00:00:00.000",
        "2023-01-31 00:00:00.000", "1h", "X05", True
    )

def test_run_ephemeris_query_missing_args(controller):
    """Test if system exits when required arguments are missing."""
    controller.args = MagicMock(ecsv=None, csv=None, target=None)
    with pytest.raises(SystemExit):
        controller.run_ephemeris_query()


@patch.object(ObjectDetectionController, "parse_args")
@patch.object(ObjectDetectionController, "run_ephemeris_query")
def test_run(mock_run_query, mock_parse_args, controller):
    """Test the main run method of the controller."""
    mock_run_query.return_value = ["result1", "result2"]
    result = controller.run()
    assert result == ["result1", "result2"]
    mock_parse_args.assert_called_once()
    mock_run_query.assert_called_once()


@patch.object(ObjectDetectionController, "parse_args")
@patch.object(ObjectDetectionController, "run_ephemeris_query")
def test_run_no_results(mock_run_query, mock_parse_args, controller):
    """Test the run method when no results are returned from the query."""
    mock_run_query.return_value = None
    result = controller.run()
    assert result is None
    mock_parse_args.assert_called_once()
    mock_run_query.assert_called_once()
