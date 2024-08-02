from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

from forcedphot.ephemeris import data_model, horizons_interface


@pytest.fixture
def mock_horizons():
    """
    Fixture to mock the Horizons class for testing.
    """
    with patch("forcedphot.ephemeris.horizons_interface.Horizons") as mock:
        yield mock


@pytest.fixture
def mock_csv_data():
    """
    Fixture to provide mock CSV data for testing.
    """
    return pd.DataFrame(
        {
            "target": ["Ceres"],
            "target_type": ["smallbody"],
            "start": ["2020-01-01"],
            "end": ["2020-01-02"],
            "step": ["1h"],
        }
    )


def test_init():
    """
    Test the initialization of HorizonsInterface with default and custom observer locations.
    """
    hi = horizons_interface.HorizonsInterface()
    assert hi.observer_location == horizons_interface.HorizonsInterface.DEFAULT_OBSERVER_LOCATION

    custom_location = "X06"
    hi_custom = horizons_interface.HorizonsInterface(observer_location=custom_location)
    assert hi_custom.observer_location == custom_location


def test_query_single_range_success(mock_horizons):
    """
    Test successful query of a single range using mocked Horizons data.
    """
    mock_ephemerides = MagicMock()
    mock_ephemerides.return_value = {
        "datetime_jd": [2459000.5],
        "RA": [100.0],
        "DEC": [-20.0],
        "RA_rate": [0.5],
        "DEC_rate": [-0.3],
        "AZ": [250.0],
        "EL": [45.0],
        "r": [1.5],
        "delta": [0.8],
        "V": [15.0],
        "alpha": [30.0],
        "RSS_3sigma": [0.1],
    }
    mock_horizons.return_value.ephemerides = mock_ephemerides

    hi = horizons_interface.HorizonsInterface()
    query = data_model.QueryInput("Ceres", "smallbody", Time("2020-01-01"), Time("2020-01-02"), "1h")
    result = hi.query_single_range(query)

    assert result is not None
    assert result.target == "Ceres"
    assert result.start == Time("2020-01-01")
    assert result.end == Time("2020-01-02")
    assert isinstance(result.ephemeris, data_model.EphemerisData)


def test_query_single_range_failure(mock_horizons):
    """
    Test failure handling when querying a single range with an invalid target.
    """
    mock_horizons.side_effect = Exception("Query failed")

    hi = horizons_interface.HorizonsInterface()
    query = data_model.QueryInput("Invalid Target", "smallbody", Time("2020-01-01"), Time("2020-01-02"), "1h")
    result = hi.query_single_range(query)

    assert result is None


@pytest.mark.parametrize(
    "target,target_type,start,end,step",
    [
        ("Ceres", "smallbody", "2020-01-01", "2020-01-02", "1h"),
        ("2021 XY", "smallbody", "2021-06-01", "2021-06-30", "2h"),
    ],
)
def test_query_input_creation(target, target_type, start, end, step):
    """
    Test creation of QueryInput objects with various parameters.
    """
    query = data_model.QueryInput(target, target_type, Time(start), Time(end), step)
    assert query.target == target
    assert query.target_type == target_type
    assert query.start == Time(start)
    assert query.end == Time(end)
    assert query.step == step


def test_ephemeris_data_creation():
    """
    Test creation of EphemerisData object and verify its attribute types.
    """
    ephemeris = data_model.EphemerisData()
    assert isinstance(ephemeris.datetime_jd, Time)
    assert isinstance(ephemeris.RA_deg, np.ndarray)
    assert isinstance(ephemeris.DEC_deg, np.ndarray)
    assert isinstance(ephemeris.RA_rate_arcsec_per_h, np.ndarray)
    assert isinstance(ephemeris.DEC_rate_arcsec_per_h, np.ndarray)
    assert isinstance(ephemeris.AZ_deg, np.ndarray)
    assert isinstance(ephemeris.EL_deg, np.ndarray)
    assert isinstance(ephemeris.r_au, np.ndarray)
    assert isinstance(ephemeris.delta_au, np.ndarray)
    assert isinstance(ephemeris.V_mag, np.ndarray)
    assert isinstance(ephemeris.alpha_deg, np.ndarray)
    assert isinstance(ephemeris.RSS_3sigma_arcsec, np.ndarray)


@patch("pandas.read_csv")
@patch("forcedphot.ephemeris.horizons_interface.HorizonsInterface.query_single_range")
@patch("astropy.table.Table.from_pandas")
@patch("astropy.table.Table.write")
def test_query_ephemeris_from_csv(
    mock_table_write, mock_table_from_pandas, mock_query_single_range, mock_read_csv, mock_csv_data
):
    """
    Test querying ephemeris data from a CSV file using mocked dependencies.
    """
    mock_read_csv.return_value = mock_csv_data

    mock_ephemeris = data_model.EphemerisData(
        datetime_jd=Time([2459000.5], format="jd"),
        RA_deg=np.array([100.0]),
        DEC_deg=np.array([-20.0]),
        RA_rate_arcsec_per_h=np.array([0.5]),
        DEC_rate_arcsec_per_h=np.array([-0.3]),
        AZ_deg=np.array([250.0]),
        EL_deg=np.array([45.0]),
        r_au=np.array([1.5]),
        delta_au=np.array([0.8]),
        V_mag=np.array([15.0]),
        alpha_deg=np.array([30.0]),
        RSS_3sigma_arcsec=np.array([0.1]),
    )
    mock_query_result = data_model.QueryResult(
        "Ceres", Time("2020-01-01"), Time("2020-01-02"), mock_ephemeris
    )
    mock_query_single_range.return_value = mock_query_result

    mock_table = MagicMock()
    mock_table_from_pandas.return_value = mock_table

    with patch("builtins.open", mock_open()) as _mock_file:
        horizons_interface.HorizonsInterface.query_ephemeris_from_csv("test.csv", save_data=True)

    mock_read_csv.assert_called_once_with("test.csv")
    mock_query_single_range.assert_called_once()
    mock_table_from_pandas.assert_called_once()

    expected_filename = "./Ceres_2020-01-01_00-00-00.000_2020-01-02_00-00-00.000.ecsv"
    expected_call = call(expected_filename, format="ascii.ecsv", overwrite=True)
    print(f"Expected call: {expected_call}")
    print(f"Actual calls: {mock_table.write.mock_calls}")

    assert expected_call in mock_table.write.mock_calls, "Expected write call not found"
