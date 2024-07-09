import pytest
from unittest.mock import patch, MagicMock
from astropy.time import Time
import pandas as pd
import numpy as np
from horizons_interface import HorizonsInterface, QueryInput, QueryResult, EphemerisData

@pytest.fixture
def mock_horizons():
    with patch('horizons_interface.Horizons') as mock:
        yield mock

@pytest.fixture
def mock_csv_data():
    return pd.DataFrame({
        'target': ['Ceres'],
        'start': ['2020-01-01'],
        'end': ['2020-01-02'],
        'step': ['1h']
    })

def test_init():
    hi = HorizonsInterface()
    assert hi.observer_location == HorizonsInterface.DEFAULT_OBSERVER_LOCATION

    custom_location = 'X06'
    hi_custom = HorizonsInterface(observer_location=custom_location)
    assert hi_custom.observer_location == custom_location

def test_query_single_range_success(mock_horizons):
    mock_ephemerides = MagicMock()
    mock_ephemerides.return_value = {
        'datetime_jd': [2459000.5],
        'RA': [100.0],
        'DEC': [-20.0],
        'RA_rate': [0.5],
        'DEC_rate': [-0.3],
        'AZ': [250.0],
        'EL': [45.0],
        'r': [1.5],
        'delta': [0.8],
        'V': [15.0],
        'alpha': [30.0],
        'RSS_3sigma': [0.1]
    }
    mock_horizons.return_value.ephemerides = mock_ephemerides

    hi = HorizonsInterface()
    query = QueryInput("Ceres", Time("2020-01-01"), Time("2020-01-02"), "1h")
    result = hi.query_single_range(query)

    assert result is not None
    assert result.target == "Ceres"
    assert result.start == Time("2020-01-01")
    assert result.end == Time("2020-01-02")
    assert isinstance(result.ephemeris, EphemerisData)

def test_query_single_range_failure(mock_horizons):
    mock_horizons.side_effect = Exception("Query failed")

    hi = HorizonsInterface()
    query = QueryInput("Invalid Target", Time("2020-01-01"), Time("2020-01-02"), "1h")
    result = hi.query_single_range(query)

    assert result is None

@pytest.mark.parametrize("target,start,end,step", [
    ("Ceres", "2020-01-01", "2020-01-02", "1h"),
    ("2021 XY", "2021-06-01", "2021-06-30", "2h"),
])
def test_query_input_creation(target, start, end, step):
    query = QueryInput(target, Time(start), Time(end), step)
    assert query.target == target
    assert query.start == Time(start)
    assert query.end == Time(end)
    assert query.step == step

def test_ephemeris_data_creation():
    ephemeris = EphemerisData()
    assert isinstance(ephemeris.datetime_jd, Time)
    assert isinstance(ephemeris.datetime_iso, Time)
    assert isinstance(ephemeris.RA_deg, np.ndarray)
    assert isinstance(ephemeris.DEC_deg, np.ndarray)
    # Add similar assertions for other attributes

@patch('pandas.read_csv')
@patch('horizons_interface.HorizonsInterface.query_single_range')
def test_query_ephemeris_from_csv(mock_query_single_range, mock_read_csv, mock_csv_data):
    mock_read_csv.return_value = mock_csv_data

    mock_ephemeris = EphemerisData(
        datetime_jd=Time([2459000.5], format='jd'),
        datetime_iso=Time(['2020-01-01 00:00:00.000'], format='iso'),
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
        RSS_3sigma_arcsec=np.array([0.1])
    )
    mock_query_result = QueryResult("Ceres", Time("2020-01-01"), Time("2020-01-02"), mock_ephemeris)
    mock_query_single_range.return_value = mock_query_result

    with patch('builtins.open', create=True), patch('pandas.DataFrame.to_csv') as mock_to_csv:
        HorizonsInterface.query_ephemeris_from_csv('test.csv')

        mock_read_csv.assert_called_once_with('test.csv')
        mock_query_single_range.assert_called_once()
        mock_to_csv.assert_called_once()