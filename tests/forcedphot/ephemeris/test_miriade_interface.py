from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from astropy.time import Time
from forcedphot.ephemeris.data_model import EphemerisData, QueryInput, QueryInputMiriade, QueryResult
from forcedphot.ephemeris.miriade_interface import MiriadeInterface


@pytest.fixture
def miriade_interface():
    """
    Fixture for creating a MiriadeInterface instance.
    """
    return MiriadeInterface()


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
    Test the initialization of MiriadeInterface with default and custom observer locations.
    """
    mi = MiriadeInterface()
    assert mi.observer_location == "X05"

    mi_custom = MiriadeInterface("500")
    assert mi_custom.observer_location == "500"


def test_calc_nsteps_for_miriade_query():
    """
    Test the calculation of the number of steps for a Miriade query.
    """
    query = QueryInput(
        target="Ceres", target_type="smallbody", start=Time("2023-01-01"), end=Time("2023-01-02"), step="1h"
    )
    mi = MiriadeInterface()
    result = mi.calc_nsteps_for_miriade_query(query)
    assert isinstance(result, QueryInputMiriade)
    assert result.nsteps == 24


def test_calc_nsteps_for_miriade_query_invalid_step():
    """
    Test the calculation of the number of steps for a Miriade query with an invalid step.
    """
    query = QueryInput(
        target="Ceres", target_type="smallbody", start=Time("2023-01-01"), end=Time("2023-01-02"), step="1x"
    )
    mi = MiriadeInterface()
    with pytest.raises(ValueError):
        mi.calc_nsteps_for_miriade_query(query)


@patch("forcedphot.ephemeris.miriade_interface.Miriade.get_ephemerides")
def test_query_single_range(mock_get_ephemerides, miriade_interface):
    """
    Test successful query of a single range using mocked Miriade data.
    """

    mock_get_ephemerides.return_value = Table(
        {
            "epoch": [2459580.5],
            "RAJ2000": [10.5],
            "DECJ2000": [20.5],
            "RAcosD_rate": [0.1],
            "DEC_rate": [0.2],
            "AZ": [30.0],
            "EL": [40.0],
            "heldist": [1.5],
            "delta": [1.0],
            "V": [5.0],
            "alpha": [60.0],
            "posunc": [0.01],
        }
    )

    query = QueryInput(
        target="Ceres", target_type="smallbody", start=Time("2023-01-01"), end=Time("2023-01-02"), step="1h"
    )
    result = miriade_interface.query_single_range(query)

    assert isinstance(result, QueryResult)
    assert result.target == "Ceres"
    assert result.start == Time("2023-01-01")
    assert result.end == Time("2023-01-02")

    assert isinstance(result.ephemeris, EphemerisData)
    assert len(result.ephemeris.datetime_jd) == 1
    assert result.ephemeris.RA_deg[0] == 10.5
    assert result.ephemeris.DEC_deg[0] == 20.5
    assert result.ephemeris.RA_rate_arcsec_per_h[0] == 0.1
    assert result.ephemeris.DEC_rate_arcsec_per_h[0] == 0.2
    assert result.ephemeris.AZ_deg[0] == 30.0
    assert result.ephemeris.EL_deg[0] == 40.0
    assert result.ephemeris.r_au[0] == 1.5
    assert result.ephemeris.delta_au[0] == 1.0
    assert result.ephemeris.V_mag[0] == 5.0
    assert result.ephemeris.alpha_deg[0] == 60.0
    assert result.ephemeris.RSS_3sigma_arcsec[0] == 0.01


def test_ephemeris_data_creation():
    """
    Test creation of EphemerisData object and verify its attribute types.
    """
    ephemeris = EphemerisData()
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
@patch("forcedphot.ephemeris.miriade_interface.MiriadeInterface.query_single_range")
@patch("astropy.table.Table.from_pandas")
@patch("astropy.table.Table.write")
def test_query_ephemeris_from_csv(
    mock_table_write, mock_table_from_pandas, mock_query_single_range, mock_read_csv, mock_csv_data
):
    """
    Test querying ephemeris data from a CSV file using mocked dependencies.
    """
    mock_read_csv.return_value = mock_csv_data

    mock_ephemeris = EphemerisData(
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
    mock_query_result = QueryResult("Ceres", Time("2020-01-01"), Time("2020-01-02"), mock_ephemeris)
    mock_query_single_range.return_value = mock_query_result

    mock_table = MagicMock()
    mock_table_from_pandas.return_value = mock_table

    mi = MiriadeInterface()
    with patch("builtins.open", mock_open()) as _mock_file:
        mi.query_ephemeris_from_csv("test.csv", save_data=True)

    mock_read_csv.assert_called_once_with("test.csv")
    mock_query_single_range.assert_called_once()
    mock_table_from_pandas.assert_called_once()

    expected_filename = "./Ceres_2020-01-01_00-00-00.000_2020-01-02_00-00-00.000.ecsv"
    expected_call = call(expected_filename, format="ascii.ecsv", overwrite=True)
    print(f"Expected call: {expected_call}")
    print(f"Actual calls: {mock_table.write.mock_calls}")

    assert expected_call in mock_table.write.mock_calls, "Expected write call not found"
