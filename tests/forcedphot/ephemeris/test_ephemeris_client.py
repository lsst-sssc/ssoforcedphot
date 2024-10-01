from unittest.mock import ANY, Mock, patch

import pytest
from astropy.time import Time
from forcedphot.ephemeris.data_model import EphemerisData, QueryResult
from forcedphot.ephemeris.ephemeris_client import EphemerisClient


@pytest.fixture
def ephemeris_client():
    """
    Fixture to provide an instance of EphemerisClient for use in tests.
    """
    return EphemerisClient()


@pytest.fixture
def mock_horizons_interface():
    """
    Fixture to mock the HorizonsInterface class for use in tests.
    """
    with patch("forcedphot.ephemeris.ephemeris_client.HorizonsInterface") as mock:
        yield mock


@pytest.fixture
def mock_miriade_interface():
    """
    Fixture to mock the MiriadeInterface class for use in tests.
    """
    with patch("forcedphot.ephemeris.ephemeris_client.MiriadeInterface") as mock:
        yield mock


def test_query_single_horizons(ephemeris_client, mock_horizons_interface):
    """
    Test the query_single method of EphemerisClient when using the JPL Horizons service.
    """
    mock_horizons_instance = Mock()
    mock_horizons_interface.return_value = mock_horizons_instance
    mock_horizons_instance.query_single_range.return_value = QueryResult(
        target="Ceres", start=Time("2023-01-01"), end=Time("2023-01-02"), ephemeris=EphemerisData()
    )

    result = ephemeris_client.query_single(
        "horizons", "Ceres", "smallbody", "2023-01-01", "2023-01-02", "1h", "X05"
    )

    assert isinstance(result, QueryResult)
    assert result.target == "Ceres"
    mock_horizons_interface.assert_called_once_with("X05")
    mock_horizons_instance.query_single_range.assert_called_once_with(ANY, save_data=False)


def test_query_single_miriade(ephemeris_client, mock_miriade_interface):
    """
    Test the query_single method of EphemerisClient when using the Miriade service.
    """
    mock_miriade_instance = Mock()
    mock_miriade_interface.return_value = mock_miriade_instance
    mock_miriade_instance.query_single_range.return_value = QueryResult(
        target="Encke", start=Time("2023-01-01"), end=Time("2023-01-02"), ephemeris=EphemerisData()
    )

    result = ephemeris_client.query_single(
        "miriade", "Encke", "comet_name", "2023-01-01", "2023-01-02", "1h", "X05"
    )

    assert isinstance(result, QueryResult)
    assert result.target == "Encke"
    mock_miriade_interface.assert_called_once_with("X05")
    mock_miriade_instance.query_single_range.assert_called_once_with(ANY, save_data=False)


def test_query_single_invalid_service(ephemeris_client):
    """
    Test the query_single method of EphemerisClient with an invalid service.
    """
    result = ephemeris_client.query_single(
        "invalid_service", "Ceres", "smallbody", "2023-01-01", "2023-01-02", "1h", "X05"
    )

    assert result is None


# @patch("forcedphot.ephemeris.horizons_interface.HorizonsInterface.query_ephemeris_from_csv")
# def test_query_from_csv_horizons(mock_query_csv, ephemeris_client):
#     """
#     Test the query_from_csv method of EphemerisClient when using the JPL Horizons service.
#     """
#     mock_query_csv.return_value = [
#         QueryResult(
#             target="Ceres", start=Time("2023-01-01"), end=Time("2023-01-02"), ephemeris=EphemerisData()
#         ),
#         QueryResult(
#             target="Vesta", start=Time("2023-01-01"), end=Time("2023-01-02"), ephemeris=EphemerisData()
#         ),
#     ]

#     results = ephemeris_client.query_from_csv("horizons", "test.csv", "X05")

#     assert len(results) == 2
#     assert all(isinstance(result, QueryResult) for result in results)
#     mock_query_csv.assert_called_once_with("test.csv", "X05", save_data=False)


# @patch("forcedphot.ephemeris.miriade_interface.MiriadeInterface.query_ephemeris_from_csv")
# def test_query_from_csv_miriade(mock_query_csv, ephemeris_client):
#     """
#     Test the query_from_csv method of EphemerisClient when using the Miriade service.
#     """
#     mock_query_csv.return_value = [
#         QueryResult(
#             target="Encke", start=Time("2023-01-01"), end=Time("2023-01-02"), ephemeris=EphemerisData()
#         ),
#         QueryResult(
#             target="Halley", start=Time("2023-01-01"), end=Time("2023-01-02"), ephemeris=EphemerisData()
#         ),
#     ]

#     results = ephemeris_client.query_from_csv("miriade", "test.csv", "X05")

#     assert len(results) == 2
#     assert all(isinstance(result, QueryResult) for result in results)
#     mock_query_csv.assert_called_once_with("test.csv", "X05", save_data=False)


# def test_query_from_csv_invalid_service(ephemeris_client):
#     """
#     Test the query_from_csv method of EphemerisClient with an invalid service.
#     """
#     result = ephemeris_client.query_from_csv("invalid_service", "test.csv", "X05")

#     assert result is None
