import numpy as np
import pytest
from astropy.table import Table
from forcedphot.ephemeris.data_loader import DataLoader
from forcedphot.ephemeris.data_model import EphemerisData


@pytest.fixture
def sample_ecsv_file(tmp_path):
    """Create a sample ECSV file for testing."""
    file_path = tmp_path / "test_ephemeris.ecsv"
    data = Table(
        {
            "datetime": [2459000.5, 2459001.5],
            "RA_deg": [100.0, 101.0],
            "DEC_deg": [-20.0, -19.5],
            "RA_rate_arcsec_per_h": [0.1, 0.2],
            "DEC_rate_arcsec_per_h": [-0.1, -0.2],
            "AZ_deg": [180.0, 185.0],
            "EL_deg": [45.0, 46.0],
            "r_au": [1.0, 1.1],
            "delta_au": [0.5, 0.6],
            "V_mag": [15.0, 15.1],
            "alpha_deg": [30.0, 31.0],
            "RSS_3sigma_arcsec": [0.01, 0.02],
        }
    )
    data.write(file_path, format="ascii.ecsv")
    return file_path


def test_load_ephemeris_from_ecsv(sample_ecsv_file):
    """Test loading ephemeris data from a valid ECSV file."""
    ephemeris_data = DataLoader.load_ephemeris_from_ecsv(sample_ecsv_file)

    assert isinstance(ephemeris_data, EphemerisData)
    assert len(ephemeris_data.datetime) == 2
    assert np.allclose(ephemeris_data.RA_deg, [100.0, 101.0])
    assert np.allclose(ephemeris_data.DEC_deg, [-20.0, -19.5])
    assert np.allclose(ephemeris_data.RA_rate_arcsec_per_h, [0.1, 0.2])
    assert np.allclose(ephemeris_data.DEC_rate_arcsec_per_h, [-0.1, -0.2])
    assert np.allclose(ephemeris_data.AZ_deg, [180.0, 185.0])
    assert np.allclose(ephemeris_data.EL_deg, [45.0, 46.0])
    assert np.allclose(ephemeris_data.r_au, [1.0, 1.1])
    assert np.allclose(ephemeris_data.delta_au, [0.5, 0.6])
    assert np.allclose(ephemeris_data.V_mag, [15.0, 15.1])
    assert np.allclose(ephemeris_data.alpha_deg, [30.0, 31.0])
    assert np.allclose(ephemeris_data.RSS_3sigma_arcsec, [0.01, 0.02])


def test_load_ephemeris_from_nonexistent_file():
    """Test loading ephemeris data from a non-existent file."""
    with pytest.raises(FileNotFoundError):
        DataLoader.load_ephemeris_from_ecsv("nonexistent_file.ecsv")


def test_load_ephemeris_from_invalid_file(tmp_path):
    """Test loading ephemeris data from an invalid ECSV file (missing columns)."""
    invalid_file = tmp_path / "invalid_ephemeris.ecsv"
    data = Table({"datetime_jd": [2459000.5], "RA_deg": [100.0]})  # Missing columns
    data.write(invalid_file, format="ascii.ecsv")

    with pytest.raises(ValueError):
        DataLoader.load_ephemeris_from_ecsv(invalid_file)


def test_load_multiple_ephemeris_files(sample_ecsv_file, tmp_path):
    """Test loading multiple ephemeris files."""
    second_file = tmp_path / "test_ephemeris2.ecsv"
    data = Table(
        {
            "datetime": [2459002.5],
            "RA_deg": [102.0],
            "DEC_deg": [-19.0],
            "RA_rate_arcsec_per_h": [0.3],
            "DEC_rate_arcsec_per_h": [-0.3],
            "AZ_deg": [190.0],
            "EL_deg": [47.0],
            "r_au": [1.2],
            "delta_au": [0.7],
            "V_mag": [15.2],
            "alpha_deg": [32.0],
            "RSS_3sigma_arcsec": [0.03],
        }
    )
    data.write(second_file, format="ascii.ecsv")

    file_paths = [sample_ecsv_file, second_file]
    ephemeris_list = DataLoader.load_multiple_ephemeris_files(file_paths)

    assert len(ephemeris_list) == 2
    assert isinstance(ephemeris_list[0], EphemerisData)
    assert isinstance(ephemeris_list[1], EphemerisData)
    assert len(ephemeris_list[0].datetime) == 2
    assert len(ephemeris_list[1].datetime) == 1


def test_load_multiple_ephemeris_files_with_error(sample_ecsv_file):
    """Test loading multiple ephemeris files with one non-existent file."""
    file_paths = [sample_ecsv_file, "nonexistent_file.ecsv"]

    with pytest.raises(FileNotFoundError):
        DataLoader.load_multiple_ephemeris_files(file_paths)
