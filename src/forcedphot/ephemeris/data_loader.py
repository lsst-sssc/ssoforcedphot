import logging
from typing import list

import numpy as np
from astropy.table import Table
from astropy.time import Time

from forcedphot.ephemeris.local_dataclasses import EphemerisData


class DataLoader:
    """
    DataLoader is a class for loading ephemeris data from ECSV files and converting it into the
    EphemerisData class.

    This class provides methods to load ephemeris data from ECSV files and convert it to the
    EphemerisData class. It supports loading ephemeris data from a single ECSV file or from
    multiple ECSV files.

    Attributes:
        logger (logging.Logger): Logger for the class.

    Methods:
        load_ephemeris_from_ecsv(file_path: str) -> EphemerisData:
            Load ephemeris data from an ECSV file and return it as an EphemerisData object.
        load_multiple_ephemeris_files(file_paths: list[str]) -> list[EphemerisData]:
            Load ephemeris data from multiple ECSV files and return them as a list of EphemerisData objects.
    """

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    @staticmethod
    def load_ephemeris_from_ecsv(file_path: str) -> EphemerisData:
        """
        Load ephemeris data from an ECSV file and return it as an EphemerisData object.

        Parameters:
        -----------
        file_path : str
            Path to the ECSV file containing ephemeris data.

        Returns:
        --------
        EphemerisData
            An EphemerisData object populated with the data from the ECSV file.

        Raises:
        -------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the ECSV file is missing required columns.
        """
        try:
            # Read the ECSV file
            table = Table.read(file_path, format='ascii.ecsv')

            # Check if all required columns are present
            required_columns = [
                'datetime_jd', 'RA_deg', 'DEC_deg', 'RA_rate_arcsec_per_h',
                'DEC_rate_arcsec_per_h', 'AZ_deg', 'EL_deg', 'r_au', 'delta_au',
                'V_mag', 'alpha_deg', 'RSS_3sigma_arcsec'
            ]
            missing_columns = [col for col in required_columns if col not in table.colnames]
            if missing_columns:
                raise ValueError(f"Missing columns in ECSV file: {', '.join(missing_columns)}")

            # Create and populate the EphemerisData object
            ephemeris_data = EphemerisData(
                datetime_jd=Time(table['datetime_jd'], format='jd'),
                RA_deg=np.array(table['RA_deg']),
                DEC_deg=np.array(table['DEC_deg']),
                RA_rate_arcsec_per_h=np.array(table['RA_rate_arcsec_per_h']),
                DEC_rate_arcsec_per_h=np.array(table['DEC_rate_arcsec_per_h']),
                AZ_deg=np.array(table['AZ_deg']),
                EL_deg=np.array(table['EL_deg']),
                r_au=np.array(table['r_au']),
                delta_au=np.array(table['delta_au']),
                V_mag=np.array(table['V_mag']),
                alpha_deg=np.array(table['alpha_deg']),
                RSS_3sigma_arcsec=np.array(table['RSS_3sigma_arcsec'])
            )

            print(f"Loaded ephemeris data with {len(ephemeris_data.datetime_jd)} points from {file_path}.")

            return ephemeris_data

        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} was not found.") from None
        except Exception as e:
            raise ValueError(f"Error loading ECSV file: {str(e)}") from None

    @staticmethod
    def load_multiple_ephemeris_files(file_paths: list[str]) -> list[EphemerisData]:
        """
        Load multiple ephemeris files and return a list of EphemerisData objects.

        Parameters:
        -----------
        file_paths : List[str]
            A list of paths to ECSV files containing ephemeris data.

        Returns:
        --------
        List[EphemerisData]
            A list of EphemerisData objects, each populated with data from one ECSV file.
        """
        return [DataLoader.load_ephemeris_from_ecsv(file_path) for file_path in file_paths]


if __name__ == "__main__":
    # Example usage
    # file_path = "./Ceres_2024-01-01_00-00-00.000_2025-12-31_23-59-00.000.ecsv"
    # try:
    #     ephemeris_data = DataLoader.load_ephemeris_from_ecsv(file_path)
    # except Exception as e:
    #     print(f"Error: {str(e)}")

    # Example of loading multiple files
    file_paths = ["./Ceres_2024-01-01_00-00-00.000_2025-12-31_23-59-00.000.ecsv",
                  "./Encke_2024-01-01_00-00-00.000_2024-06-30_23-59-00.000.ecsv"]
    try:
        ephemeris_list = DataLoader.load_multiple_ephemeris_files(file_paths)
        print(f"Loaded {len(ephemeris_list)} ephemeris files.")
    except Exception as e:
        print(f"Error: {str(e)}")
