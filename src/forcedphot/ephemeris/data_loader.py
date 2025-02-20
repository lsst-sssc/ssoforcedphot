import logging

import numpy as np
from astropy.table import Table
from astropy.time import Time

from forcedphot.ephemeris.data_model import EphemerisData


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
    logger = logging.getLogger("data_loader")

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
            table = Table.read(file_path, format="ascii.ecsv")

            # Check if all required columns are present
            required_columns = [
                "datetime",
                "RA_deg",
                "DEC_deg",
                "RA_rate_arcsec_per_h",
                "DEC_rate_arcsec_per_h",
                "AZ_deg",
                "EL_deg",
                "r_au",
                "delta_au",
                "V_mag",
                "alpha_deg",
                "RSS_3sigma_arcsec",
                "SMAA_3sigma_arcsec",
                "SMIA_3sigma_arcsec",
                "Theta_3sigma_deg",
            ]
            missing_columns = [col for col in required_columns if col not in table.colnames]
            if missing_columns:
                raise ValueError(f"Missing columns in ECSV file: {', '.join(missing_columns)}")

            # Create and populate the EphemerisData object
            ephemeris_data = EphemerisData(
                datetime=Time(table["datetime"], scale="utc", format="jd"),
                RA_deg=np.array(table["RA_deg"]),
                DEC_deg=np.array(table["DEC_deg"]),
                RA_rate_arcsec_per_h=np.array(table["RA_rate_arcsec_per_h"]),
                DEC_rate_arcsec_per_h=np.array(table["DEC_rate_arcsec_per_h"]),
                AZ_deg=np.array(table["AZ_deg"]),
                EL_deg=np.array(table["EL_deg"]),
                r_au=np.array(table["r_au"]),
                delta_au=np.array(table["delta_au"]),
                V_mag=np.array(table["V_mag"]),
                alpha_deg=np.array(table["alpha_deg"]),
                RSS_3sigma_arcsec=np.array(table["RSS_3sigma_arcsec"]),
                SMAA_3sigma_arcsec= np.array(table["SMAA_3sigma_arcsec"]),
                SMIA_3sigma_arcsec= np.array(table["SMIA_3sigma_arcsec"]),
                Theta_3sigma_deg= np.array(table["Theta_3sigma_deg"]),
            )

            DataLoader.logger.info(
                f"Loaded ephemeris data with {len(ephemeris_data.datetime)} points from {file_path}."
            )

            return ephemeris_data

        except FileNotFoundError:
            DataLoader.logger.error(f"The file {file_path} was not found.")
            raise
        except ValueError as ve:
            DataLoader.logger.error(f"Value error in file {file_path}: {str(ve)}")
            raise
        except Exception as e:
            DataLoader.logger.error(f"Unexpected error loading ECSV file {file_path}: {str(e)}")
            raise ValueError(f"Error loading ECSV file: {str(e)}") from e

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
        ephemeris_list = []
        for file_path in file_paths:
            try:
                ephemeris_data = DataLoader.load_ephemeris_from_ecsv(file_path)
                ephemeris_list.append(ephemeris_data)
            except (FileNotFoundError, ValueError) as e:
                DataLoader.logger.error(f"Error loading file {file_path}: {str(e)}")
                raise  # Re-raise the exception to be caught by the calling function

        return ephemeris_list
