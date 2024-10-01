import logging
from typing import Union

import pandas as pd
from astropy.time import Time

from forcedphot.ephemeris.data_loader import DataLoader
from forcedphot.ephemeris.data_model import EphemerisData, QueryInput
from forcedphot.ephemeris.horizons_interface import HorizonsInterface
from forcedphot.ephemeris.miriade_interface import MiriadeInterface


class EphemerisClient:
    """
    A client class to manage ephemeris queries using either JPL Horizons or Miriade services.

    This class provides a unified interface to query ephemeris data using either
    the HorizonsInterface or MiriadeInterface. It supports both single target queries
    and batch processing from CSV files.

    Attributes:
        DEFAULT_OBSERVER_LOCATION (str): Default location code for the observer (set to "X05"
        for Rubin Observatory).
        DEFAULT_SAVE_DATA (bool): Default flag to save data (set to False).
        logger (logging.Logger): Logger for the class.

    Methods:
        query_single(service: str, target: str, start: str, end: str, step: str, observer_location: str,
        save_data: bool):
            Query ephemeris for a single target using the specified service.

        query_from_csv(service: str, csv_file: str, observer_location: str, save_data: bool):
            Process multiple queries from a CSV file using the specified service.
    """

    DEFAULT_OBSERVER_LOCATION = "X05"
    DEFAUT_SAVE_DATA = False

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def query_single(
        self,
        service: str,
        target: str,
        target_type: str,
        start: str,
        end: str,
        step: str,
        observer_location: str,
        save_data: bool = DEFAUT_SAVE_DATA,
    ) -> Union[QueryInput, None]:
        """
        Query ephemeris for a single target using the specified service.

        Args:
            service (str): The service to use ('horizons' or 'miriade').
            target (str): The target object's name or designation.
            target_type (str): The target object's type.
            start (str): The start time for the ephemeris query.
            end (str): The end time for the ephemeris query.
            step (str): The time step for the ephemeris query.
            observer_location (str): The observer's location code.
            save_data (bool): Whether to save the query result as an ECSV file.

        Returns:
            Union[QueryInput, None]: The query result if successful, None otherwise.
        """
        query = QueryInput(
            target=target,
            target_type=target_type,
            start=Time(start, format="iso", scale="utc"),
            end=Time(end, format="iso", scale="utc"),
            step=step,
        )

        if service.lower() == "horizons":
            interface = HorizonsInterface(observer_location)
        elif service.lower() == "miriade":
            interface = MiriadeInterface(observer_location)
        else:
            self.logger.error(f"Invalid service: {service}. Use 'horizons' or 'miriade'.")
            return None

        return interface.query_single_range(query, save_data=save_data)

    def query_from_csv(
        self, service: str, csv_file: str, observer_location: str, save_data: bool = DEFAUT_SAVE_DATA
    ):
        """
        Process multiple queries from a CSV file using the specified service.

        Args:
            service (str): The service to use ('horizons' or 'miriade').
            csv_file (str): Path to the CSV file containing query parameters.
            observer_location (str): The observer's location code.
            save_data (bool): Whether to save the query result as an ECSV file.

        Returns:
            List of query results.
        """
        try:
            results = []
            df = pd.read_csv(csv_file)

            for _index, row in df.iterrows():
                query = QueryInput(
                    target=row.iloc[0],
                    target_type=row.iloc[1],
                    start=Time(row.iloc[2], format="iso", scale="utc"),
                    end=Time(row.iloc[3], format="iso", scale="utc"),
                    step=row.iloc[4],
                )

                query_result = self.query_single(
                    service,
                    query.target,
                    query.target_type,
                    query.start,
                    query.end,
                    query.step,
                    observer_location,
                    save_data=save_data,
                )

                if query_result is not None:
                    results.append(query_result)

            return results

        except Exception as e:
            self.logger.error(f"An error occured during query for CSV file {csv_file}")
            self.logger.error(f"Error details: {str(e)}")

    def load_ephemeris_from_ecsv(self, ecsv_file: str) -> EphemerisData:
        """
        Load ephemeris data from an ECSV file.

        Args:
            ecsv_file (str): Path to the ECSV file containing ephemeris data.

        Returns:
            EphemerisData: The ephemeris data as a dataclass.
        """
        return DataLoader.load_ephemeris_from_ecsv(ecsv_file)

    def load_ephemeris_from_multi_ecsv(self, ecsv_files: list[str]) -> EphemerisData:
        """
        Load ephemeris data from multiple ECSV files.

        Args:
            ecsv_files (list[str]): List of paths to the ECSV files containing ephemeris data.

        Returns:
            List of EphemerisData: List of ephemeris data as a dataclass.
        """
        return DataLoader.load_multiple_ephemeris_files(ecsv_files)
