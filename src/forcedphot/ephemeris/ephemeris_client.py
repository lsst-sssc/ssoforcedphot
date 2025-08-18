import logging
from typing import Union

import pandas as pd
from astropy.time import Time
from ephemeris.data_loader import DataLoader
from ephemeris.data_model import EphemerisData, QueryInput
from ephemeris.horizons_interface import HorizonsInterface
from ephemeris.miriade_interface import MiriadeInterface


class EphemerisClient:
    """
    A client class to manage ephemeris queries using either JPL Horizons or Miriade services.

    This class provides a unified interface to query ephemeris data using either
    the HorizonsInterface or MiriadeInterface. It supports both single target queries
    and batch processing from CSV files.

    Attributes:
        DEFAULT_OBSERVER_LOCATION (str): Default location code for the observer (set to "X05"
        for Rubin Observatory).
        DEFAULT_save_ephemeris_data (bool): Default flag to save ephemeris data (set to False).
        logger (logging.Logger): Logger for the class.

    Methods:
        query_single(service: str, target: str, start: str, end: str, step: str, observer_location: str,
        save_ephem_data: bool):
            Query ephemeris for a single target using the specified service.

        query_from_csv(service: str, csv_file: str, observer_location: str, save_ephem_data: bool):
            Process multiple queries from a CSV file using the specified service.
    """

    DEFAULT_OBSERVER_LOCATION = "X05"
    DEFAULT_save_ephemeris_data = False

    def __init__(self):
        self.logger = logging.getLogger("ephemeris_client")
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
        save_ephem_data: bool = DEFAULT_save_ephemeris_data,
        output_folder: str = "./output",
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
            save_ephem_data (bool): Whether to save the query result as an ECSV file.
            output_folder (str): Directory path where output files will be saved.

        Returns:
            Union[QueryInput, None]: The query result if successful, None otherwise.
        """
        try:
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

            self.logger.info(f"Querying {service} for target: {target}")

            return interface.query_single_range(
                query=query, save_ephem_data=save_ephem_data, output_folder=output_folder
            )

        except Exception as e:
            self.logger.error(f"Error in query_single for target {target}: {str(e)}")
            return None

    def query_from_csv(
        self,
        service: str,
        csv_file: str,
        observer_location: str,
        save_ephem_data: bool = DEFAULT_save_ephemeris_data,
        output_folder: str = "./output",
    ):
        """
        Process multiple queries from a CSV file using the specified service.

        Args:
            service (str): The service to use ('horizons' or 'miriade').
            csv_file (str): Path to the CSV file containing query parameters.
            observer_location (str): The observer's location code.
            save_ephem_data (bool): Whether to save the query result as an ECSV file.
            output_folder (str): Directory path where output files will be saved.

        Returns:
            List of query results.
        """
        try:
            results = []

            self.logger.info(f"Processing CSV file: {csv_file}")

            df = pd.read_csv(csv_file)

            if df.empty:
                self.logger.warning(f"CSV file {csv_file} is empty")
                return results

            for index, row in df.iterrows():
                try:
                    if len(row) < 5:
                        self.logger.warning(f"Row {index} has insufficient columns, skipping")
                        continue

                    query = QueryInput(
                        target=str(row.iloc[0]),
                        target_type=str(row.iloc[1]),
                        start=Time(row.iloc[2], format="iso", scale="utc"),
                        end=Time(row.iloc[3], format="iso", scale="utc"),
                        step=str(row.iloc[4]),
                    )

                    self.logger.info(f"Processing row {index + 1}/{len(df)}: {query.target}")

                    query_result = self.query_single(
                        service,
                        query.target,
                        query.target_type,
                        query.start.iso,
                        query.end.iso,
                        query.step,
                        observer_location,
                        save_ephem_data=save_ephem_data,
                        output_folder=output_folder,
                    )

                    if query_result is not None:
                        results.append(query_result)
                        self.logger.info(f"Successfully processed {query.target}")
                    else:
                        self.logger.warning(f"Failed to process {query.target}")

                except Exception as row_error:
                    self.logger.error(f"Error processing row {index}: {str(row_error)}")
                    continue

            self.logger.info(f"Completed processing {len(results)} successful queries out of {len(df)} total")
            return results

        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {csv_file}")
            return []
        except pd.errors.EmptyDataError:
            self.logger.error(f"CSV file is empty: {csv_file}")
            return []
        except Exception as e:
            self.logger.error(f"An error occurred during query for CSV file {csv_file}")
            self.logger.error(f"Error details: {str(e)}")
            return []

    def load_ephemeris_from_ecsv(self, ecsv_file: str) -> EphemerisData:
        """
        Load ephemeris data from an ECSV file.

        Args:
            ecsv_file (str): Path to the ECSV file containing ephemeris data.

        Returns:
            EphemerisData: The ephemeris data as a dataclass.
        """
        try:
            self.logger.info(f"Loading ephemeris data from ECSV file: {ecsv_file}")
            return DataLoader.load_ephemeris_from_ecsv(ecsv_file)
        except Exception as e:
            self.logger.error(f"Error loading ECSV file {ecsv_file}: {str(e)}")
            raise

    def load_ephemeris_from_multi_ecsv(self, ecsv_files: list[str]) -> EphemerisData:
        """
        Load ephemeris data from multiple ECSV files.

        Args:
            ecsv_files (list[str]): List of paths to the ECSV files containing ephemeris data.

        Returns:
            List of EphemerisData: List of ephemeris data as a dataclass.
        """
        try:
            self.logger.info(f"Loading ephemeris data from {len(ecsv_files)} ECSV files")
            return DataLoader.load_multiple_ephemeris_files(ecsv_files)
        except Exception as e:
            self.logger.error(f"Error loading multiple ECSV files: {str(e)}")
            raise
