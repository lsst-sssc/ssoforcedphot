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


# def main():
#     """
#     Main function to handle command-line arguments and execute ephemeris queries.

#     This function parses command-line arguments to determine whether to perform a single
#     query or batch processing from a CSV file. It supports querying ephemeris data using
#     either the JPL Horizons or Miriade services.

#     Command-line Arguments:
#         --service (str): The service to use for querying ('horizons' or 'miriade') deafult is 'horizons'.
#         --csv (str): Path to the CSV file for batch processing (optional).
#         --ecsv (str): Path to the ECSV file for single query (optional) or
#         list of ECSV files for batch processing (optional).
#         --target (str): Target object for a single query (optional).
#         --target_type (str): Target object type for a single query (optional).
#         --start (str): Start time for a single query (optional).
#         --end (str): End time for a single query (optional).
#         --step (str): Time step for a single query (optional).
#         --location (str): Observer location code (default is 'X05').
#         --save_data (bool): Flag to save query results as ECSV files (default is False).

#     Behavior:
#         - If the --csv argument is provided, the function will process multiple queries from the specified
#         CSV file.
#         - If all single query parameters (--target, --target_type, --start, --end, --step) are provided,
#         the function will perform a single query.
#         - If neither a CSV file nor all single query parameters are provided, the function will display
#         an error message.

#     Example Usage:
#         python ephemeris_client.py --service horizons --csv queries.csv --save_data
#         python ephemeris_client.py --service miriade --target Ceres --target_type smallbody
#         --start 2023-01-01 --end 2023-01-02 --step 1h
#         python ephemeris_client.py --ecsv ceres_ephemeris.ecsv,vesta_ephemeris.ecsv

#     Returns:
#         result (list[EphemerisData]): List of ephemeris data as a dataclass.
#     """
#     parser = argparse.ArgumentParser(
#         description="Query ephemeris data using Horizons or Miriade services or"
#         " load ephemeris data from existing ECSV."
#     )
#     parser.add_argument(
#         "--service", choices=["horizons", "miriade"], default="horizons", help="Service to use for querying"
#     )
#     parser.add_argument(
#         "--ecsv", help="Path to ECSV file (or a list separated with ,) containing ephemeris data"
#     )
#     parser.add_argument("--csv", help="Path to CSV file for batch processing")
#     parser.add_argument("--target", help="Target object for single query")
#     parser.add_argument("--target_type", help="Target object type for single query")
#     parser.add_argument("--start", help="Start time for single query")
#     parser.add_argument("--end", help="End time for single query")
#     parser.add_argument("--step", help="Time step for single query")
#     parser.add_argument(
#         "--location",
#         default=EphemerisClient.DEFAULT_OBSERVER_LOCATION,
#         help="Observer location code, default: Rubin(X05)",
#     )
#     parser.add_argument("--save_data", action="store_true", help="Save query results as ECSV files")

#     args = parser.parse_args()

#     client = EphemerisClient()

#     if args.csv:
#         results = client.query_from_csv(args.service, args.csv, args.location, args.save_data)
#     elif all([args.target, args.target_type, args.start, args.end, args.step]):
#         result = client.query_single(
#             args.service,
#             args.target,
#             args.target_type,
#             args.start,
#             args.end,
#             args.step,
#             args.location,
#             args.save_data,
#         )
#         results = [result] if result else []
#     elif args.ecsv:
#         ecsv_files = args.ecsv.split(",")  # Assume multiple files are comma-separated
#         if len(ecsv_files) > 1:
#             results = client.load_ephemeris_from_multi_ecsv(ecsv_files)
#         else:
#             results = client.load_ephemeris_from_ecsv(args.ecsv)
#     else:
#         parser.error(
#             "Either provide a CSV file or all single query parameters"
#             " like target, target_type,start, end, step"
#             " or ECSV file containing ephemeris data"
#         )

#     if results:
#         print(f"Successfully queried {len(results)} object(s)")
#         return results
#     else:
#         print("No results obtained")


# if __name__ == "__main__":
#     main()
