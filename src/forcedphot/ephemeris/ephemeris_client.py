import argparse
import logging
from typing import Union

from astropy.time import Time

from forcedphot.ephemeris.horizons_interface import HorizonsInterface
from forcedphot.ephemeris.local_dataclasses import QueryInput
from forcedphot.ephemeris.miriade_interface import MiriadeInterface


class EphemerisClient:
    """
    A client class to manage ephemeris queries using either JPL Horizons or Miriade services.

    This class provides a unified interface to query ephemeris data using either
    the HorizonsInterface or MiriadeInterface. It supports both single target queries
    and batch processing from CSV files.

    Attributes:
        DEFAULT_OBSERVER_LOCATION (str): Default location code for the observer (set to "X05" for Rubin Observatory).
        logger (logging.Logger): Logger for the class.

    Methods:
        query_single(service: str, target: str, start: str, end: str, step: str, observer_location: str):
            Query ephemeris for a single target using the specified service.

        query_from_csv(service: str, csv_file: str, observer_location: str):
            Process multiple queries from a CSV file using the specified service.
    """

    DEFAULT_OBSERVER_LOCATION = "X05"
    DEFAUT_SAVE_DATA = False

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def query_single(self, service: str, target: str, target_type: str, start: str, end: str, step: str,
                     observer_location: str, save_data: bool = DEFAUT_SAVE_DATA) -> Union[QueryInput, None]:
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

        Returns:
            Union[QueryInput, None]: The query result if successful, None otherwise.
        """
        query = QueryInput(
            target=target,
            target_type=target_type,
            start=Time(start, scale="utc"),
            end=Time(end, scale="utc"),
            step=step,
        )

        if service.lower() == 'horizons':
            interface = HorizonsInterface(observer_location, save_data=save_data)
        elif service.lower() == 'miriade':
            interface = MiriadeInterface(observer_location, save_data=save_data)
        else:
            self.logger.error(f"Invalid service: {service}. Use 'horizons' or 'miriade'.")
            return None

        return interface.query_single_range(query)

    def query_from_csv(self, service: str, csv_file: str, observer_location: str, save_data: bool = DEFAUT_SAVE_DATA):
        """
        Process multiple queries from a CSV file using the specified service.

        Args:
            service (str): The service to use ('horizons' or 'miriade').
            csv_file (str): Path to the CSV file containing query parameters.
            observer_location (str): The observer's location code.

        Returns:
            List of query results.
        """
        if service.lower() == 'horizons':
            return HorizonsInterface.query_ephemeris_from_csv(csv_file, observer_location, save_data=save_data)
        elif service.lower() == 'miriade':
            return MiriadeInterface.query_ephemeris_from_csv(csv_file, observer_location, save_data=save_data)
        else:
            self.logger.error(f"Invalid service: {service}. Use 'horizons' or 'miriade'.")
            return None


def main():
    parser = argparse.ArgumentParser(description="Query ephemeris data using JPL Horizons or Miriade services.")
    parser.add_argument('service', choices=['horizons', 'miriade'], help="Service to use for querying")
    parser.add_argument('--csv', help="Path to CSV file for batch processing")
    parser.add_argument('--target', help="Target object for single query")
    parser.add_argument('--target_type', help="Target object type for single query")
    parser.add_argument('--start', help="Start time for single query")
    parser.add_argument('--end', help="End time for single query")
    parser.add_argument('--step', help="Time step for single query")
    parser.add_argument('--location', default=EphemerisClient.DEFAULT_OBSERVER_LOCATION,
                        help="Observer location code")

    args = parser.parse_args()

    client = EphemerisClient()

    if args.csv:
        results = client.query_from_csv(args.service, args.csv, args.location)
    elif all([args.target, args.target_type, args.start, args.end, args.step]):
        result = client.query_single(args.service, args.target, args.target_type, args.start, args.end,
            args.step, args.location)
        results = [result] if result else []
    else:
        parser.error("Either provide a CSV file or all single query parameters"
                     "like target, target_type,start, end, step")

    if results:
        print(f"Successfully queried {len(results)} object(s)")
    else:
        print("No results obtained")


if __name__ == "__main__":
    main()
