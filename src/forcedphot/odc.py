import argparse
from typing import Any

import astropy.units as u
from astropy.time import Time

from forcedphot.ephemeris.data_loader import DataLoader
from forcedphot.ephemeris.ephemeris_client import EphemerisClient


class ObjectDetectionController:
    """
    This class handles argument parsing, ephemeris queries, and overall control
    flow for object detection tasks for forced photometry. It provides a wide
    range of options for configuring the detection process, including ephemeris
    service selection, target specification, time range settings, input file
    handling, and various service configurations.

    Command-line Arguments:
        --service-selection {all, ephemeris, catalog, image, photometry}:
            Selects the service to use.

        -e, --ephemeris-service {Horizons,Miriade}:
            Ephemeris service to use. Default is "Horizons".

        -t, --target:
            Target object for single query.

        --target-type:
            Target object type for single query.

        -s, --start-time:
            Start time for the search (format: YYYY-MM-DD HH:MM:SS).

        -e, --end-time:
            End time for the search (format: YYYY-MM-DD HH:MM:SS).

        -d, --day-range:
            Number of days to search forward from start time.

        --step:
            Time step for ephemeris query. Default is "1h".

        --ecsv:
            Path to ECSV file containing input ephemeris data. (see template_ephemeris.ecsv
            in tests/forcedphot/data)

        --csv:
            Path to CSV file for batch processing.

        --location:
            Observer location code. Default is "X05" for Rubin Observatory.

        --save-data:
            Flag to save query results (Ephemeris data).

        --image-service {Rubin,ZTF}:
            Image service to use. Default is "Rubin".

        --return-cutouts:
            Flag to return cutouts from image service. Default is False.

        -m, --min-cutout-size:
            Minimum size of cutouts to return (max is error ellipse size).

        -f, --filter:
            Filter for image service. Default is "r".

        --catalog-service {Rubin,ZTF}:
            Catalog service to use. Default is "Rubin".

        -x, --max-search-ellipse:
            Maximum size of error ellipse to search (3-sigma value).

        --photometry-service {choice1,choice2}:
            Photometry service to use. Default is "choice1".

        --threshold:
            Threshold SNR for forced photometry. Default is 3.

    Example usage:
        python odc.py --ephemeris-service Horizons --target Ceres --target-type smallbody
        --start-time "2023-01-01 00:00:00" --day-range 30 --step 1h --save-data

        python odc.py --csv targets.csv --save-data

        python odc.py @query_args.txt
            query_args.txt shoud contain the arguments for the query 1 per line (see query_args.txt)
    """

    def __init__(self):
        """
        Initialize the ObjectDetectionController.

        Sets up the argument parser and initializes the args attribute.
        """

        self.parser = self.create_parser()
        self.args = None

    def create_parser(self):
        """
        Create and configure the argument parser.

        Returns:
            argparse.ArgumentParser: Configured argument parser object.
        """

        parser = argparse.ArgumentParser(description="Object Detection Controller", fromfile_prefix_chars="@")

        # Selection of services
        parser.add_argument(
            "--service-selection",
            choices=["all", "ephemeris", "catalog", "image", "photometry"],
            default="ephemeris",
            help="Select which services to use (default: ephemeris)",
        )

        # Ephemeris Service selection
        parser.add_argument(
            "-es",
            "--ephemeris-service",
            choices=["Horizons", "Miriade"],
            default="Horizons",
            help="Ephemeris service to use",
        )

        # Target options
        parser.add_argument("-t", "--target", help="Target object for single query")
        parser.add_argument("--target-type", help="Target object type for single query")

        # Time range options
        parser.add_argument(
            "-s", "--start-time", help="Start time for the search (format: YYYY-MM-DD HH:MM:SS)"
        )
        parser.add_argument("-e", "--end-time", help="End time for the search (format: YYYY-MM-DD HH:MM:SS)")
        parser.add_argument(
            "-d", "--day-range", type=int, help="Number of days to search forward from start time"
        )

        parser.add_argument("--step", default="1h", help="Time step for ephemeris query (default: 1h)")

        # Input options
        parser.add_argument("--ecsv", help="Path to ECSV file containing input data")
        parser.add_argument("--csv", help="Path to CSV file for batch processing")

        # Other options
        parser.add_argument(
            "--location", default="X05", help="Observer location code (default: X05 for Rubin Observatory)"
        )
        parser.add_argument("--save-data", action="store_true", help="Save query results (Ephemeris data)")

        # Image service
        parser.add_argument(
            "--image-service",
            choices=["LSST-Butler", "ZTF"],
            default="LSST-Butler",
            help="Image service to use",
        )

        parser.add_argument(
            "--return-cutouts",
            action="store_false",
            help="Return cutouts from image service (default is False)",
        )

        parser.add_argument(
            "-m",
            "--min-cutout-size",
            type=int,
            help="Minimum size of cutouts to return (max is error ellipse size)",
        )

        parser.add_argument(
            "--f",
            "--filters",
            choices=["u", "g", "r", "i", "z", "y"],
            default="r",
            help="Comma-separated list of filters to use for image service",
        )

        # Catalog service
        parser.add_argument(
            "--catalog-service",
            choices=["LSST-TAP", "ZTF"],
            default="LSST-TAP",
            help="Catalog service to use",
        )

        parser.add_argument(
            "-x",
            "--max-search-ellipse",
            type=float,
            help="Maximum size of error ellipse to search (3-sigma value)",
        )

        # Phometry service
        parser.add_argument(
            "--photometry-service",
            choices=["choice1", "choice2"],
            default="choice1",
            help="Photometry service to use",
        )

        parser.add_argument(
            "--threshold", type=int, default=3, help="Threshold SNR for forced photometry (defaullt: 3)"
        )

        return parser

    def parse_args(self, args=None):
        """
        Parse command-line arguments and process time-related arguments.

        Args:
            args (list, optional): List of command-line arguments. Defaults to None.

        Returns:
            argparse.Namespace: Parsed argument object.
        """
        self.args = self.parser.parse_args(args)

        # Handle start and end times
        if self.args.start_time:
            self.args.start_time = Time(self.args.start_time, scale="utc")
        if self.args.end_time:
            self.args.end_time = Time(self.args.end_time, scale="utc")
        elif self.args.day_range:
            self.args.end_time = self.args.start_time + (self.args.day_range * u.day)

        return self.args

    def run_ephemeris_query(self):
        """
        Execute the ephemeris query based on the provided arguments.

        Returns:
            object: Results of the ephemeris query.

        Raises:
            argparse.ArgumentError: If required arguments for the query are missing.
        """

        client = EphemerisClient()

        if self.args.ecsv:
            return DataLoader.load_ephemeris_from_ecsv(self.args.ecsv)
        elif self.args.csv:
            return client.query_from_csv(
                self.args.ephemeris_service, self.args.csv, self.args.location, self.args.save_data
            )
        elif all([self.args.target, self.args.target_type, self.args.start_time, self.args.end_time]):
            return client.query_single(
                self.args.ephemeris_service,
                self.args.target,
                self.args.target_type,
                self.args.start_time.iso,
                self.args.end_time.iso,
                self.args.step,
                self.args.location,
                self.args.save_data,
            )
        else:
            self.parser.error("Either provide a CSV/ECSV file or all single query parameters")

    def api_connection(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Handle API connections and execute the object detection process based on input data.

        This method accepts a dictionary of input parameters and uses the EphemerisClient
        to perform ephemeris queries.

        Args:
            input_data (dict[str, Any]): A dictionary containing input parameters for the object
            detection process.

        Returns:
            dict[str, Any]: A dictionary containing the results of the object detection process.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        try:
            results = {}

            if "ephemeris" in input_data:
                ephemeris_data = input_data["ephemeris"]

                if "ecsv_file" in ephemeris_data:
                    results["ephemeris"] = self.ephemeris_client.load_ephemeris_from_ecsv(
                        ephemeris_data["ecsv_file"]
                    )

                elif "ecsv_files" in ephemeris_data:
                    results["ephemeris"] = self.ephemeris_client.load_ephemeris_from_multi_ecsv(
                        ephemeris_data["ecsv_files"]
                    )

                elif "service" in ephemeris_data:
                    if "csv_file" in ephemeris_data:
                        results["ephemeris"] = self.ephemeris_client.query_from_csv(
                            service=ephemeris_data["service"],
                            csv_file=ephemeris_data["csv_file"],
                            observer_location=ephemeris_data.get(
                                "observer_location", EphemerisClient.DEFAULT_OBSERVER_LOCATION
                            ),
                            save_data=ephemeris_data.get("save_data", EphemerisClient.DEFAUT_SAVE_DATA),
                        )
                    else:
                        results["ephemeris"] = self.ephemeris_client.query_single(
                            service=ephemeris_data["service"],
                            target=ephemeris_data["target"],
                            target_type=ephemeris_data["target_type"],
                            start=ephemeris_data["start"],
                            end=ephemeris_data["end"],
                            step=ephemeris_data["step"],
                            observer_location=ephemeris_data.get(
                                "observer_location", EphemerisClient.DEFAULT_OBSERVER_LOCATION
                            ),
                            save_data=ephemeris_data.get("save_data", EphemerisClient.DEFAUT_SAVE_DATA),
                        )
                else:
                    raise ValueError("Invalid ephemeris query parameters")

            # Placeholder for other services (to be implemented)
            if "catalog" in input_data:
                # results["catalog"] = self.run_catalog_query(input_data["catalog"])
                pass

            if "image" in input_data:
                # results["image"] = self.run_image_query(input_data["image"])
                pass

            if "photometry" in input_data:
                # results["photometry"] = self.run_photometry_query(input_data["photometry"])
                pass

            return results

        except Exception as e:
            return {"error": str(e)}

    def run(self):
        """
        Execute the main control flow of the object detection process.

        This method parses arguments, runs the ephemeris query, and prints the results.

        Returns:
            object: Results of the ephemeris query if successful, None otherwise.
        """

        self.parse_args()
        results = self.run_ephemeris_query()

        if results:
            print(f"Successfully queried {len(results) if isinstance(results, list) else 1} object(s)")
            return results
        else:
            print("No results obtained")


if __name__ == "__main__":
    controller = ObjectDetectionController()
    controller.run()
