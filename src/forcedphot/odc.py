import argparse
import logging
from dataclasses import asdict
from typing import Any, Optional

import astropy.units as u
from astropy.time import Time

from forcedphot.ephemeris.data_loader import DataLoader
from forcedphot.ephemeris.data_model import QueryResult
from forcedphot.ephemeris.ephemeris_client import EphemerisClient
from forcedphot.image_photometry.image_service import ImageService
from forcedphot.image_photometry.imphot_control import ImPhotController
from forcedphot.image_photometry.photometry_service import PhotometryService
from forcedphot.image_photometry.utils import EphemerisDataCompressed, ImageMetadata

logger = logging.getLogger("odc")

class ObjectDetectionController:
    """
    This class handles argument parsing, ephemeris queries, and overall control
    flow for object detection tasks for forced photometry.
    """

    def __init__(self):
        """
        Initialize the ObjectDetectionController.

        Sets up the argument parser and initializes the args attribute.
        """
        self.parser = self.create_parser()
        self.args = argparse.Namespace()
        self.args.filters = ["r"]
        self.logger = logging.getLogger("odc")
        self.ephemeris_client = EphemerisClient()
        self.ephemeris_results: list[EphemerisDataCompressed] = []
        self.image_service = ImageService()
        self.image_results : list[ImageMetadata]
        self.photometry_service = PhotometryService()
        self.imphot_controller = ImPhotController()

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
        parser.add_argument("--ephem-ecsv", help="Path to ECSV file containing input data")
        parser.add_argument("--csv", help="Path to CSV file for batch processing")

        # Other options
        parser.add_argument(
            "--location", default="X05", help="Observer location code (default: X05 for Rubin Observatory)"
        )
        parser.add_argument("--save-data", action="store_true", help="Save query results (Ephemeris data)")

        # Image service
        parser.add_argument(
            "--ephemeris_file",
            help="Path to the ephemeris file"
            )

        parser.add_argument(
        "--filters",
        "--f",
        nargs='+',
        choices=["u", "g", "r", "i", "z", "y"],
        default=["r"],
        help="List of filters for image search (e.g., --filters g r i)",
        )

        parser.add_argument(
            "--output-dir",
            default="./output",
            help="Directory to save output files (default: ./output)"
        )

        parser.add_argument(
            "--save-cutouts",
            action="store_true",
            help="Save image cutouts (default: False)",
        )

        parser.add_argument(
            "--min-cutout-size",
            type=int,
            default=800,
            help="Minimum size of cutouts (default: 800)",
        )

        # Catalog service
        # parser.add_argument(
        #     "--catalog-service",
        #     choices=["LSST-TAP", "ZTF"],
        #     default="LSST-TAP",
        #     help="Catalog service to use",
        # )

        # parser.add_argument(
        #     "-x",
        #     "--max-search-ellipse",
        #     type=float,
        #     help="Maximum size of error ellipse to search (3-sigma value)",
        # )

        # Phometry service
        parser.add_argument(
            "--photometry-service",
            choices=["Rubin", "choice2"],
            default="Rubin",
            help="Photometry service to use",
        )

        parser.add_argument(
            "--image-type",
            choices=["calexp", "goodSeeingDiff_differenceExp"],
            default="calexp",
            help="Select the type of image. calexp or goodSeeingDiff_differenceExp"
        )

        parser.add_argument(
            "--threshold", type=int, default=3, help="Threshold SNR for forced photometry (defaullt: 3)"
        )

        parser.add_argument(
            "--display",
            action="store_true",
            help="Display the images and the error ellipses in Firefly (default: False)"
        )

        parser.add_argument(
            "--save_json",
            action="store_true",
            help="Save results as JSON (default: False)",
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
        logger.info(f"Starting query for object name: {self.args.target}")
        # Load ephemeris data from ecsv into Ephemeris dataclass
        if self.args.ephem_ecsv:
            ephemeris_data = DataLoader.load_ephemeris_from_ecsv(self.args.ephem_ecsv)
            return QueryResult(self.args.target, self.args.start_time, self.args.end_time, ephemeris_data)

        # Batch process from csv file
        elif self.args.csv:
            return self.ephemeris_client.query_from_csv(
                self.args.ephemeris_service, self.args.csv, self.args.location, self.args.save_data
            )
        # Single query
        elif all([self.args.target, self.args.target_type, self.args.start_time, self.args.end_time]):
            return self.ephemeris_client.query_single(
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


    def run_image_query(self, ephemeris_results: Optional[dict] = None) -> list[Any]:
        """Execute image search using ephemeris data."""
        # Determine ephemeris source
        if self.args.ephem_ecsv:
            ephemeris_data_temp = DataLoader.load_ephemeris_from_ecsv(self.args.ephem_ecsv)
            self.ephemeris_results = QueryResult(
                self.args.target,
                self.args.start_time,
                self.args.end_time,
                ephemeris_data_temp
            )
        else:
            if not self.ephemeris_results:
                raise ValueError("Run ephemeris query first or provide --ephem-ecsv")

        # Configure search parameters
        if ephemeris_results:
            image_metadata = self.imphot_controller.search_images()
        else:
            bands = set(self.args.filters)
            self.imphot_controller.configure_search(
                bands=bands,
                ephemeris_data=self.ephemeris_results
            )

            # Execute search
            image_metadata = self.imphot_controller.search_images()
        return image_metadata

    def run_photometry(self, image_results: list[Any]) -> list[Any]:
        """Execute photometry on the image results."""
        # Configure photometry parameters
        self.imphot_controller.detection_threshold = self.args.threshold
        # Execute photometry
        self.imphot_controller.process_images(
            target_name=self.args.target,
            target_type=self.args.target_type,
            image_type=self.args.image_type,
            ephemeris_service=self.args.ephemeris_service,
            image_metadata=image_results,
            save_cutout=self.args.save_cutouts,
            cutout_size=self.args.min_cutout_size,
            display=self.args.display,
        )
        if self.args.save_json:
            self.imphot_controller.save_results(target_name=self.args.target)

        self.imphot_controller.print_summary()

        print("-" * 40)
        print("Photometry Service is done.")
        print("-" * 40)

    def api_connection(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Handle API connections and execute the object detection process based on input data.

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
                self.args.target = ephemeris_data.get("target")
                self.args.target_type = ephemeris_data.get("target_type")
                self.args.ephemeris_service = ephemeris_data.get("ephemeris_service")

                if "ecsv_file" in ephemeris_data:
                    loaded_ephemeris = self.ephemeris_client.load_ephemeris_from_ecsv(
                        ephemeris_data["ecsv_file"]
                    )

                    results["ephemeris"] = QueryResult(
                        ephemeris_data.get("target", "UploadedData"),
                        ephemeris_data.get("start"),
                        ephemeris_data.get("end"),
                        loaded_ephemeris
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


            # Handle image and photometry services
            process_image = "image" in input_data
            process_photometry = "photometry" in input_data

            if process_image:
                # Configure image parameters from input_data
                self.logger.info("Running Image search")
                image_params = input_data.get("image", {})
                self.args.filters = image_params.get("filters", ["r"])
                self.args.ephem_ecsv = image_params.get("ephemeris_file")
                ephemeris_data = image_params.get("ephemeris_data")

                self.imphot_controller.configure_search(
                    bands=self.args.filters,
                    ephemeris_data=ephemeris_data
                )

                self.image_results = self.run_image_query(ephemeris_data)
                results["image"] = [asdict(md) for md in self.image_results] if self.image_results else None

            # Process photometry if requested and image results exist
            if process_photometry:
                self.logger.info("Running Photometry")
                photometry_params = input_data.get("photometry", {})
                self.args.image_type = photometry_params.get("image_type", "calexp")
                self.args.threshold = photometry_params.get("threshold", 5)
                self.args.save_cutouts = photometry_params.get("save_cutouts", False)
                self.args.min_cutout_size = photometry_params.get("min_cutout_size", 800)
                self.args.display = photometry_params.get("display", False)
                self.args.save_json = photometry_params.get("save_json", False)

                if self.image_results:
                    self.run_photometry(self.image_results)
                    results["photometry"] = [asdict(res) for res in self.imphot_controller.results]
                else:
                    results["photometry"] = None

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

        if self.args.service_selection in ["all", "ephemeris"]:
            self.ephemeris_results = self.run_ephemeris_query()
            print(f"Ephemeris data loaded: {len(self.ephemeris_results.ephemeris.datetime)} entries")

        if self.args.service_selection in ["image"]:
            self.image_results = self.run_image_query()

        if self.args.service_selection in ["all", "photometry"]:
            self.image_results = self.run_image_query()

            if self.image_results:
                photometry_results = self.run_photometry(self.image_results)
                return photometry_results


if __name__ == "__main__":
    controller = ObjectDetectionController()
    controller.run()
