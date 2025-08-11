import argparse
import logging
from dataclasses import asdict
from typing import Any, Optional
import time 

import astropy.units as u
from astropy.time import Time
from ephemeris.data_loader import DataLoader
from ephemeris.data_model import QueryResult
from ephemeris.ephemeris_client import EphemerisClient
from image_photometry.image_service import ImageService
from image_photometry.image_service_butler import ImageServiceButler
from image_photometry.imphot_control import ImPhotController
from image_photometry.photometry_service import PhotometryService
from image_photometry.utils import EphemerisDataCompressed, ImageMetadata

logger = logging.getLogger("odc")
logging.getLogger("httpx").setLevel(logging.WARNING)


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
        self.args.time_interval = 5.0
        self.args.widening = 1.0
        self.args.output_folder = "./output"
        self.logger = logging.getLogger("odc")
        self.ephemeris_client = EphemerisClient()
        self.ephemeris_results: list[EphemerisDataCompressed] = []
        self.image_service = ImageService()
        self.image_service_butler = ImageServiceButler()
        self.image_results: list[ImageMetadata]
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
        parser.add_argument("--save-ephem-data", action="store_true", help="Save query results (Ephemeris data)")

        # Image service
        parser.add_argument("--ephemeris_file", help="Path to the ephemeris file")

        parser.add_argument(
            "--filters",
            "--f",
            nargs="+",
            choices=["u", "g", "r", "i", "z", "y"],
            default=["r"],
            help="List of filters for image search (e.g., --filters g r i)",
        )

        parser.add_argument(
            "--output-folder", default="./output", help="Directory to save output files (default: ./output)"
        )

        parser.add_argument(
            "--min-cutout-size",
            type=int,
            default=800,
            help="Minimum size of cutouts (default: 800)",
        )

        # Phometry service
        parser.add_argument(
            "--photometry-service",
            choices=["Rubin", "choice2"],
            default="Rubin",
            help="Photometry service to use",
        )

        parser.add_argument(
            "--image-type",
            choices=["visit_image", "goodSeeingDiff_differenceExp"],
            default="visit_image",
            help="Select the type of image. visit_image or goodSeeingDiff_differenceExp",
        )

        parser.add_argument(
            "--image-search-method",
            choices=["point", "polygon"],
            default="point",
            help="Select the method for image search. Point: based on the ephemeris rows (slower for longer ephemeris data), Polygon: create a polygon from the ephemeris data, and look for overlapsed images",
        )

        # Parameters for polygon search
        parser.add_argument(
            "--time-interval",
            type=float,
            default=5.0,
            help="Time interval in days for polygon search method (default: 1.0 day)",
        )

        parser.add_argument(
            "--widening",
            type=float,
            default=1,
            help="Widening factor for polygon search method (default: 1 arcsec)",
        )
        
        parser.add_argument(
            "--threshold", type=int, default=3, help="Threshold SNR for forced photometry (defaullt: 3)"
        )

        parser.add_argument(
            "--override-error",
            type=float,
            default=0,
            help="Override the error ellipse with a user defined circle, default: 0 (no override)",
        )
        
        parser.add_argument(
            "--display",
            action="store_true",
            help="Display the images and the error ellipses in Firefly (default: False)",
        )

        # Save options
        parser.add_argument(
            "--save-json",
            action="store_true",
            help="Save results as JSON (default: False)",
        )

        parser.add_argument(
            "--save-csv",
            action="store_true",
            help="Save results as csv (default: False)",
        )

        parser.add_argument(
            "--all-ellipse-sources",
            action="store_true",
            help="If True, create separate rows (in the result csv file) for each source within error ellipse. If False, only include the best source per result. Default is False.",
        )
        
        parser.add_argument(
            "--save-diag-plots",
            action="store_true",
            help="Save diagnostic images (png) marked with sources and error ellipse (default: False)",
        )

        parser.add_argument(
            "--save-fits",
            action="store_true",
            help="Save the fits images (default: False)",
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

        print("-" * 40)
        print("The ephemeris service is now initiated.")
        print("-" * 40)

        logger.info(f"Starting query for object name: {self.args.target}")
        # Load ephemeris data from ecsv into Ephemeris dataclass
        if self.args.ephem_ecsv:
            ephemeris_data = DataLoader.load_ephemeris_from_ecsv(self.args.ephem_ecsv)
            return QueryResult(self.args.target, self.args.start_time, self.args.end_time, ephemeris_data)

        # Batch process from csv file
        elif self.args.csv:
            return self.ephemeris_client.query_from_csv(
                self.args.ephemeris_service, self.args.csv, self.args.location, self.args.save_ephem_data
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
                self.args.save_ephem_data,
                self.args.output_folder,
            )
        else:
            self.parser.error("Either provide a CSV/ECSV file or all single query parameters")

    def run_image_query(self, ephemeris_results: Optional[dict] = None, search_method: Optional[str] = "point") -> list[Any]:
        """
        Execute image search using ephemeris data with the specified search method.
        
        Args:
            ephemeris_results: Optional ephemeris data dictionary
            search_method: Optional search method override ('point' or 'polygon')
            
        Returns:
            list[Any]: List of image metadata results
        """

        print("-" * 40)
        print("The image search is now initiated.")
        print("-" * 40)

        start_time = time.time()
        
        # Determine ephemeris source
        if self.args.ephem_ecsv:
            ephemeris_data_temp = DataLoader.load_ephemeris_from_ecsv(self.args.ephem_ecsv)
            self.ephemeris_results = QueryResult(
                self.args.target, self.args.start_time, self.args.end_time, ephemeris_data_temp
            )
        elif ephemeris_results:
            # self.logger.info("Loading ephemeris data.")
            self.ephemeris_results = ephemeris_results
        else:
            if not self.ephemeris_results:
                raise ValueError("Run ephemeris query first or provide --ephem-ecsv")

        # Use provided search method or fall back to args
        effective_search_method = self.args.image_search_method or search_method.lower()
        
        # Configure search parameters
        self.imphot_controller.configure_search(
            bands=self.args.filters,
            ephemeris_data=self.ephemeris_results,
            time_interval=self.args.time_interval,
            widening=self.args.widening,
        )
        
        # Execute search based on the selected method
        if effective_search_method == "polygon":
            self.logger.info("Using polygon-based image search method")
            image_metadata = self.imphot_controller.search_images_polygon()
        else:
            self.logger.info("Using point-based image search method")
            image_metadata = self.imphot_controller.search_images()

        print("-" * 40)
        print("The image search has been completed.")
        print(f"The process took {(time.time()-start_time)/60:.2f} minutes.")
        print("-" * 40)
        
        return image_metadata

    def run_photometry(self, image_results: list[Any], output_folder: Optional[str] = "./output") -> list[Any]:
        """Execute photometry on the image results."""

        print("-" * 40)
        print("The photometry service is now initiated.")
        print("-" * 40)

        start_time = time.time()
        
        # Configure photometry parameters
        self.imphot_controller.detection_threshold = self.args.threshold
        # Execute photometry
        self.imphot_controller.process_images(
            target_name=self.args.target,
            target_type=self.args.target_type,
            image_type=self.args.image_type,
            ephemeris_service=self.args.ephemeris_service,
            image_metadata=image_results,
            save_diag_plots=self.args.save_diag_plots,
            save_fits=self.args.save_fits,
            cutout_size=self.args.min_cutout_size,
            override_error=self.args.override_error,
            display=self.args.display,
            output_folder=output_folder,
        )
        if self.args.save_json:
            self.imphot_controller.save_results_to_json(
                    target_name=self.args.target,
                    output_folder=self.args.output_folder,
                )

        if self.args.save_csv:
            self.imphot_controller.save_results_to_csv(
                    target_name=self.args.target,
                    output_folder=self.args.output_folder,
                    all_ellipse_sources=self.args.all_ellipse_sources,
                )

        self.imphot_controller.print_summary()

        print("-" * 40)
        print("The photometry service has been completed.")
        print(f"The process took {(time.time()-start_time)/60:.2f} minutes.")
        print("-" * 40)

        return self.imphot_controller.results

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

            # Set output folder from input data if provided
            if "output_folder" in input_data:
                self.args.output_folder = input_data["output_folder"]

            if "ephemeris" in input_data:
                ephemeris_input = input_data["ephemeris"]
                self.args.target = ephemeris_input.get("target")
                self.args.target_type = ephemeris_input.get("target_type")
                self.args.ephemeris_service = ephemeris_input.get("ephemeris_service")

                if "ecsv_file" in ephemeris_input:
                    loaded_ephemeris = self.ephemeris_client.load_ephemeris_from_ecsv(
                        ephemeris_input["ecsv_file"]
                    )

                    results["ephemeris"] = QueryResult(
                        ephemeris_input.get("target", "UploadedData"),
                        ephemeris_input.get("start"),
                        ephemeris_input.get("end"),
                        loaded_ephemeris,
                    )

                elif "ephemeris_service" in ephemeris_input:
                    if "csv_file" in ephemeris_input:
                        results["ephemeris"] = self.ephemeris_client.query_from_csv(
                            service=ephemeris_input.get("ephemeris_service"),
                            csv_file=ephemeris_input.get("csv_file"),
                            observer_location=ephemeris_input.get(
                                "observer_location", EphemerisClient.DEFAULT_OBSERVER_LOCATION
                            ),
                            save_ephem_data=ephemeris_input.get("save_ephem_data", EphemerisClient.DEFAUT_save_ephem_data),
                            output_folder=self.args.output_folder,
                        )
                    else:
                        try:
                            results["ephemeris"] = self.ephemeris_client.query_single(
                                service=ephemeris_input.get("ephemeris_service"),
                                target=ephemeris_input.get("target"),
                                target_type=ephemeris_input.get("target_type"),
                                start=ephemeris_input.get("start"),
                                end=ephemeris_input.get("end"),
                                step=ephemeris_input.get("step"),
                                observer_location=ephemeris_input.get("observer_location"),
                                save_ephem_data=ephemeris_input.get("save_ephem_data"),
                                output_folder=ephemeris_input.get("output_folder"),
                            )
                        except Exception as e:
                            root_logger.error(f"Ephemeris query failed: {e}")
                            self.result_pane.object = {"error": str(e)}
                else:
                    raise ValueError("Invalid ephemeris query parameters")

            # Handle image and photometry services
            process_image = "image" in input_data
            process_photometry = "photometry" in input_data

            if process_image:
                # Configure image parameters from input_data
                # self.logger.info("Running Image search")
                image_params = input_data.get("image", {})
                self.args.filters = image_params.get("filters", ["r"])
                self.args.ephem_ecsv = image_params.get("ephemeris_file")
                self.args.image_search_method = image_params.get("image_search_method", "point")
                self.args.time_interval = image_params.get("time_interval", 1.0)
                self.args.widening = image_params.get("widening", 1.0)
                ephemeris_data = image_params.get("ephemeris_data")
                
                # Pass the search method to run_image_query
                search_method = image_params.get("image_search_method")
                self.image_results = self.run_image_query(ephemeris_results = ephemeris_data["ephemeris"], search_method = search_method)
                
                results["image"] = [asdict(md) for md in self.image_results] if self.image_results else None

            # Process photometry if requested and image results exist
            if process_photometry:
                self.logger.info("Running Photometry")
                photometry_params = input_data.get("photometry", {})
                self.args.image_type = photometry_params.get("image_type", "visit_image")
                self.args.ephemeris_service = self.args.ephemeris_service
                self.args.threshold = photometry_params.get("threshold", 5)
                self.args.save_diag_plots = photometry_params.get("save_diag_plots", False)
                self.args.save_fits = photometry_params.get("save_fits", False)
                self.args.min_cutout_size = photometry_params.get("min_cutout_size", 800)
                self.args.override_error = photometry_params.get("override_error")
                self.args.display = photometry_params.get("display", False)
                self.args.save_json = photometry_params.get("save_json", False)
                self.args.save_csv = photometry_params.get("save_csv", False)
                self.args.all_ellipse_sources = photometry_params.get("save_error_sources", False)
                self.args.output_folder = photometry_params.get("output_folder", "./output")

                if self.image_results:
                    photometry_results = self.run_photometry(self.image_results, self.args.output_folder)
                    results["photometry"] = [asdict(res) for res in photometry_results]
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
        start_time = time.time()
        
        if self.args.service_selection in ["ephemeris"]:
            self.ephemeris_results = self.run_ephemeris_query()
            print(f"Ephemeris data loaded: {len(self.ephemeris_results.ephemeris.datetime)} entries")

        if self.args.service_selection in ["image"]:
            self.ephemeris_results = self.run_ephemeris_query()
            self.image_results = self.run_image_query()

        if self.args.service_selection in ["all", "photometry"]:
            self.ephemeris_results = self.run_ephemeris_query()
            self.image_results = self.run_image_query()

            if self.image_results:
                photometry_results = self.run_photometry(self.image_results, self.args.output_folder)
                # print(photometry_results)

        print(f"The total duration of the process was {(time.time()-start_time)/60:.2f} minutes.")
        
if __name__ == "__main__":
    controller = ObjectDetectionController()
    controller.run()
