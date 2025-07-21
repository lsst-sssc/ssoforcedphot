"""
Control module for image search (based on ephemeris data) on ObsCore database
and photometry on the found images.
"""

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Optional, Any

from image_photometry.image_service_butler import ImageServiceButler
from image_photometry.image_service import ImageService
from image_photometry.photometry_service import PhotometryService
from image_photometry.polygon import calculate_polygons
from image_photometry.utils import (
    QueryResult,
    EndResult,
    ImageMetadata,
    SearchParameters,
    SearchParametersPolygon,
    EphemerisDataCompressed,
)


class ImPhotController:
    """
    A controller class to manage the image service and photometry service modules.

    Handles image search, photometry processing, result aggregation, and output.

    Attributes:
        image_service (ImageService): Service for image search operations.
        phot_service (PhotometryService): Service for photometry measurements.
        search_params (SearchParameters): Configuration for image search.
        search_params_polygon (SearchParametersPolygon): Config for image search with polygon.
        image_metadata (list[ImageMetadata]): Results from image search.
        results (list[EndResult]): Aggregated photometry results.
        output_folder (str): Directory for saving outputs.
    """

    def __init__(self, detection_threshold: float = 5, output_folder: str = "./output"):
        """
        Initialize the controller with services and default settings.

        Args:
            detection_threshold: SNR threshold for source detection (default: 5).
            output_folder: Directory to save results (default: "./output").
        """
        self.logger = logging.getLogger("im_phot_controller")
        self.image_service = ImageService()
        self.image_service_butler = ImageServiceButler()
        self.phot_service = PhotometryService(detection_threshold=detection_threshold)
        self.search_params: Optional[SearchParameters] = None
        self.search_params_polygon: Optional[SearchParametersPolygon] = None
        self.polygon_data: Optional[list] = None
        self.image_metadata: Optional[list[ImageMetadata]] = None
        self.results: list[EndResult] = []
        self.output_folder = output_folder
        # os.makedirs(self.output_folder, exist_ok=True)

    def configure_search(self, bands: set[str], ephemeris_data: Any, time_interval: Optional[float] = 5, widening: Optional[float] = 1) -> None:
        """
        Set up search parameters for image retrieval.

        Args:
            bands: Set of photometric bands to search (e.g., {"g", "r"}).
            ephemeris_data: Path to the ephemeris data file or EphemerisData dataclass object.
            time_interval: The maximum duration for each polygon segment in days.
            widening: The desired width of the polygon on either side of the path, in arcseconds. 
        """
        self.search_params = SearchParameters(bands=bands, ephemeris_file=ephemeris_data)
        self.search_params_polygon = SearchParametersPolygon(bands=bands, ephemeris_file=ephemeris_data, time_interval=time_interval, widening=widening)

    def search_images(self) -> list[ImageMetadata]:
        """
        Execute the image search using configured parameters (point-based).

        Returns:
            list[ImageMetadata]: A list of metadata for the found images.
        """
        if self.search_params is None:
            raise ValueError("Search parameters not configured. Call configure_search() first.")

        # self.logger.info("Starting point-based image search.")
        self.image_metadata = self.image_service.search_images(
            self.search_params.bands, self.search_params.ephemeris_file
        )
        return self.image_metadata

    def search_images_polygon(self) -> list[ImageMetadata]:
        """
        Execute the image search using configured parameters.

        Returns:
            list[ImageMetadata]: A list of metadata for the found images.
        """
        if self.search_params_polygon is None:
            raise ValueError("Search parameters not configured. Call configure_search() first.")

        # Calculate the polygon corners and time boundaries
        self.logger.info(
            f"Calculating polygon with time interval: {self.search_params_polygon.time_interval} days "
            f"and widening: {self.search_params_polygon.widening} arcsec"
        )
        if isinstance(self.search_params_polygon.ephemeris_file, QueryResult):
                ephemeris_compressed = EphemerisDataCompressed.compress_ephemeris(self.search_params_polygon.ephemeris_file)
        else:
            ephemeris_compressed = self.search_params_polygon.ephemeris_file

        self.polygon_data = calculate_polygons(
            ephemeris_data = ephemeris_compressed,
            time_interval = self.search_params_polygon.time_interval,
            widening = self.search_params_polygon.widening
        )
        
        if not self.polygon_data:
            self.logger.warning("Polygon calculation resulted in no data. Aborting search.")
            return []
        
        # Search images based on the polygons
        self.logger.info(f"Searching for images intersecting with {len(self.polygon_data)} polygons.")
        self.image_metadata = self.image_service_butler.search_images_polygon(
            polygons = self.polygon_data,
            bands = self.search_params_polygon.bands,
            ephemeris = ephemeris_compressed,
        )

        return self.image_metadata
    
    def process_images(
        self,
        target_name: str,
        target_type: str,
        image_type: str,
        ephemeris_service: str,
        image_metadata: list[ImageMetadata],
        cutout_size: int = 800,
        override_error: float = 0,
        save_cutout: bool = False,
        display: bool = True,
        output_folder: str = "./output",
        save_json: bool = False,
        save_csv: bool = False,
    ) -> None:
        """
        Perform photometry on all retrieved images.

        Args:
            target_name: Name of the target (e.g., "Example Target").
            target_type: Classification of the target (e.g., "smallbody").
            image_type: Type of the image (calexp, goodSeeingDiff_differenceExp).
            ephemeris_service: Source of ephemeris data (e.g., "JPL Horizons").
            image_metadata: Metadata for the images.
            cutout_size: Size of image cutout in pixels (default: 800).
            override_error: Overwrite the error ellipse with the value (cicle).
            save_cutout: Whether to save cutout images (default: False).
            display: Whether to display results (default: True).
            output_folder: Directory for output files, by default None
            save_json: Whether to save results as JSON, by default False
            save_csv: Whether to save results as csv, by default False
        """
        if image_metadata is None:
            raise ValueError("No images found. Run search_images() first.")

        self.results.clear()
        for metadata in image_metadata:
            result = self.phot_service.process_image(
                image_metadata=metadata,
                target_name=target_name,
                target_type=target_type,
                image_type=image_type,
                ephemeris_service=ephemeris_service,
                cutout_size=cutout_size,
                override_error=override_error,
                save_cutout=save_cutout,
                display=display,
                output_folder=output_folder,
                save_json=save_json,
                save_csv=save_csv,
            )
            # Skip results where target photometry is None (edge proximity)
            if result.forced_phot_on_target is not None:
                self.results.append(result)
            else:
                self.logger.warning(
                    f"Skipping image (Visit ID: {metadata.visit_id}, Detector: {metadata.detector_id}) "
                    "due to target proximity to edge or outside of the image boundaries."
                )

    def save_results_to_json(
        self, target_name: Optional[str] = "target", output_folder: str = "./output"
    ) -> str:
        """
        Save all photometry results to a JSON file.

        Args:
            target_name: Optional. Name of the target.
            output_folder: Directory for output files, by default ./output

        Returns:
            Path to the saved file.
        """
        if not self.results:
            raise ValueError("No results to save. Run process_images() first.")

        json_data = [asdict(result) for result in self.results]
        t_name = str(target_name).replace(":", "-").replace(" ", "_").replace("/", "_")
        filename = t_name + "_photometry_results.json"

        output_path = os.path.join(output_folder, filename)

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        print(f"Results saved to {output_path}")
        return output_path
    
    def save_results_to_csv(
        self, target_name: Optional[str] = "target", output_folder: str = "./output", all_ellipse_sources: bool = False
    ) -> str:
        """
        Save all photometry results to a csv file.

        Args:
            target_name: Optional. Name of the target.
            output_folder: Directory for output files, by default ./output
            all_ellipse_sources: If True, create separate rows for each source within error ellipse.
                If False, only include the best source per result. Default is False.

        Returns:
            Path to the saved file.
        """
        if not self.results:
            raise ValueError("No results to save. Run process_images() first.")

        t_name = str(target_name).replace(":", "-").replace(" ", "_").replace("/", "_")
        filename = t_name + "_photometry_results.csv"

        output_path = os.path.join(output_folder, filename)

        EndResult.save_results_to_csv(self.results, output_path, include_all_ellipse_sources=all_ellipse_sources)

        print(f"Results saved to {output_path}")
        return output_path


    
    def print_summary(self) -> None:
        """Print a consolidated summary of all photometry results."""
        for i, result in enumerate(self.results):
            print(f"\n{'=' * 30} Result {i + 1} {'=' * 30}")
            print(f"Visit ID: {result.visit_id} | Detector: {result.detector_id} | Band: {result.band}")

            if result.forced_phot_on_target:
                phot = result.forced_phot_on_target
                print("\nTarget Photometry:")
                print(f"RA: {phot.ra:.5f}°, Dec: {phot.dec:.5f}°")
                print(f"Flux: {phot.flux:.2f} ± {phot.flux_err:.2f} nJy")
                print(f"Magnitude: {phot.mag:.2f} ± {phot.mag_err:.2f} (AB)")
                print(f"SNR: {phot.snr:.1f}")

            if result.phot_within_error_ellipse:
                print(f"\n{len(result.phot_within_error_ellipse)} nearby sources:")
                for j, source in enumerate(result.phot_within_error_ellipse, 1):
                    print(f"  Source {j}: Separation = {source.separation:.2f} arcsec")
                    print(f"    Flux: {source.flux:.2f} ± {source.flux_err:.2f} nJy")
                    print(f"    Sigma: {source.sigma:.2f}σ")

    

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_time = time.time()
    controller = ImPhotController(detection_threshold=5, output_folder="./output")

    # Configure and execute search
    # This is just an example path, ensure it exists or is passed via command line in a real scenario.
    ephem_file_path = "./test_ephem.ecsv"
    if not os.path.exists(ephem_file_path):
        print(f"Warning: Example ephemeris file not found at {ephem_file_path}")
        exit()
        
    controller.configure_search(bands={"g", "r", "i"}, ephemeris_data=ephem_file_path)

    image_metadata = controller.search_images()

    if not image_metadata:
        print("No images found!")
        exit()

    # Process images and save results
    controller.process_images(
        target_name="Example Target",
        target_type="smallbody",
        image_type="goodSeeingDiff_differenceExp",
        ephemeris_service="JPL Horizons",
        image_metadata=image_metadata,
        cutout_size=800,
        display=False,
    )

    controller.save_results(target_name="Example_Target")
    controller.print_summary()
    print(f"The process took {(time.time()-start_time)/60:.2f} minutes.")
