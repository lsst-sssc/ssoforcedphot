"""
Control module for image search (based on ephemeris data) on ObsCore database
and photometry on the found images.
"""

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Optional

from forcedphot.image_photometry.image_service import ImageService
from forcedphot.image_photometry.photometry_service import PhotometryService
from forcedphot.image_photometry.utils import (
    EndResult,
    ImageMetadata,
    SearchParameters,
)


class ImPhotController:
    """
    A controller class to manage the image service and photometry service modules.

    Handles image search, photometry processing, result aggregation, and output.

    Attributes:
        image_service (ImageService): Service for image search operations.
        phot_service (PhotometryService): Service for photometry measurements.
        search_params (SearchParameters): Configuration for image search.
        image_metadata (list[ImageMetadata]): Results from image search.
        results (list[EndResult]): Aggregated photometry results.
        output_dir (str): Directory for saving outputs.
    """

    def __init__(self, detection_threshold: float = 5, output_dir: str = "./output"):
        """
        Initialize the controller with services and default settings.

        Args:
            detection_threshold: SNR threshold for source detection (default: 5).
            output_dir: Directory to save results (default: "./output").
        """
        self.logger = logging.getLogger("im_phot_controller")
        self.image_service = ImageService()
        self.phot_service = PhotometryService(detection_threshold=detection_threshold)
        self.search_params: Optional[SearchParameters] = None
        self.image_metadata: Optional[list[ImageMetadata]] = None
        self.results: list[EndResult] = []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def configure_search(self, bands: set[str], ephemeris_data) -> None:
        """
        Set up search parameters for image retrieval.

        Args:
            bands: Set of photometric bands to search (e.g., {'g', 'r'}).
            ephemeris_data: Path to the ephemeris data file or EphemerisData dataclass object.
        """
        self.search_params = SearchParameters(bands=bands, ephemeris_file=ephemeris_data)

    def search_images(self):
        """
        Execute the image search using configured parameters.

        Returns:
            True if images were found, False otherwise.
        """
        if self.search_params is None:
            raise ValueError("Search parameters not configured. Call configure_search() first.")

        self.image_metadata = self.image_service.search_images(
            self.search_params.bands,
            self.search_params.ephemeris_file
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
        save_cutout: bool = False,
        display: bool = True
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
            save_cutout: Whether to save cutout images (default: False).
            display: Whether to display results (default: True).
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
                save_cutout=save_cutout,
                display=display,
                output_dir=self.output_dir,
                save_json=False  # Handled separately by save_results()
            )
            self.results.append(result)

    def save_results(self, filename: str = "photometry_results.json",
                     target_name: Optional[str] = "target") -> str:
        """
        Save all photometry results to a JSON file.

        Args:
            filename: Name of the output file (default: "photometry_results.json").
            target_name: Optional. Name of the target.

        Returns:
            Path to the saved file.
        """
        if not self.results:
            raise ValueError("No results to save. Run process_images() first.")

        json_data = [asdict(result) for result in self.results]

        filename = target_name + "_photometry_results.json"

        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

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
    start_time = time.time()
    controller = ImPhotController(detection_threshold=5, output_dir="./output")

    # Configure and execute search
    controller.configure_search(
        bands={'g', 'r', 'i'},
        ephemeris_data='./test_ephem.ecsv'
    )

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
        display=False
    )

    controller.save_results()
    controller.print_summary()
    print(f"The process took {(time.time()-start_time)/60} minutes.")
