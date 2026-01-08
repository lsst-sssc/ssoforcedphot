"""
Standalone Photometry API Module

Provides photometry services without ephemeris dependency, enabling
measurements at arbitrary coordinates in LSST images.

This module decouples photometry operations from ephemeris queries, allowing users to:
- Measure flux at any coordinate without ephemeris data
- Re-measure images without re-querying Horizons/Miriade
- Support non-SSO science cases (transients, variable stars, manual detections)
- Process batch measurements from CSV files
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from image_photometry.photometry_service import PhotometryService
from image_photometry.utils import (
    EndResult,
    EphemerisDataCompressed,
    ImageMetadata,
)
from lsst.daf.butler import Butler


@dataclass
class PhotometryRequest:
    """
    Single photometry measurement request.

    This dataclass specifies a single forced photometry measurement
    at a given coordinate in a specific image.

    Attributes
    ----------
    visit_id : int
        LSST visit identifier
    detector : int
        LSST detector ID (CCD number)
    band : str
        Photometric band ('u', 'g', 'r', 'i', 'z', 'y')
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    error_radius : float, optional
        Circular search radius in arcseconds for additional sources.
        If 0, only forced photometry at exact coordinates is performed.
        Default: 3.0 arcsec
    detection_threshold : float, optional
        SNR threshold for source detection within error_radius.
        Default: 5.0
    image_type : str, optional
        Type of image to process: "visit_image" or "difference_image".
        Default: "visit_image"
    aperture_radii : list[float], optional
        List of aperture radii in arcseconds for aperture photometry.
        If None, only PSF photometry is performed.
        Example: [3.0, 5.0, 7.0]
        Default: None (not yet implemented)
    target_name : str, optional
        Optional name for this target (for output labeling).
        Default: "standalone_target"
    """

    visit_id: int
    detector: int
    band: str
    ra: float
    dec: float

    # Optional parameters
    error_radius: float = 3.0
    detection_threshold: float = 5.0
    image_type: str = "visit_image"
    aperture_radii: Optional[list[float]] = None
    target_name: str = "standalone_target"


class StandalonePhotometryService:
    """
    Performs forced photometry from user-provided coordinates without ephemeris.

    This service enables photometry measurements at arbitrary celestial coordinates,
    decoupling the measurement process from ephemeris dependency. It's useful for:
    - Measuring coordinates from external catalogs
    - Re-measuring previously processed images
    - Transient follow-up observations
    - Non-SSO science cases

    The service wraps the existing PhotometryService by creating synthetic
    ImageMetadata objects from user-provided coordinates.

    Parameters
    ----------
    dr : str
        Data release identifier (default: "dp1")
    collection : str
        Butler collection to access (default: "LSSTComCam/DP1")
    output_folder : str
        Default directory for saving outputs (default: "./output")
    detection_threshold : float
        Default SNR threshold for source detection (default: 5.0)
    """

    def __init__(
        self,
        dr: str = "dp1",
        collection: str = "LSSTComCam/DP1",
        output_folder: str = "./output",
        detection_threshold: float = 5.0,
    ):
        """Initialize the standalone photometry service."""
        self.logger = logging.getLogger("standalone_photometry")
        self.butler = Butler(dr, collections=collection)
        self.photometry_service = PhotometryService(
            dr=dr,
            collection=collection,
            detection_threshold=detection_threshold,
        )
        self.output_folder = output_folder
        self.dr = dr
        self.collection = collection

    def measure_single(
        self,
        request: PhotometryRequest,
        save_diag_plots: bool = False,
        save_fits: bool = False,
        output_folder: Optional[str] = None,
    ) -> EndResult:
        """
        Measure flux at a single coordinate in a single image.

        Parameters
        ----------
        request : PhotometryRequest
            Image and coordinate specification
        save_diag_plots : bool
            Save diagnostic PNG plot
            Default: False
        save_fits : bool
            Save FITS cutout
            Default: False
        output_folder : str, optional
            Where to save outputs (overrides default)

        Returns
        -------
        EndResult
            Photometry measurements including forced PSF flux, detected sources,
            and optional aperture photometry

        Raises
        ------
        ValueError
            If image doesn't exist or coordinates are invalid
        """
        self.logger.info(
            f"Measuring photometry for visit {request.visit_id}, "
            f"detector {request.detector}, band {request.band} "
            f"at RA={request.ra:.5f}, Dec={request.dec:.5f}"
        )

        # Use provided output folder or default
        out_folder = output_folder or self.output_folder

        # Validate coordinates
        self._validate_coordinates(request.ra, request.dec)

        # Validate image exists
        if not self._validate_image_exists(
            request.visit_id, request.detector, request.band, request.image_type
        ):
            raise ValueError(
                f"Image not found: visit={request.visit_id}, "
                f"detector={request.detector}, band={request.band}, "
                f"image_type={request.image_type}"
            )

        # Create synthetic ImageMetadata
        image_metadata = self._create_image_metadata(request)

        # Set detection threshold for this measurement
        self.photometry_service.detection_threshold = request.detection_threshold

        # Perform photometry using existing service
        result = self.photometry_service.process_image(
            image_metadata=image_metadata,
            target_name=request.target_name,
            target_type="standalone",
            image_type=request.image_type,
            ephemeris_service="N/A",
            cutout_size=800,
            override_error=request.error_radius,
            save_diag_plots=save_diag_plots,
            save_fits=save_fits,
            display=False,
            output_folder=out_folder,
        )

        return result

    def measure_batch(
        self,
        requests: list[PhotometryRequest],
        parallel: bool = True,
        max_workers: int = 4,
        save_diag_plots: bool = False,
        save_fits: bool = False,
        output_folder: Optional[str] = None,
    ) -> list[EndResult]:
        """
        Process multiple photometry requests in parallel.

        Parameters
        ----------
        requests : list[PhotometryRequest]
            Multiple measurements to perform
        parallel : bool
            Use multiprocessing for parallel execution
            Default: True
        max_workers : int
            Number of parallel workers
            Default: 4
        save_diag_plots : bool
            Save diagnostic plots for each measurement
            Default: False
        save_fits : bool
            Save FITS cutouts for each measurement
            Default: False
        output_folder : str, optional
            Where to save outputs

        Returns
        -------
        list[EndResult]
            Results for each request (same order as input)
        """
        self.logger.info(
            f"Processing {len(requests)} photometry requests "
            f"({'parallel' if parallel else 'sequential'})"
        )

        out_folder = output_folder or self.output_folder

        if not parallel or len(requests) == 1:
            # Sequential processing
            results = []
            for i, req in enumerate(requests):
                self.logger.info(f"Processing request {i+1}/{len(requests)}")
                try:
                    result = self.measure_single(
                        request=req,
                        save_diag_plots=save_diag_plots,
                        save_fits=save_fits,
                        output_folder=out_folder,
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing request {i+1}: {str(e)}")
                    results.append(None)
            return results

        # Parallel processing
        results = [None] * len(requests)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self._measure_single_wrapper,
                    req,
                    save_diag_plots,
                    save_fits,
                    out_folder,
                ): i
                for i, req in enumerate(requests)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    self.logger.info(f"Completed request {index+1}/{len(requests)}")
                except Exception as e:
                    self.logger.error(f"Error in request {index+1}: {str(e)}")
                    results[index] = None

        return results

    def measure_from_csv(
        self,
        csv_path: str,
        parallel: bool = True,
        max_workers: int = 4,
        save_diag_plots: bool = False,
        save_fits: bool = False,
        output_folder: Optional[str] = None,
        output_csv: Optional[str] = None,
        all_ellipse_sources: bool = False,
    ) -> pd.DataFrame:
        """
        Load requests from CSV and return results as DataFrame.

        CSV Format:
        ```
        visit_id,detector,band,ra,dec,error_radius,target_name
        512055,75,g,150.123,-23.456,5.0,target1
        512060,75,r,150.130,-23.450,5.0,target2
        ```

        Required columns: visit_id, detector, band, ra, dec
        Optional columns: error_radius, detection_threshold, image_type,
                         aperture_radii, target_name

        Parameters
        ----------
        csv_path : str
            Path to input CSV file
        parallel : bool
            Use parallel processing
            Default: True
        max_workers : int
            Number of parallel workers
            Default: 4
        save_diag_plots : bool
            Save diagnostic plots
            Default: False
        save_fits : bool
            Save FITS cutouts
            Default: False
        output_folder : str, optional
            Where to save outputs
        output_csv : str, optional
            Path to save results CSV. If None, not saved.
        all_ellipse_sources : bool
            If True, create separate rows for each source detected within the error ellipse.
            If False, only include the forced photometry result. Default: False

        Returns
        -------
        pd.DataFrame
            Results with columns: visit_id, detector, band, ra, dec,
            flux_psf, flux_err_psf, mag_psf, mag_err_psf, snr, ...
        """
        self.logger.info(f"Loading photometry requests from {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Validate required columns
        required = ["visit_id", "detector", "band", "ra", "dec"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Parse requests
        requests = []
        for _, row in df.iterrows():
            # Handle aperture_radii (can be comma-separated string)
            aperture_radii = None
            if "aperture_radii" in row and pd.notna(row["aperture_radii"]):
                aperture_radii = [float(x.strip()) for x in str(row["aperture_radii"]).split(",")]

            request = PhotometryRequest(
                visit_id=int(row["visit_id"]),
                detector=int(row["detector"]),
                band=str(row["band"]),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                error_radius=float(row.get("error_radius", 3.0)),
                detection_threshold=float(row.get("detection_threshold", 5.0)),
                image_type=str(row.get("image_type", "visit_image")),
                aperture_radii=aperture_radii,
                target_name=str(row.get("target_name", f"target_{len(requests)}")),
            )
            requests.append(request)

        # Process requests
        results = self.measure_batch(
            requests=requests,
            parallel=parallel,
            max_workers=max_workers,
            save_diag_plots=save_diag_plots,
            save_fits=save_fits,
            output_folder=output_folder,
        )

        # Convert to DataFrame
        results_df = self._results_to_dataframe(
            results, requests, include_all_ellipse_sources=all_ellipse_sources
        )

        # Save if requested
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            self.logger.info(f"Results saved to {output_csv}")

        return results_df

    def measure_multi_targets_in_image(
        self,
        visit_id: int,
        detector: int,
        band: str,
        coordinates: list[tuple[float, float]],
        target_names: Optional[list[str]] = None,
        error_radius: float = 3.0,
        image_type: str = "visit_image",
        aperture_radii: Optional[list[float]] = None,
        save_diag_plots: bool = False,
        save_fits: bool = False,
        output_folder: Optional[str] = None,
    ) -> dict[str, EndResult]:
        """
        Measure multiple targets in a single image.

        Note: Current implementation still loads image multiple times.
        Future optimization: load image once and reuse.

        Parameters
        ----------
        visit_id : int
            Visit identifier
        detector : int
            Detector ID
        band : str
            Photometric band
        coordinates : list[tuple[float, float]]
            List of (ra, dec) tuples in degrees
        target_names : list[str], optional
            Names for each target. If None, auto-generated.
        error_radius : float
            Search radius in arcseconds
            Default: 3.0
        image_type : str
            "visit_image" or "difference_image"
            Default: "visit_image"
        aperture_radii : list[float], optional
            Aperture radii in arcseconds
        save_diag_plots : bool
            Save diagnostic plots for each target
            Default: False
        save_fits : bool
            Save FITS cutout (only once for the image)
            Default: False
        output_folder : str, optional
            Where to save outputs

        Returns
        -------
        dict[str, EndResult]
            Mapping of target_name → EndResult
        """
        self.logger.info(
            f"Measuring {len(coordinates)} targets in single image: "
            f"visit {visit_id}, detector {detector}, band {band}"
        )

        # Generate target names if not provided
        if target_names is None:
            target_names = [f"target_{i}" for i in range(len(coordinates))]

        if len(target_names) != len(coordinates):
            raise ValueError(
                f"Number of target_names ({len(target_names)}) must match "
                f"coordinates ({len(coordinates)})"
            )

        # Create requests
        requests = [
            PhotometryRequest(
                visit_id=visit_id,
                detector=detector,
                band=band,
                ra=ra,
                dec=dec,
                error_radius=error_radius,
                image_type=image_type,
                aperture_radii=aperture_radii,
                target_name=name,
            )
            for (ra, dec), name in zip(coordinates, target_names)
        ]

        # Process all targets
        # TODO: Optimize to load image once and reuse
        results_list = []
        for i, request in enumerate(requests):
            result = self.measure_single(
                request=request,
                save_diag_plots=save_diag_plots,
                save_fits=save_fits and i == 0,  # Save FITS only once
                output_folder=output_folder,
            )
            results_list.append(result)

        # Return as dictionary
        return {name: result for name, result in zip(target_names, results_list)}

    def _create_image_metadata(self, request: PhotometryRequest) -> ImageMetadata:
        """
        Create synthetic ImageMetadata from PhotometryRequest.

        This creates the ImageMetadata structure expected by PhotometryService,
        but without actual ephemeris data. Instead, uses user-provided coordinates.

        Parameters
        ----------
        request : PhotometryRequest
            User request with coordinates and image specification

        Returns
        -------
        ImageMetadata
            Synthetic metadata suitable for PhotometryService.process_image()
        """
        # Get image time information from Butler
        image_info = self._get_image_time_info(request.visit_id, request.detector, request.image_type)

        # Create synthetic ephemeris data at the user-provided coordinate
        exact_ephemeris = EphemerisDataCompressed(
            datetime=image_info["mid_time"],
            ra_deg=request.ra,
            dec_deg=request.dec,
            ra_rate=0.0,  # Unknown, but not used in standalone mode
            dec_rate=0.0,
            uncertainty={
                "rss": request.error_radius,
                "smaa": request.error_radius,
                "smia": request.error_radius,
                "theta": 0.0,
            },
        )

        # Create ImageMetadata
        metadata = ImageMetadata(
            visit_id=request.visit_id,
            detector_id=request.detector,
            band=request.band,
            coordinates_central=(request.ra, request.dec),
            t_min=image_info["begin_time"],
            t_max=image_info["end_time"],
            ephemeris_data=[exact_ephemeris],
            exact_ephemeris=exact_ephemeris,
        )

        return metadata

    def _get_image_time_info(self, visit_id: int, detector: int, image_type: str) -> dict:
        """
        Get timing information for an image from Butler.

        Parameters
        ----------
        visit_id : int
            Visit identifier
        detector : int
            Detector ID
        image_type : str
            "visit_image" or "difference_image"

        Returns
        -------
        dict
            Dictionary with keys: begin_time, end_time, mid_time (all Time objects)
        """
        try:
            # Get visit info from Butler
            visit_info = self.butler.get(
                "visit_image.visitInfo", visit=visit_id, detector=detector
            )

            # Get mid-exposure time
            t_mid = visit_info.date.toAstropy()

            # Ensure TAI scale (LSST convention)
            if t_mid.scale != "tai":
                t_mid = t_mid.tai

            # Get exposure time and calculate begin/end times
            exp_time = visit_info.exposureTime  # seconds
            t_min = t_mid - (TimeDelta(exp_time, format="sec") / 2)
            t_max = t_mid + (TimeDelta(exp_time, format="sec") / 2)

            return {"begin_time": t_min, "end_time": t_max, "mid_time": t_mid}

        except Exception as e:
            self.logger.error(f"Error getting image time info: {e}")
            raise ValueError(
                f"Failed to retrieve timing information for visit {visit_id}, "
                f"detector {detector}: {str(e)}"
            ) from e

    def _validate_image_exists(
        self, visit_id: int, detector: int, band: str, image_type: str
    ) -> bool:
        """
        Check if specified image exists in Butler.

        Parameters
        ----------
        visit_id : int
            Visit identifier
        detector : int
            Detector ID
        band : str
            Photometric band
        image_type : str
            "visit_image" or "difference_image"

        Returns
        -------
        bool
            True if image exists, False otherwise
        """
        try:
            # Query Butler for image using bind parameters
            refs = list(
                self.butler.query_datasets(
                    image_type,
                    where="visit = :visit AND detector = :detector AND band = :band",
                    bind={"visit": visit_id, "detector": detector, "band": band},
                )
            )
            return len(refs) > 0
        except Exception as e:
            self.logger.error(f"Error validating image: {e}")
            return False

    def _validate_coordinates(self, ra: float, dec: float) -> bool:
        """
        Validate coordinate ranges.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees
        dec : float
            Declination in degrees

        Returns
        -------
        bool
            True if valid, raises ValueError otherwise

        Raises
        ------
        ValueError
            If coordinates are out of valid range
        """
        if not (0 <= ra < 360):
            raise ValueError(f"RA must be in range [0, 360): got {ra}")

        if not (-90 <= dec <= 90):
            raise ValueError(f"Dec must be in range [-90, 90]: got {dec}")

        return True

    def _measure_single_wrapper(
        self,
        request: PhotometryRequest,
        save_diag_plots: bool,
        save_fits: bool,
        output_folder: str,
    ) -> EndResult:
        """Wrapper for parallel execution."""
        return self.measure_single(
            request=request,
            save_diag_plots=save_diag_plots,
            save_fits=save_fits,
            output_folder=output_folder,
        )

    def _results_to_dataframe(
        self,
        results: list[EndResult],
        requests: list[PhotometryRequest],
        include_all_ellipse_sources: bool = False,
    ) -> pd.DataFrame:
        """
        Convert list of EndResult objects to pandas DataFrame.

        Parameters
        ----------
        results : list[EndResult]
            Photometry results
        requests : list[PhotometryRequest]
            Original requests (for metadata)
        include_all_ellipse_sources : bool
            If True, create separate rows for each source detected within the error ellipse.
            If False, only include the forced photometry result. Default: False

        Returns
        -------
        pd.DataFrame
            Flattened results table
        """
        rows = []
        for result, request in zip(results, requests):
            if result is None or result.forced_phot_on_target is None:
                # Failed measurement
                row = {
                    "visit_id": request.visit_id,
                    "detector": request.detector,
                    "band": request.band,
                    "ra": request.ra,
                    "dec": request.dec,
                    "target_name": request.target_name,
                    "success": False,
                }
                rows.append(row)
            else:
                # Base row data from forced photometry result
                phot = result.forced_phot_on_target
                base_row = {
                    "visit_id": result.visit_id,
                    "detector": result.detector_id,
                    "band": result.band,
                    "target_name": result.target_name,
                    "target_ra": request.ra,
                    "target_dec": request.dec,
                    "forced_ra": phot.ra,
                    "forced_dec": phot.dec,
                    "forced_flux": phot.flux,
                    "forced_flux_err": phot.flux_err,
                    "forced_mag": phot.mag,
                    "forced_mag_err": phot.mag_err,
                    "forced_snr": phot.snr,
                    "forced_x": phot.x,
                    "forced_y": phot.y,
                    "success": True,
                    "n_nearby_sources": len(result.phot_within_error_ellipse or []),
                }

                if include_all_ellipse_sources and result.phot_within_error_ellipse:
                    # Create separate row for each source in error ellipse
                    for i, source in enumerate(result.phot_within_error_ellipse):
                        row = base_row.copy()
                        row.update(
                            {
                                "ellipse_source_index": i,
                                "ellipse_source_ra": source.ra,
                                "ellipse_source_dec": source.dec,
                                "ellipse_source_ra_err": source.ra_err,
                                "ellipse_source_dec_err": source.dec_err,
                                "ellipse_source_x": source.x,
                                "ellipse_source_y": source.y,
                                "ellipse_source_x_err": source.x_err,
                                "ellipse_source_y_err": source.y_err,
                                "ellipse_source_snr": source.snr,
                                "ellipse_source_flux": source.flux,
                                "ellipse_source_flux_err": source.flux_err,
                                "ellipse_source_mag": source.mag,
                                "ellipse_source_mag_err": source.mag_err,
                                "ellipse_source_separation": source.separation,
                                "ellipse_source_sigma": source.sigma,
                            }
                        )
                        rows.append(row)
                else:
                    # Standard mode - one row per result
                    rows.append(base_row)

        return pd.DataFrame(rows)
