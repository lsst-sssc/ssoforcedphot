"""
Photometry Service Module

This module provides comprehensive functionality for performing photometry on LSST images,
including source detection, flux measurements, and error analysis. It supports both forced
photometry at exact ephemeris coordinates and detection of sources within specified error ellipses.

Main Components:
    - PhotometryService: Core class for photometry operations
    - Supporting dataclasses for results and parameters (imported from `image_photometry.utils`)
    - Utility functions for coordinate handling, measurements, and visualization
"""

import logging
import os
from typing import Optional

import astropy.units as u
import lsst.afw.display as afwdisplay
import lsst.afw.image as afwimage
import lsst.afw.table as afwtable
import lsst.daf.base as dafbase
import lsst.geom as geom
import numpy as np
from astropy.coordinates import SkyCoord
from image_photometry.utils import (
    EndResult,
    ErrorEllipse,
    ImageMetadata,
    PhotometryResult,
    target_name_maker,
)

# from image_photometry.utils_json import save_results_to_json
from image_photometry.visualization import create_diagnostic_plot
from lsst.daf.butler import Butler
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.base import ForcedMeasurementTask, SingleFrameMeasurementTask


class PhotometryService:
    """
    Service for performing photometry operations on LSST images.

    Provides functionality for:
    - Loading calibrated exposures from an LSST Butler.
    - Performing forced photometry at designated celestial coordinates.
    - Detecting and measuring additional sources within a specified error ellipse.
    - Generating image cutouts centered on the target.
    - Saving diagnostic plots and FITS cutouts for analysis.
    - Preparing structured photometry results.

    Attributes:
        detection_threshold (float): Signal-to-noise threshold used for source detection.
                                     Sources with SNR below this value are typically ignored.
        display (afwdisplay.Display): A display object for visualization, typically Firefly.
                                      Initialized to None and set up when display is requested.
        butler (lsst.daf.butler.Butler): An LSST Data Butler instance configured to access
                                         LSST data repositories (e.g., 'dp1').
        logger (logging.Logger): Logger for reporting status and errors specific to this service.
    """

    def __init__(
        self,
        detection_threshold: float = 5,
        dr: str = "dp1",
        collection: str = "LSSTComCam/DP1",
    ):
        """
        Initialize the PhotometryService.

        Parameters
        ----------
        detection_threshold : float, optional
            Threshold value for source detection (default: 5)
        dr : str, optional
            Parameter for the 'PhotometryService' and 'ImageServiceButler'
            Select the data release, later can be changed
        collection : str, optional
            Parameter for the 'PhotometryService' and 'ImageServiceButler'
            Select the collection later can be changed
        """
        self.logger = logging.getLogger("photometry_service")
        self.display = None
        self.detection_threshold = detection_threshold
        self.butler = Butler(dr, collections=collection)

    def process_image(
        self,
        image_metadata: ImageMetadata,
        target_name: str,
        target_type: str,
        image_type: str,
        ephemeris_service: str,
        cutout_size: int = 800,
        save_diag_plots: bool = False,
        save_fits: bool = False,
        override_error: float = 0,
        display: bool = True,
        output_folder: Optional[str] = None,
        save_json: bool = False,
        json_filename: Optional[str] = None,
        save_csv: bool = False,
    ) -> EndResult:
        """
        Process a single image by performing forced photometry at the ephemeris
        coordinates and detecting additional sources within an error ellipse.

        This method orchestrates:
        1. Loading the correct image exposure (visit_image or differenceExp) using the Butler.
        2. Constructing an error ellipse based on ephemeris uncertainty or a user-defined override.
        3. Generating a base filename for saved outputs.
        4. Calling `perform_photometry` to do the core measurement.
        5. Structuring the results into an `EndResult` dataclass.

        Parameters
        ----------
        image_metadata : ImageMetadata
            Complete metadata for the image, including visit, detector, band, and ephemeris data.
        target_name : str
            The name of the astronomical target (e.g., "C/2020 F3 (NEOWISE)").
        target_type : str
            The classification of the target (e.g., "COMET", "ASTEROID").
        image_type : str
            The type of image to retrieve and process: "visit_image" for raw visit data
            or "difference_image" for difference images.
        ephemeris_service : str
            The name of the ephemeris service used to obtain the ephemeris data
            (e.g., "Horizons", "Miriade").
        cutout_size : int, optional
            The desired size in pixels for the square image cutout centered on the target.
            Defaults to 800.
        save_diag_plots : bool, optional
            If True, a diagnostic PNG plot showing the image, target, and detected sources
            within the error ellipse will be saved. Defaults to False.
        save_fits : bool, optional
            If True, the FITS cutout of the image will be saved. Defaults to False.
        override_error : float, optional
            If greater than 0, this value in arcseconds will be used as the radius
            for a circular error ellipse, overriding the ephemeris's reported uncertainty.
            Defaults to 0 (no override).
        display : bool, optional
            If True, attempts to display the image and photometry results in Firefly
            for real-time visualization. Defaults to True.
        output_folder : str, optional
            The path to the directory where diagnostic plots and FITS cutouts will be saved.
            If None, and saving is requested, a warning will be logged. Defaults to None.
        save_json : bool, optional
            Save the final photometry results as JSON (default: False)
        json_filename : str, optional
            Name of the json file.
        save_csv : bool, optional
            Save the final photometry results as csv (default: False)

        Returns
        -------
        EndResult
            A dataclass containing all compiled results of the photometry processing
            for the given image, including forced photometry on the target and
            measurements for sources within the error ellipse.

        Raises
        ------
        ValueError
            If `image_metadata` is None.
        """

        # Check if image_metadata is valid
        if image_metadata is None:
            raise ValueError("image_metadata is None")

        # Get the exposure
        if image_type == "difference_image":
            visit_image = self.butler.get(
                "difference_image",
                dataId={"visit": image_metadata.visit_id, "detector": image_metadata.detector_id},
            )
        else:
            visit_image = self.butler.get(
                "visit_image",
                dataId={"visit": image_metadata.visit_id, "detector": image_metadata.detector_id},
            )

        if override_error > 0:
            print(f"Override the error ellipse with the value of {override_error} arcsec")
            error_ellipse = ErrorEllipse(
                smaa_3sig=override_error,
                smia_3sig=override_error,
                theta=0,
                center_coord=(image_metadata.exact_ephemeris.ra_deg, image_metadata.exact_ephemeris.dec_deg),
            )
        else:
            # Create error ellipse from ephemeris data
            if image_metadata.exact_ephemeris.uncertainty["smaa"] > 0:
                error_ellipse = ErrorEllipse(
                    smaa_3sig=image_metadata.exact_ephemeris.uncertainty["smaa"],
                    smia_3sig=image_metadata.exact_ephemeris.uncertainty["smia"],
                    theta=image_metadata.exact_ephemeris.uncertainty["theta"],
                    center_coord=(
                        image_metadata.exact_ephemeris.ra_deg,
                        image_metadata.exact_ephemeris.dec_deg,
                    ),
                )
            else:
                print("No error ellipse data found, using RSS instead")
                error_ellipse = ErrorEllipse(
                    smaa_3sig=image_metadata.exact_ephemeris.uncertainty["rss"],
                    smia_3sig=image_metadata.exact_ephemeris.uncertainty["rss"],
                    theta=0,
                    center_coord=(
                        image_metadata.exact_ephemeris.ra_deg,
                        image_metadata.exact_ephemeris.dec_deg,
                    ),
                )

        # Create base image name if saving output
        base_image_name = ""
        if (save_diag_plots or save_fits) and output_folder:
            # target_name_modified = target_name.replace(":", "-").replace(" ", "_").replace("/", "_")
            target_name_modified = target_name_maker(target_name)

            base_image_name = (
                f"{target_name_modified}_visit{image_metadata.visit_id}_"
                f"detector{image_metadata.detector_id}_band{image_metadata.band}"
            )
            print(f"Base image name: {base_image_name}")

        # Perform photometry
        # self.logger.info(f"perform photometry, error_ellipse: {error_ellipse}")
        target_result, sources_within_error = self.perform_photometry(
            visit_image=visit_image,
            ra_deg=image_metadata.exact_ephemeris.ra_deg,
            dec_deg=image_metadata.exact_ephemeris.dec_deg,
            cutout_size=cutout_size,
            display=display,
            error_ellipse=error_ellipse,
            save_diag_plots=save_diag_plots,
            save_fits=save_fits,
            output_folder=output_folder,
            image_name=base_image_name,
            image_metadata_for_plot=image_metadata,
            target_name_for_plot=target_name,
        )

        # Prepare end result
        saved_image_name_for_result = ""
        if save_diag_plots and output_folder and base_image_name:
            saved_image_name_for_result = base_image_name

        end_result = self._prepare_end_results(
            target_name=target_name,
            target_type=target_type,
            image_type=image_type,
            ephemeris_service=ephemeris_service,
            image_metadata=image_metadata,
            cutout_size=cutout_size,
            saved_image_name=saved_image_name_for_result,
            target_result=target_result,
            sources_within_error=sources_within_error,
        )

        return end_result

    def perform_photometry(
        self,
        visit_image,
        ra_deg,
        dec_deg,
        cutout_size=400,
        display=True,
        psf_only=True,
        find_sources_flag=True,
        save_diag_plots=False,
        save_fits=False,
        output_folder=None,
        image_name: Optional[str] = None,
        error_ellipse: Optional[ErrorEllipse] = None,
        image_metadata_for_plot: Optional[ImageMetadata] = None,
        target_name_for_plot: Optional[str] = None,
    ):
        """
        Performs core photometry operations: creating a cutout, identifying sources,
        and measuring their fluxes.

        This method encapsulates the lower-level LSST Science Pipelines tasks for:
        1. Creating an image cutout centered on the target RA/Dec.
        2. Optionally detecting nearby sources within the cutout (and within an error ellipse if provided).
        3. Performing forced photometry on the target coordinates and any detected nearby sources.
        4. Handling display in Firefly and saving FITS cutouts and diagnostic PNG plots.

        Parameters
        ----------
        visit_image : lsst.afw.image.ExposureF
            The full calibrated exposure from which a cutout will be made.
        ra_deg : float
            The Right Ascension (in degrees) of the target for photometry.
        dec_deg : float
            The Declination (in degrees) of the target for photometry.
        cutout_size : int, optional
            The size of the square cutout in pixels (e.g., 400 for 400x400). Defaults to 400.
        display : bool, optional
            If True, the cutout image will be displayed in Firefly, along with marked sources
            and the error ellipse. Defaults to True.
        psf_only : bool, optional
            If True, only performs PSF (Point Spread Function) photometry. If False,
            additional measurement plugins (GaussianFlux, SdssShape, CircularApertureFlux)
            would be included (though currently not fully implemented in the provided code).
            Defaults to True.
        find_sources_flag : bool, optional
            If True, the method will attempt to find additional sources around the target
            within the cutout, which are then also measured. Defaults to True.
        save_diag_plots : bool, optional
            If True, a diagnostic PNG plot is generated and saved. Defaults to False.
        save_fits : bool, optional
            If True, the FITS cutout image is saved. Defaults to False.
        output_folder : str, optional
            The directory path where any saved FITS cutouts or diagnostic plots will be stored.
            Required if `save_fits` or `save_diag_plots` is True. Defaults to None.
        image_name : str, optional
            A base name for the output image files (FITS, PNG). If None, a default name
            will be constructed based on image metadata when saving. Defaults to None.
        error_ellipse : ErrorEllipse, optional
            An `ErrorEllipse` object defining the 3-sigma error region around the target.
            Used for filtering detected sources and for plotting. Defaults to None.
        image_metadata_for_plot : ImageMetadata, optional
            The full `ImageMetadata` object, used to populate plot titles and other
            diagnostic plot elements. Defaults to None.
        target_name_for_plot : Optional[str], optional
            The name of the target, specifically for display in plot titles. Defaults to None.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - PhotometryResult: The photometry result for the target coordinates.
            - list[PhotometryResult]: A list of results for other sources detected within the error ellipse.
        """
        # Get WCS and prepare coordinates
        target_img, bbox, offsets = self._prepare_image(visit_image, ra_deg, dec_deg, cutout_size)
        x_offset, y_offset = offsets

        if target_img is None:
            return None, None

        # Initialize coordinate lists
        ra_list, dec_list, found_sources = self._initialize_coordinates(
            target_img, ra_deg, dec_deg, find_sources_flag, error_ellipse
        )

        print(f"Found {len(ra_list)} total coordinates for forced photometry (target + nearby)")

        # Setup and perform measurements
        forced_meas_cat = self._perform_forced_photometry(
            target_img, ra_list, dec_list, x_offset, y_offset, psf_only
        )

        # Prepare PhotometryResult dataclasses for the target and any other found sources
        target_phot_result, sources_phot_results_list = self._prepare_photometry_results(
            forced_meas_cat=forced_meas_cat, ra=ra_deg, dec=dec_deg, found_sources=found_sources
        )

        # Handle display and visualization (including saving diagnostic plot)
        if display or save_fits:
            # wcs = target_img.getWcs() # WCS needed for both display and plot

            # Firefly display logic (existing)
            if display:
                afwdisplay.setDefaultBackend("firefly")
                afw_display = afwdisplay.Display(frame=1)
                afw_display.mtv(target_img)
                afw_display.setMaskTransparency(100)

            if save_fits and output_folder:
                # Save fits
                fits_filename = image_name + ".fits"
                output_filepath_fits = os.path.join(output_folder, fits_filename)
                try:
                    target_img.writeFits(output_filepath_fits)
                    print(f"Fits saved successfully to: {output_filepath_fits}")
                except Exception as e:
                    self.logger.error(f"Failed to save FITS file {output_filepath_fits}: {e}", exc_info=True)

        # Save diagnostic plot if requested
        # image_name here is the base name like "diag_plot_visitX_detY_bandZ"
        if save_diag_plots and output_folder and image_name and image_metadata_for_plot:
            png_filename = image_name + ".png"
            output_filepath_png = os.path.join(output_folder, png_filename)

            # Prepare data for create_diagnostic_plot
            plot_target_skycoord = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg")

            nearby_skycoords_for_plot = []
            if sources_phot_results_list:
                for src_result in sources_phot_results_list:
                    nearby_skycoords_for_plot.append(
                        SkyCoord(ra=src_result.ra, dec=src_result.dec, unit="deg")
                    )

            plot_title = (
                f"Target: {target_name_for_plot or 'N/A'} | Visit: {image_metadata_for_plot.visit_id} "
                f"| Det: {image_metadata_for_plot.detector_id} | Band: {image_metadata_for_plot.band}"
            )

            try:
                # self.logger.info(f"Attempting to create diagnostic plot: {output_filepath_png}")
                # error_ellipse object is passed directly to perform_photometry
                create_diagnostic_plot(
                    image_exposure=target_img,
                    target_skycoord=plot_target_skycoord,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    error_ellipse_obj=error_ellipse,
                    nearby_skycoords=nearby_skycoords_for_plot,
                    output_filepath=output_filepath_png,
                    title=plot_title,
                )
                self.logger.info(f"Diagnostic plot saved successfully to: {output_filepath_png}")
            except Exception as e:
                self.logger.error(f"Failed to create diagnostic plot: {e}", exc_info=True)
        elif save_diag_plots:
            missing_info = []
            if not output_folder:
                missing_info.append("output_folder")
            if not image_name:
                missing_info.append("image_name (base)")
            if not image_metadata_for_plot:
                missing_info.append("image_metadata_for_plot")
            self.logger.warning(
                f"""Skipping diagnostic plot generation because some required information is
                missing: {', '.join(missing_info)}."""
            )

        # Return the results from _prepare_photometry_results
        return target_phot_result, sources_phot_results_list

    def _calculate_separations(
        self, table, ra_coord: float, dec_coord: float, sigma3: float
    ) -> afwtable.SourceTable:
        """
        Calculate angular separations between a given target coordinate and all sources
        within an `afwtable.SourceTable`. Adds "separation" and "sigma" columns to the table.
        The "sigma" column represents the separation normalized by a `sigma3` factor.

        Parameters
        ----------
        table : lsst.afw.table.SourceTable
            A source table containing 'coord_ra' and 'coord_dec' columns (in radians).
        ra_coord : float
            The Right Ascension (in degrees) of the target.
        dec_coord : float
            The Declination (in degrees) of the target.
        sigma3 : float
            The 3-sigma semi-major axis of the error ellipse in arcseconds, used for
            normalizing the separation to calculate the 'sigma' value.

        Returns
        -------
        lsst.afw.table.SourceTable
            The original source table with two new columns added:
            - "separation": Angular separation from the target in arcseconds.
            - "sigma": The separation divided by `sigma3` and multiplied by 3
                       (representing how many "3-sigma" units away the source is).
        """
        target_coord = geom.SpherePoint(np.radians(ra_coord), np.radians(dec_coord), geom.radians)
        separations = [
            target_coord.separation(geom.SpherePoint(ra, dec, geom.radians)).asArcseconds()
            for ra, dec in zip(table["coord_ra"], table["coord_dec"])
        ]

        # Add separation column to table
        table["separation"] = separations
        table["sigma"] = [(sep / sigma3) * 3 for sep in separations]

        return table

    def find_measure_sources(
        self, visit_image, ra_coord, dec_coord, error_ellipse: Optional[ErrorEllipse] = None
    ):
        """
        Detects sources in an image and performs single-frame measurements on them.
        Optionally filters these detected sources to include only those within a
        specified error ellipse and sorts them by 'sigma' (normalized separation).

        Parameters
        ----------
        visit_image : lsst.afw.image.ExposureF
            The calibrated exposure in which to detect and measure sources.
        ra_coord : float
            The Right Ascension (in degrees) of the target, used for calculating
            separations and filtering by error ellipse.
        dec_coord : float
            The Declination (in degrees) of the target, used for calculating
            separations and filtering by error ellipse.
        error_ellipse : ErrorEllipse, optional
            An `ErrorEllipse` object. If provided, detected sources will be filtered
            to only include those geometrically inside this ellipse. If None,
            only the single closest detected source will be returned. Defaults to None.

        Returns
        -------
        astropy.table.Table
            An Astropy `Table` containing the detected and measured sources.
            - If `error_ellipse` is provided, it contains sources within the ellipse,
              sorted by 'sigma'.
            - If `error_ellipse` is None, it contains only the single closest detected source.
            The table includes 'separation' and 'sigma' columns.
        """
        print("Starting source detection and measurement")

        # Setup detection and measurement
        schema = afwtable.SourceTable.makeMinimalSchema()
        schema.addField("coord_raErr", type="F")
        schema.addField("coord_decErr", type="F")

        config = SourceDetectionTask.ConfigClass()
        config.thresholdValue = self.detection_threshold
        config.thresholdType = "stdev"

        sourcedetectiontask = SourceDetectionTask(schema=schema, config=config)
        sourcemeasurementtask = SingleFrameMeasurementTask(
            schema=schema, config=SingleFrameMeasurementTask.ConfigClass(), algMetadata=dafbase.PropertyList()
        )

        # Run detection and measurement
        tab = afwtable.SourceTable.make(schema)
        result = sourcedetectiontask.run(tab, visit_image)
        sources = result.sources
        sourcemeasurementtask.run(measCat=sources, exposure=visit_image)

        # Get all sources and add separations
        sources_copy = sources.copy(True)
        table = sources_copy.asAstropy()
        table_with_sep = self._calculate_separations(table, ra_coord, dec_coord, error_ellipse.smaa_3sig)

        if error_ellipse is None:
            # If no error ellipse, return closest source
            min_index = np.argmin(table_with_sep["separation"])
            return table_with_sep[min_index : min_index + 1]
        else:
            # Filter sources within error ellipse
            mask = []
            for row in table_with_sep:
                ra = np.rad2deg(row["coord_ra"])
                dec = np.rad2deg(row["coord_dec"])
                mask.append(error_ellipse.is_point_inside(ra, dec))

            filtered_table = table_with_sep[mask]
            filtered_table.sort("sigma")

            return filtered_table

    def _perform_forced_photometry(self, target_img, ra_list, dec_list, x_offset, y_offset, psf_only):
        """
        Sets up and executes forced photometry on a list of specified celestial coordinates
        within a given image exposure.

        Forced photometry measures the flux at pre-defined locations, useful for
        tracking objects even if they are too faint to be detected.

        Parameters
        ----------
        target_img : lsst.afw.image.ExposureF
            The calibrated image cutout on which to perform forced photometry.
        ra_list : list[float]
            A list of Right Ascensions (in degrees) for each point where photometry
            should be performed.
        dec_list : list[float]
            A list of Declinations (in degrees) corresponding to `ra_list`.
        x_offset : int
            The X-coordinate offset of the `target_img`'s origin relative to the
            original full image. Used internally by LSST tasks, though not directly
            by this method for `geom.SpherePoint`.
        y_offset : int
            The Y-coordinate offset of the `target_img`'s origin relative to the
            original full image.
        psf_only : bool
            If True, only PSF-based flux measurements (`base_PsfFlux`) are performed.
            If False, additional plugins could be included (though currently not in this code).

        Returns
        -------
        lsst.afw.table.SourceCatalog
            A `SourceCatalog` containing the forced photometry measurements for
            each input coordinate in `ra_list` and `dec_list`. Each record in the
            catalog corresponds to one input coordinate.
        """
        # Create schema and configure measurement
        schema = afwtable.SourceTable.makeMinimalSchema()
        schema.addField("centroid_x", type="D")
        schema.addField("centroid_y", type="D")
        schema.addField("shape_xx", type="D")
        schema.addField("shape_yy", type="D")
        schema.addField("shape_xy", type="D")
        schema.addField("type_flag", type="F")

        alias_map = schema.getAliasMap()
        alias_map.set("slot_Centroid", "centroid")
        alias_map.set("slot_Shape", "shape")

        config = ForcedMeasurementTask.ConfigClass()
        config.copyColumns = {}
        config.plugins.names = ["base_TransformedCentroid", "base_PsfFlux", "base_TransformedShape"]

        # Not implemented...
        if not psf_only:
            config.plugins.names.extend(["base_GaussianFlux", "base_SdssShape", "base_CircularApertureFlux"])

        config.doReplaceWithNoise = False

        # Create measurement task and source catalog
        forced_measurement_task = ForcedMeasurementTask(schema, config=config)
        forced_source = afwtable.SourceCatalog(schema)

        # Add sources and perform measurements
        wcs = target_img.getWcs()
        for ra, dec in zip(ra_list, dec_list):
            source_rec = forced_source.addNew()
            coord = geom.SpherePoint(np.radians(ra), np.radians(dec), geom.radians)
            source_rec.setCoord(coord)

            pixel_coord = wcs.skyToPixel(coord)
            source_rec["centroid_x"] = pixel_coord.getX()
            source_rec["centroid_y"] = pixel_coord.getY()
            source_rec["type_flag"] = 0

        # Run forced photometry
        forced_meas_cat = forced_measurement_task.generateMeasCat(
            target_img, forced_source, target_img.getWcs()
        )
        forced_measurement_task.run(forced_meas_cat, target_img, forced_source, target_img.getWcs())

        return forced_meas_cat

    def _prepare_image(self, visit_image, ra_deg, dec_deg, cutout_size):
        """
        Prepares the image for photometry by creating a cutout centered around
        the specified celestial coordinates. Handles cases where the target is
        too close to the image edge or the cutout size is invalid.

        Parameters
        ----------
        visit_image : lsst.afw.image.ExposureF
            The full calibrated LSST exposure from which to create a cutout.
        ra_deg : float
            The Right Ascension (in degrees) of the center of the desired cutout.
        dec_deg : float
            The Declination (in degrees) of the center of the desired cutout.
        cutout_size : int
            The desired size in pixels for the square cutout (e.g., 800 for 800x800).

        Returns
        -------
        tuple
            A tuple (target_img, bbox, offsets) where:
            - target_img (lsst.afw.image.ExposureF or None): The cutout image exposure,
              or None if the target is too close to the edge. If `cutout_size` is 0
              or the cutout would extend outside the image, the full `visit_image`
              might be returned.
            - bbox (lsst.geom.Box2I or None): The bounding box of the cutout within
              the original image coordinates, or None if the full image is used or
              cutout creation fails.
            - offsets (tuple[float, float]): A tuple `(x_offset, y_offset)` representing
              the minimum X and Y pixel coordinates of the cutout's origin relative
              to the original `visit_image`. Returns `(0, 0)` if no cutout is made.
        """
        wcs = visit_image.getWcs()
        coord = geom.SpherePoint(np.radians(ra_deg), np.radians(dec_deg), geom.radians)

        pixel_coord = wcs.skyToPixel(coord)
        x_center, y_center = pixel_coord.getX(), pixel_coord.getY()
        half_size = cutout_size // 2

        min_x, max_x = 0, visit_image.getWidth()
        min_y, max_y = 0, visit_image.getHeight()

        # Check if target is within 10 pixels of any edge
        if x_center < 10 or x_center > (max_x - 10) or y_center < 10 or y_center > (max_y - 10):
            print("Target is within 10 pixels of image edge or outside of the boundaries. Skipping image.")
            return None, None, (0, 0)

        if cutout_size <= 0:
            print("Using the complete image.")
            return visit_image, None, (0, 0)

        elif (
            (x_center - half_size) < min_x
            or (x_center + half_size) > max_x
            or (y_center - half_size) < min_y
            or (y_center + half_size) > max_y
        ):
            print("The cutout boundaries are outside of the image.")
            print("Using the complete image.")
            return visit_image, None, (0, 0)

        print(f"Creating cutout with size {cutout_size} pixels")

        # Create bounding box for cutout
        bbox = geom.Box2I()
        bbox.include(geom.Point2I(float(x_center - half_size), float(y_center - half_size)))
        bbox.include(geom.Point2I(float(x_center + half_size), float(y_center + half_size)))

        # Create cutout with PARENT origin to preserve WCS
        # Using PARENT keeps the original pixel coordinate system and WCS intact
        target_img = visit_image.Factory(visit_image, bbox, origin=afwimage.PARENT, deep=False)

        return target_img, bbox, (bbox.getMinX(), bbox.getMinY())

    def _initialize_coordinates(
        self, target_img, ra_deg, dec_deg, find_sources_flag, error_ellipse: Optional[ErrorEllipse] = None
    ):
        """
        Initializes the list of coordinates for which forced photometry will be performed.
        This list always includes the target's coordinates. If `find_sources_flag` is True,
        it also includes coordinates of additional sources detected within the image,
        optionally filtered by an `error_ellipse`.

        Parameters
        ----------
        target_img : lsst.afw.image.ExposureF
            The image exposure (typically a cutout) in which to find additional sources.
        ra_deg : float
            The Right Ascension (in degrees) of the primary target.
        dec_deg : float
            The Declination (in degrees) of the primary target.
        find_sources_flag : bool
            If True, the `find_measure_sources` method will be called to detect
            and measure additional sources in `target_img`.
        error_ellipse : ErrorEllipse, optional
            An `ErrorEllipse` object used to filter `found_sources`. Only sources
            geometrically inside this ellipse will be added to the list. Defaults to None.

        Returns
        -------
        tuple[list[float], list[float], afwtable.SourceTable]
            A tuple containing:
            - ra_list (list[float]): List of RA coordinates (in degrees) for forced photometry.
                                     Starts with `ra_deg`.
            - dec_list (list[float]): List of Dec coordinates (in degrees) for forced photometry.
                                      Starts with `dec_deg`.
            - found_sources (afwtable.SourceTable): An Astropy table of the additional
                                                    sources found (and measured) by `find_measure_sources`,
                                                    filtered by `error_ellipse` if provided.
                                                    This table contains the raw source measurements.
        """
        ra_list = [ra_deg]
        dec_list = [dec_deg]

        if find_sources_flag:
            sources = self.find_measure_sources(target_img, ra_deg, dec_deg, error_ellipse)
            for source in sources:
                ra_list.append(np.rad2deg(source["coord_ra"]))
                dec_list.append(np.rad2deg(source["coord_dec"]))

        return ra_list, dec_list, sources

    def _prepare_photometry_results(self, forced_meas_cat, ra, dec, found_sources):
        """
        Prepares structured `PhotometryResult` dataclasses from the raw LSST
        measurement catalog for the target and any additional sources found
        within the error ellipse.

        This method extracts relevant photometry information (RA, Dec, flux, SNR, mag, errors)
        from the `forced_meas_cat` and the `found_sources` table, handling potential
        missing values or division by zero for error calculations.

        Parameters
        ----------
        forced_meas_cat : lsst.afw.table.SourceCatalog
            The `SourceCatalog` containing the results of forced photometry for
            all requested coordinates (target + nearby sources). The first entry
            is expected to correspond to the primary target.
        ra : float
            The Right Ascension (in degrees) of the primary target.
        dec : float
            The Declination (in degrees) of the primary target.
        found_sources : astropy.table.Table
            An Astropy `Table` containing the detected and measured sources
            (likely filtered by an error ellipse) that were also included in
            `forced_meas_cat` (from index 1 onwards). Can be None or empty if no
            additional sources were found.

        Returns
        -------
        tuple[PhotometryResult, list[PhotometryResult]]
            A tuple containing:
            - target_result (PhotometryResult): The detailed photometry results
                                               for the primary target coordinates.
            - sources_phot_results_list (list[PhotometryResult]): A list of
                                                                 `PhotometryResult`
                                                                 objects for
                                                                 the additional
                                                                 sources detected
                                                                 within the error
                                                                 ellipse.
        """

        # Prepare forced photometry results for the coordinates
        target_result = PhotometryResult(
            ra=ra,
            dec=dec,
            ra_err=0,
            dec_err=0,
            x=forced_meas_cat[0].get("slot_Centroid_x"),
            y=forced_meas_cat[0].get("slot_Centroid_y"),
            x_err=0,
            y_err=0,
            snr=(
                forced_meas_cat[0].get("base_PsfFlux_instFlux")
                / forced_meas_cat[0].get("base_PsfFlux_instFluxErr")
                if (len(forced_meas_cat) > 0 and forced_meas_cat[0].get("base_PsfFlux_instFluxErr") > 0)
                else 0
            ),
            flux=(
                forced_meas_cat[0].get("base_PsfFlux_instFlux")
                if (
                    forced_meas_cat
                    and len(forced_meas_cat) > 0
                    and not np.isnan(forced_meas_cat[0].get("base_PsfFlux_instFlux"))
                )
                else 0
            ),
            flux_err=(
                forced_meas_cat[0].get("base_PsfFlux_instFluxErr")
                if (
                    forced_meas_cat
                    and len(forced_meas_cat) > 0
                    and forced_meas_cat[0].get("base_PsfFlux_instFluxErr") > 0
                )
                else 0
            ),
            mag=(
                # -2.5 * np.log10(forced_meas_cat[0].get("base_PsfFlux_instFlux")) + 31.4
                (forced_meas_cat[0].get("base_PsfFlux_instFlux") * u.nJy).to(u.ABmag).value
                if (
                    forced_meas_cat
                    and len(forced_meas_cat) > 0
                    and forced_meas_cat[0].get("base_PsfFlux_instFlux") > 0
                )
                else 0
            ),
            mag_err=(
                2.5
                / np.log(10)
                * forced_meas_cat[0].get("base_PsfFlux_instFluxErr")
                / forced_meas_cat[0].get("base_PsfFlux_instFlux")
                if (
                    forced_meas_cat
                    and len(forced_meas_cat) > 0
                    and forced_meas_cat[0].get("base_PsfFlux_instFlux") > 0
                    and forced_meas_cat[0].get("base_PsfFlux_instFluxErr") > 0
                )
                else 0
            ),
            separation=0,
            sigma=0,
            flags={},
        )

        # Prepare results for additional sources (originally from found_sources table)
        sources_phot_results_list = []
        if found_sources is not None and len(found_sources) > 0:
            for i, source_row in enumerate(found_sources):
                fm_idx = i + 1
                if fm_idx < len(forced_meas_cat):
                    meas_record = forced_meas_cat[fm_idx]

                    src_ra_deg = np.degrees(source_row["coord_ra"])
                    src_dec_deg = np.degrees(source_row["coord_dec"])

                    source_result = PhotometryResult(
                        ra=src_ra_deg,
                        dec=src_dec_deg,
                        ra_err=(
                            float(source_row.get("coord_raErr", 0))
                            if float(source_row.get("coord_raErr", 0)) > 0
                            else 0
                        ),
                        dec_err=(
                            float(source_row.get("coord_decErr", 0))
                            if float(source_row.get("coord_decErr", 0)) > 0
                            else 0
                        ),
                        x=meas_record.get("slot_Centroid_x"),
                        y=meas_record.get("slot_Centroid_y"),
                        x_err=0,
                        y_err=0,
                        snr=(
                            meas_record.get("base_PsfFlux_instFlux")
                            / meas_record.get("base_PsfFlux_instFluxErr")
                            if meas_record.get("base_PsfFlux_instFluxErr") > 0
                            else 0
                        ),
                        flux=(
                            meas_record.get("base_PsfFlux_instFlux")
                            if not np.isnan(meas_record.get("base_PsfFlux_instFlux"))
                            else 0
                        ),
                        flux_err=(
                            meas_record.get("base_PsfFlux_instFluxErr")
                            if meas_record.get("base_PsfFlux_instFluxErr") > 0
                            else 0
                        ),
                        mag=(
                            # -2.5 * np.log10(meas_record.get("base_PsfFlux_instFlux")) + 31.4
                            (meas_record.get("base_PsfFlux_instFlux") * u.nJy).to(u.ABmag).value
                            if meas_record.get("base_PsfFlux_instFlux") > 0
                            else 0
                        ),
                        mag_err=(
                            2.5
                            / np.log(10)
                            * meas_record.get("base_PsfFlux_instFluxErr")
                            / meas_record.get("base_PsfFlux_instFlux")
                            if (
                                meas_record.get("base_PsfFlux_instFlux") > 0
                                and meas_record.get("base_PsfFlux_instFluxErr") > 0
                            )
                            else 0
                        ),
                        separation=source_row.get("separation", 0),
                        sigma=source_row.get("sigma", 0),
                        flags={
                            # "base_PsfFlux_flag_edge": meas_record.get("base_PsfFlux_flag_edge", False)
                        },
                    )
                    sources_phot_results_list.append(source_result)
                else:
                    self.logger.warning(
                        f"Index {fm_idx} for a source from 'found_sources' is out of bounds "
                        f"for 'forced_meas_cat' (length {len(forced_meas_cat)}). "
                        "This suggests a mismatch in the number of sources processed."
                    )

        return target_result, sources_phot_results_list

    def _prepare_end_results(
        self,
        target_name,
        target_type,
        image_type,
        ephemeris_service,
        image_metadata,
        cutout_size,
        saved_image_name,
        target_result,
        sources_within_error,
    ):
        """
        Consolidates all collected information into a comprehensive `EndResult` dataclass.
        This provides a structured summary of the photometry processing for a single image.

        Parameters
        ----------
        target_name : str
            The name of the astronomical target.
        target_type : str
            The classification of the target (e.g., "COMET").
        image_type : str
            The type of image processed ("visit_image" or "difference_image").
        ephemeris_service : str
            The name of the ephemeris service used (e.g., "Horizons").
        image_metadata : ImageMetadata
            The full metadata object for the image that was processed.
        cutout_size : int
            The size in pixels of the image cutout used for photometry.
        saved_image_name : str
            The filename of the saved diagnostic plot or FITS cutout (if any),
            or an empty string if nothing was saved.
        target_result : PhotometryResult
            The detailed photometry result for the primary target coordinates.
        sources_within_error : list[PhotometryResult]
            A list of `PhotometryResult` objects for all other sources detected
            and measured within the target's error ellipse.

        Returns
        -------
        EndResult
            A fully populated `EndResult` dataclass summarizing the photometry
            outcome for one image.
        """
        end_result = EndResult(
            target_name=target_name,
            target_type=target_type,
            image_type=image_type,
            ephemeris_service=ephemeris_service,
            visit_id=image_metadata.visit_id,
            detector_id=image_metadata.detector_id,
            band=image_metadata.band,
            coordinates_central=image_metadata.coordinates_central,
            obs_time=image_metadata.t_min,
            cutout_size=cutout_size,
            saved_image_name=saved_image_name,
            uncertainty=image_metadata.exact_ephemeris.uncertainty,
            forced_phot_on_target=target_result,
            phot_within_error_ellipse=sources_within_error,
        )

        return end_result
