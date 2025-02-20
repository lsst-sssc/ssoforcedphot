"""
Photometry Service Module

This module provides comprehensive functionality for performing photometry on LSST images,
including source detection, flux measurements, and error analysis. It supports both forced
photometry and source detection within specified error ellipses.

Main Components:
    - PhotometryService: Core class for photometry operations
    - Supporting dataclasses for results and parameters
    - Utility functions for coordinate handling and measurements
"""

import logging
import os
from typing import Optional

import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.geom as geom
import numpy as np
from forcedphot.image_photometry.utils import EndResult, ErrorEllipse, ImageMetadata, PhotometryResult
from forcedphot.image_photometry.utils_json import save_results_to_json
from lsst.daf.butler import Butler
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.base import ForcedMeasurementTask, SingleFrameMeasurementTask


class PhotometryService:
    """
    Service for performing photometry operations on LSST images.

    Provides functionality for:
    - Loading and handling calibrated exposures
    - Forced photometry at specified coordinates
    - Source detection within error ellipses
    - Image cutout generation and analysis
    - Measurement and visualization of detected sources

    Attributes:
        detection_threshold (float): Signal-to-noise threshold for source detection
        display: Display object for visualization
        butler: Data butler for accessing LSST data repository
    """

    def __init__(self, detection_threshold: float = 5):
        """
        Initialize the PhotometryService.

        Parameters
        ----------
        detection_threshold : float, optional
            Threshold value for source detection (default: 5)
        """
        self.logger = logging.getLogger("photometry_service")
        self.display = None
        self.detection_threshold = detection_threshold
        self.butler = Butler('dp02', collections='2.2i/runs/DP0.2')


    def process_image(self, image_metadata: ImageMetadata, target_name: str,
                     target_type: str, image_type: str,
                     ephemeris_service: str,
                     cutout_size: int = 800, save_cutout: bool = False,
                     display: bool = True,
                     output_dir: Optional[str] = None,
                     save_json: bool = False,
                     json_filename: Optional[str] = None) -> EndResult:
        """
        Process an image using provided metadata and parameters.

        Parameters
        ----------
        image_metadata : ImageMetadata
            Complete metadata for the image
        target_name : str
            Name of the astronomical target
        target_type : str
            Classification of the target
        image_type : str
            Type of image (calexp or goodSeeingDiff_differenceExp)
        ephemeris_service : str
            Service used for ephemeris data
        cutout_size : int, optional
            Size of image cutout in pixels, by default 800
        save_cutout : bool, optional
            Whether to save the cutout image, by default False
        display : bool, optional
            Whether to display results, by default True
        output_dir : str, optional
            Directory for output files, by default None
        save_json : bool, optional
            Whether to save results as JSON, by default False
        json_filename : str, optional
            Name for JSON output file, by default None

        Returns
        -------
        EndResult
            Complete results of photometry processing
        """

        # Check if image_metadata is valid
        if image_metadata is None:
            raise ValueError("image_metadata is None")


        # Get the exposure
        if image_type == "goodSeeingDiff_differenceExp":
            calexp = self.butler.get("goodSeeingDiff_differenceExp", dataId={
                'visit': image_metadata.visit_id,
                'detector': image_metadata.detector_id
            })
        else:
            calexp = self.butler.get("calexp", dataId={
                'visit': image_metadata.visit_id,
                'detector': image_metadata.detector_id
            })

        # Create error ellipse from ephemeris data
        if image_metadata.exact_ephemeris.uncertainty['smaa'] > 0:
            error_ellipse = ErrorEllipse(
                smaa_3sig=image_metadata.exact_ephemeris.uncertainty['smaa'],
                smia_3sig=image_metadata.exact_ephemeris.uncertainty['smia'],
                theta=image_metadata.exact_ephemeris.uncertainty['theta'],
                center_coord=(image_metadata.exact_ephemeris.ra_deg,
                            image_metadata.exact_ephemeris.dec_deg)
            )
        else:
            print("No error ellipse data found, using RSS instead")
            error_ellipse = ErrorEllipse(
                smaa_3sig=image_metadata.exact_ephemeris.uncertainty['rss'],
                smia_3sig=image_metadata.exact_ephemeris.uncertainty['rss'],
                theta=0,
                center_coord=(image_metadata.exact_ephemeris.ra_deg,
                            image_metadata.exact_ephemeris.dec_deg)
            )

        # Create image name if saved
        image_name = ""
        if save_cutout and output_dir:
            image_name = f"cutout_visit{image_metadata.visit_id}_"\
                        f"detector{image_metadata.detector_id}_band{image_metadata.band}.fits"

        # Perform photometry
        target_result, sources_within_error = self.perform_photometry(
            calexp=calexp,
            ra_deg=image_metadata.exact_ephemeris.ra_deg,
            dec_deg=image_metadata.exact_ephemeris.dec_deg,
            cutout_size=cutout_size,
            display=display,
            error_ellipse=error_ellipse,
            save_cutout=save_cutout,
            output_dir=output_dir,
            image_name=image_name
        )

        # Prepare end result
        end_result = self._prepare_end_results(
            target_name=target_name,
            target_type=target_type,
            image_type=image_type,
            ephemeris_service=ephemeris_service,
            image_metadata=image_metadata,
            cutout_size=cutout_size,
            saved_image_name=image_name,
            target_result=target_result,
            sources_within_error=sources_within_error
        )

        # Save results to JSON if requested
        if save_json and output_dir:
            save_results_to_json(end_result, output_dir, json_filename)

        return end_result


    def perform_photometry(self, calexp, ra_deg, dec_deg, cutout_size=400,
                                display=True, psf_only=True, find_sources_flag=True,
                                save_cutout=False, output_dir=None, image_name=str,
                                error_ellipse: Optional[ErrorEllipse] = None):
        """
        Perform source detection and photometry on the image.
        Perform forced photometry at specified coordinates.

        Parameters
        ----------
        calexp : lsst.afw.image.ExposureF
            The calibrated exposure for photometry
        ra_deg : float
            Right ascension in degrees
        dec_deg : float
            Declination in degrees
        cutout_size : int, optional
            Size of the cutout in pixels (default: 400)
        display : bool, optional
            Whether to display the cutout (default: True)
        psf_only : bool, optional
            If True, only perform PSF photometry (default: True)
        find_sources_flag : bool, optional
            Whether to find nearby sources (default: True)
        save_cutout : bool, optional
            Whether to save the cutout (default: False)
        output_dir : str, optional
            Directory to save cutouts (default: None)
        image_name : str, optional
            Name for saved image (default: None)
        error_ellipse : ErrorEllipse, optional
            Error ellipse parameters for source filtering

        Returns
        -------
        dict
            Contains photometry results (target_result, sources_within_error)
        """
        # Get WCS and prepare coordinates
        target_img, bbox, offsets = self._prepare_image(calexp, ra_deg, dec_deg, cutout_size)
        x_offset, y_offset = offsets

        # Initialize coordinate lists
        ra_list, dec_list, found_sources = self._initialize_coordinates(
            target_img, ra_deg, dec_deg, find_sources_flag, error_ellipse)

        print(f"Found {len(ra_list)} sources")

        # Setup and perform measurements
        forced_meas_cat = self._perform_forced_photometry(
            target_img, ra_list, dec_list, x_offset, y_offset, psf_only)

        # Handle display and visualization
        if display or save_cutout:
            # Calculate display coordinates
            display_coords = []
            wcs = target_img.getWcs()

            for ra, dec in zip(ra_list, dec_list):
                coord = geom.SpherePoint(np.radians(ra), np.radians(dec), geom.radians)
                pixel_coord = wcs.skyToPixel(coord)
                display_coords.append((
                    pixel_coord.getX() - x_offset,
                    pixel_coord.getY() - y_offset
                ))

            if display:
                # Display using Firefly
                afwDisplay.setDefaultBackend('firefly')
                afw_display = afwDisplay.Display(frame=1)
                afw_display.mtv(target_img)
                afw_display.setMaskTransparency(100)

                with afw_display.Buffering():
                    # Plot original coordinates
                    afw_display.dot('o', display_coords[0][0], display_coords[0][1],
                                  size=10, ctype='red')

                    # Plot detected sources
                    for coord in display_coords[1:]:
                        afw_display.dot('o', coord[0], coord[1],
                                      size=20, ctype='blue')

                    # Plot error ellipse if provided
                    if error_ellipse:
                        error_ellipse._plot_error_ellipse(
                            display=afw_display,
                            wcs=wcs,
                            x_offset=x_offset,
                            y_offset=y_offset
                        )

            if save_cutout and output_dir:
                # Save cutout
                output_path = os.path.join(output_dir, image_name)
                target_img.writeFits(output_path)
                print(f"Saved cutout to: {output_path}")

        #  create PhotometryResult dataclass
        target_result, sources_within_error = self._prepare_photometry_results(
            forced_meas_cat=forced_meas_cat,
            ra=ra_deg,
            dec=dec_deg,
            found_sources=found_sources
        )

        return target_result, sources_within_error

    def _calculate_separations(
        self,
        table,
        ra_coord: float,
        dec_coord: float,
        sigma3: float
    ) -> afwTable.SourceTable:
        """
        Calculate angular separations between target coordinate and detected sources,
        and add them to the source table.

        Parameters
        ----------
        table : astropy.table.Table
            Table containing source detections
        ra_coord : float
            Right ascension in degrees
        dec_coord : float
            Declination in degrees
        sigma3 : float
            Sigma3 parameter for error ellipse

        Returns
        -------
        astropy.table.Table
            Original table with added 'separation' and 'sigma' columns
        """
        target_coord = geom.SpherePoint(np.radians(ra_coord), np.radians(dec_coord), geom.radians)
        separations = [
            target_coord.separation(
                geom.SpherePoint(ra, dec, geom.radians)
            ).asArcseconds()
            for ra, dec in zip(table['coord_ra'], table['coord_dec'])
        ]

        # Add separation column to table
        table['separation'] = separations
        table['sigma'] = [(sep / sigma3) * 3 for sep in separations]

        return table


    def find_measure_sources(self, calexp, ra_coord, dec_coord, error_ellipse: Optional[ErrorEllipse] = None):
        """
        Detect and measure sources in an image.

        Performs source detection and filtering:
        - Configures detection parameters
        - Runs source detection
        - Performs photometry
        - Filters sources within error ellipse

        Parameters
        ----------
        calexp : lsst.afw.image.ExposureF
            The calibrated exposure
        ra_coord : float
            Right ascension in degrees
        dec_coord : float
            Declination in degrees
        error_ellipse : ErrorEllipse, optional
            Error ellipse for filtering, by default None

        Returns
        -------
        astropy.table.Table
            Detected sources with measurements
        """
        print('Starting source detection and measurement')

        # Setup detection and measurement
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("coord_raErr", type="F")
        schema.addField("coord_decErr", type="F")

        config = SourceDetectionTask.ConfigClass()
        config.thresholdValue = self.detection_threshold
        config.thresholdType = "stdev"

        sourcedetectiontask = SourceDetectionTask(schema=schema, config=config)
        sourcemeasurementtask = SingleFrameMeasurementTask(
            schema=schema,
            config=SingleFrameMeasurementTask.ConfigClass(),
            algMetadata=dafBase.PropertyList()
        )

        # Run detection and measurement
        tab = afwTable.SourceTable.make(schema)
        result = sourcedetectiontask.run(tab, calexp)
        sources = result.sources
        sourcemeasurementtask.run(measCat=sources, exposure=calexp)

        # Get all sources and add separations
        sources_copy = sources.copy(True)
        table = sources_copy.asAstropy()
        table_with_sep = self._calculate_separations(table, ra_coord, dec_coord, error_ellipse.smaa_3sig)

        if error_ellipse is None:
            # If no error ellipse, return closest source
            min_index = np.argmin(table_with_sep['separation'])
            return table_with_sep[min_index:min_index+1]
        else:
            # Filter sources within error ellipse
            mask = []
            for row in table_with_sep:
                ra = np.rad2deg(row['coord_ra'])
                dec = np.rad2deg(row['coord_dec'])
                mask.append(error_ellipse.is_point_inside(ra, dec))

            filtered_table = table_with_sep[mask]
            filtered_table.sort("sigma")

            return filtered_table


    def _perform_forced_photometry(self, target_img, ra_list, dec_list,
                                   x_offset, y_offset, psf_only):
        """
        Setup forced photometry and perform measurements.

        Parameters
        ----------
        target_img : lsst.afw.image.ExposureF
            The calibrated exposure
        ra_list : list
            List of right ascensions in degrees
        dec_list : list
            List of declinations in degrees
        x_offset : int
            X offset of the cutout
        y_offset : int
            Y offset of the cutout
        psf_only : bool
            Whether to perform PSF photometry only

        Returns
        -------
        lsst.afw.table.SourceCatalog
            Source catalog with forced photometry on target
        """
        # Create schema and configure measurement
        schema = afwTable.SourceTable.makeMinimalSchema()
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
            config.plugins.names.extend([
                "base_GaussianFlux",
                "base_SdssShape",
                "base_CircularApertureFlux"
            ])

        config.doReplaceWithNoise = False

        # Create measurement task and source catalog
        forced_measurement_task = ForcedMeasurementTask(schema, config=config)
        forced_source = afwTable.SourceCatalog(schema)

        # Add sources and perform measurements
        wcs = target_img.getWcs()
        for ra, dec in zip(ra_list, dec_list):
            source_rec = forced_source.addNew()
            coord = geom.SpherePoint(np.radians(ra), np.radians(dec), geom.radians)
            source_rec.setCoord(coord)

            pixel_coord = wcs.skyToPixel(coord)
            source_rec['centroid_x'] = pixel_coord.getX()
            source_rec['centroid_y'] = pixel_coord.getY()
            source_rec['type_flag'] = 0

        # Run forced photometry
        forced_meas_cat = forced_measurement_task.generateMeasCat(
            target_img, forced_source, target_img.getWcs())
        forced_measurement_task.run(
            forced_meas_cat, target_img, forced_source, target_img.getWcs())

        return forced_meas_cat

    def _prepare_image(self, calexp, ra_deg, dec_deg, cutout_size):
        """
        Prepare the image by creating a cutout around the specified coordinates.

        Parameters
        ----------
        calexp : lsst.afw.image.ExposureF
            The calibrated exposure
        ra_deg : float
            Right ascension in degrees
        dec_deg : float
            Declination in degrees
        cutout_size : int
            Size of the cutout in pixels

        Returns
        -------
        tuple
            (target_img, bbox, offsets) where:
            - target_img is the cutout image
            - bbox is the bounding box
            - offsets is a tuple of (x_offset, y_offset)
        """
        wcs = calexp.getWcs()
        coord = geom.SpherePoint(np.radians(ra_deg), np.radians(dec_deg), geom.radians)

        pixel_coord = wcs.skyToPixel(coord)
        x_center, y_center = pixel_coord.getX(), pixel_coord.getY()
        half_size = cutout_size // 2


        min_x, max_x = 0, calexp.getWidth()
        min_y, max_y= 0, calexp.getHeight()

        if cutout_size <= 0:
            print("Using the complete image.")
            return calexp, None, (0, 0)

        elif (x_center - half_size) < min_x or \
        (x_center + half_size) > max_x or \
        (y_center - half_size) < min_y or \
        (y_center + half_size) > max_y:
            print("The cutout boundaries are outside of the image.")
            print("Using the complete image.")
            return calexp, None, (0, 0)

        print(f"Creating cutout with size {cutout_size} pixels")

        # Create bounding box for cutout
        bbox = geom.Box2I()
        bbox.include(geom.Point2I(int(x_center - half_size), int(y_center - half_size)))
        bbox.include(geom.Point2I(int(x_center + half_size), int(y_center + half_size)))

        # Create cutout
        target_img = calexp.Factory(calexp, bbox, origin=afwImage.LOCAL, deep=False)

        return target_img, bbox, (bbox.getMinX(), bbox.getMinY())

    def _initialize_coordinates(self, target_img, ra_deg, dec_deg, find_sources_flag,
                              error_ellipse: Optional[ErrorEllipse] = None):
        """
        Initialize coordinate lists and find sources if requested.

        Parameters
        ----------
        target_img : lsst.afw.image.ExposureF
            The calibrated exposure
        ra_deg : float
            Right ascension in degrees
        dec_deg : float
            Declination in degrees
        find_sources_flag : bool
            Whether to find sources
        error_ellipse : ErrorEllipse, optional
            Error ellipse for filtering, by default None

        Returns
        -------
        tuple
            Tuple containing lists of right ascensions and declinations
            and detected sources
        """
        ra_list = [ra_deg]
        dec_list = [dec_deg]

        if find_sources_flag:
            sources = self.find_measure_sources(target_img, ra_deg, dec_deg, error_ellipse)
            for source in sources:
                ra_list.append(np.rad2deg(source['coord_ra']))
                dec_list.append(np.rad2deg(source['coord_dec']))

        return ra_list, dec_list, sources


    def _prepare_photometry_results(self, forced_meas_cat, ra, dec, found_sources):
        """
        Prepare the photometry results for the target and sources within the error ellipse.

        Parameters
        ----------
        forced_meas_cat : lsst.afw.table.SourceCatalog
            Source catalog with measurements
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        found_sources : astropy.table.Table
            Detected sources within error ellipse

        Returns
        -------
        tuple
            Tuple containing the target result and the list of sources within the error ellipse
        """

        # Prepare forced photometry results for the coordinates
        target_result = PhotometryResult(
            ra=ra,
            dec=dec,
            ra_err=0,
            dec_err=0,
            x=0,
            y=0,
            x_err=0,
            y_err=0,
            snr=(forced_meas_cat[0].get(
                'base_PsfFlux_instFlux'
            )/ forced_meas_cat[0].get(
                'base_PsfFlux_instFluxErr'
            )) if forced_meas_cat[0].get(
                'base_PsfFlux_instFluxErr'
            ) > 0 else 0,
            flux=forced_meas_cat[0].get('base_PsfFlux_instFlux') if forced_meas_cat[0].get(
                'base_PsfFlux_instFlux'
            ) > 0 else 0,
            flux_err=forced_meas_cat[0].get('base_PsfFlux_instFluxErr') if forced_meas_cat[0].get(
                'base_PsfFlux_instFluxErr'
            ) > 0 else 0 ,
            mag=-2.5 * np.log10(forced_meas_cat[0].get(
                'base_PsfFlux_instFlux'
            )) + 31.4 if forced_meas_cat[0].get(
                'base_PsfFlux_instFlux'
            ) > 0 else 0,
            mag_err=2.5 / np.log(10) * forced_meas_cat[0].get(
                'base_PsfFlux_instFluxErr'
            ) / forced_meas_cat[0].get(
                'base_PsfFlux_instFlux'
            ) if forced_meas_cat[0].get(
                'base_PsfFlux_instFlux'
            ) > 0 else 0,
            separation=0,
            sigma=0,
            flags={}
        )

        # Prepare results for additional sources (if any)
        #  TODO make a method for precise sigma calculation
        sources_within_error = []
        if len(found_sources["coord_ra"]) > 1:
            for i in range(1, len(found_sources)):
                source_result = PhotometryResult(
                    ra=found_sources[i].get('coord_ra'),
                    dec=found_sources[i].get('coord_ra'),
                    ra_err=float(found_sources[i].get('coord_raErr')),
                    dec_err=float(found_sources[i].get('coord_decErr')),
                    x=found_sources[i].get('slot_Centroid_x'),
                    y=found_sources[i].get('slot_Centroid_y'),
                    x_err=float(found_sources[i].get('slot_Centroid_xErr')),
                    y_err=float(found_sources[i].get('slot_Centroid_yErr')),
                    snr=(found_sources[i].get(
                        'base_PsfFlux_instFlux'
                    )/ found_sources[i].get(
                        'base_PsfFlux_instFluxErr'
                    )) if found_sources[i].get(
                        'base_PsfFlux_instFluxErr'
                    ) > 0 else 0,
                    flux=found_sources[i].get('base_PsfFlux_instFlux'),
                    flux_err=found_sources[i].get('base_PsfFlux_instFluxErr'),
                    mag=-2.5 * np.log10(found_sources[i].get(
                        'base_PsfFlux_instFlux'
                    )) + 31.4 if found_sources[i].get(
                        'base_PsfFlux_instFlux'
                    ) > 0 else np.nan,
                    mag_err=2.5 / np.log(10) * found_sources[i].get(
                        'base_PsfFlux_instFluxErr'
                    ) / found_sources[i].get(
                        'base_PsfFlux_instFlux'
                    ) if found_sources[i].get('base_PsfFlux_instFlux') > 0 else 0,
                    separation=found_sources[i].get('separation'),
                    sigma=found_sources[i].get('sigma'),
                    flags={
                        'base_PsfFlux_flag_badCentroid' : found_sources[0].get(
                            'base_PsfFlux_flag_badCentroid'
                        ),
                        'base_PsfFlux_flag_badCentroid_edge' : found_sources[0].get(
                            'base_PsfFlux_flag_badCentroid_edge'
                        ),
                        'slot_Centroid_flag_edge' : found_sources[0].get('slot_Centroid_flag_edge')
                          }
                )

                sources_within_error.append(source_result)

        return target_result, sources_within_error

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
        sources_within_error
    ):
        """
        Prepare the end result dataclass.

        Parameters
        ----------
        target_name : str
            Name of the target
        target_type : str
            Classification of the target
        image_type : str
            Type of image (calexp or goodSeeingDiff_differenceExp)
        ephemeris_service : str
            Service used for ephemeris data
        image_metadata : ImageMetadata
            Metadata for the image
        cutout_size : int
            Size of the cutout in pixels
        saved_image_name : str
            Name of the saved image
        target_result : PhotometryResult
            Forced photometry result for the target coordinates
        sources_within_error : list
            List of photometry results for sources within the error ellipse

        Returns
        -------
        EndResult
            Complete results of photometry processing
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
            phot_within_error_ellipse=sources_within_error
        )

        return end_result
