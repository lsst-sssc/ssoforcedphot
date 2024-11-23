"""
Photometry Service Module provides functionality for performing photometry 
on LSST images, including source detection and flux measurements.
"""

import numpy as np
import os
import lsst.geom as geom
import lsst.afw.table as afwTable
import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
from typing import Union, Optional, Tuple, List
from dataclasses import dataclass
from lsst.daf.butler import Butler
from lsst.meas.base import ForcedMeasurementTask
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.base import SingleFrameMeasurementTask
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.visualization.wcsaxes import SphericalCircle




def calculate_position_angle(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """
    Calculate the position angle between two points on a sphere.
    """
    coord1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    coord2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    return coord1.position_angle(coord2).wrap_at(2 * np.pi * u.rad).radian

@dataclass
class ErrorEllipse:
    smaa_3sig: float
    smia_3sig: float
    theta: float
    center_coord: Tuple[float, float]

    def is_point_inside(self, ra: float, dec: float) -> bool:
        """
        Check if a point (ra, dec) is inside the error ellipse.
        """
        center = SkyCoord(ra=self.center_coord[0] * u.deg, dec=self.center_coord[1] * u.deg)
        point = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        
        sep = center.separation(point).arcsec
        pa = center.position_angle(point) - Angle(self.theta, unit=u.deg)
        
        dx = sep * np.sin(pa)
        dy = sep * np.cos(pa)
        
        return (dx / self.smaa_3sig) ** 2 + (dy / self.smia_3sig) ** 2 <= 1

    def get_ellipse_points(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate points along the error ellipse.
        """
        angles = np.linspace(0, 2 * np.pi, n_points)
        dx = self.smaa_3sig * np.cos(angles)
        dy = self.smia_3sig * np.sin(angles)
        
        # Rotate by theta angle
        theta_rad = np.radians(self.theta)
        rotated_dx = dx * np.cos(theta_rad) - dy * np.sin(theta_rad)
        rotated_dy = dx * np.sin(theta_rad) + dy * np.cos(theta_rad)
        
        # Convert to RA and Dec offsets
        center = SkyCoord(ra=self.center_coord[0] * u.deg, dec=self.center_coord[1] * u.deg)
        ellipse_points = center.spherical_offsets_by(rotated_dx * u.arcsec, rotated_dy * u.arcsec)
        
        return ellipse_points.ra.deg, ellipse_points.dec.deg

    def _plot_error_ellipse(self, display, wcs, x_offset: float, y_offset: float, n_points: int = 100):
        """
        Plot error ellipse on the display.
        """
        ra_deg, dec_deg = self.get_ellipse_points(n_points)
        
        pixel_coords = []
        for ra, dec in zip(ra_deg, dec_deg):
            coord = geom.SpherePoint(np.radians(ra), np.radians(dec), geom.radians)
            pixel_coord = wcs.skyToPixel(coord)
            pixel_coords.append((pixel_coord.getX() - x_offset, pixel_coord.getY() - y_offset))
        
        for i in range(len(pixel_coords)):
            j = (i + 1) % len(pixel_coords)
            display.line([(pixel_coords[i][0], pixel_coords[i][1]),
                          (pixel_coords[j][0], pixel_coords[j][1])],
                         ctype='red')


class PhotometryService:
    """
    A service class for performing photometry operations on astronomical images.
    
    This class provides methods for forced photometry and source detection on
    LSST calibrated exposures.

    Attributes
    ----------
    detection_threshold : float
        Threshold value for source detection (default: 5)
    display : object
        Display object for visualization
    """
    
    def __init__(self, detection_threshold: float = 5):
        """
        Initialize the PhotometryService.

        Parameters
        ----------
        detection_threshold : float, optional
            Threshold value for source detection (default: 5)
        """
        self.display = None
        self.detection_threshold = detection_threshold

    def perform_batch_photometry(self, calexp, coordinates: list[tuple[float, float]], 
                               cutout_size=0, display=True, psf_only=True, 
                               find_sources_flag=True, save_cutouts=False,
                               output_dir=None, error_ellipse: Optional[ErrorEllipse] = None) -> list[dict]:
        """
        Perform forced photometry for multiple sets of coordinates.

        Parameters
        ----------
        calexp : lsst.afw.image.ExposureF
            The calibrated exposure for photometry
        coordinates : List[Tuple[float, float]]
            List of (ra, dec) coordinate pairs in degrees
        cutout_size : int, optional
            Size of the cutout in pixels (default: 400)
        display : bool, optional
            Whether to display the cutouts (default: True)
        psf_only : bool, optional
            If True, only perform PSF photometry (default: True)
        find_sources_flag : bool, optional
            Whether to find nearby sources (default: True)
        save_cutouts : bool, optional
            Whether to save cutouts to disk (default: False)
        output_dir : str, optional
            Directory to save cutouts (default: None)
        error_ellipse : ErrorEllipse, optional
            Error ellipse parameters for source filtering

        Returns
        -------
        List[Dict]
            List of result dictionaries for each coordinate pair
        """
        if save_cutouts and output_dir is None:
            output_dir = os.path.join(os.getenv("HOME"), 'WORK/ssoforcedphot/tests/forcedphot/data/')
            print(output_dir)
            os.makedirs(output_dir, exist_ok=True)

        results = []
        for i, (ra_deg, dec_deg) in enumerate(coordinates):
            print(f"\nProcessing coordinate set {i+1}/{len(coordinates)}:")
            print(f"RA: {ra_deg}, Dec: {dec_deg}")
            
            result = self.perform_forced_photometry(
                calexp, ra_deg, dec_deg, 
                cutout_size=cutout_size,
                display=display,
                psf_only=psf_only,
                find_sources_flag=find_sources_flag,
                save_cutout=save_cutouts,
                output_dir=output_dir,
                cutout_index=i,
                error_ellipse=error_ellipse
            )
            results.append(result)
        
        return results
     
    def perform_forced_photometry(self, calexp, ra_deg, dec_deg, cutout_size=400,
                                display=True, psf_only=True, find_sources_flag=True,
                                save_cutout=False, output_dir=None, cutout_index=0,
                                error_ellipse: Optional[ErrorEllipse] = None):
        """
        Perform forced photometry on an exposure at specified coordinates.
        
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
        cutout_index : int, optional
            Index for naming the cutout file (default: 0)
        error_ellipse : ErrorEllipse, optional
            Error ellipse parameters for source filtering
        
        Returns
        -------
        dict
            Contains flux measurements, coordinates, and cutout data
        """
        # Get WCS and prepare coordinates
        target_img, bbox, offsets = self._prepare_image(calexp, ra_deg, dec_deg, cutout_size)
        x_offset, y_offset = offsets
        
        # Initialize coordinate lists
        ra_list, dec_list, original_sources = self._initialize_coordinates(
            target_img, ra_deg, dec_deg, find_sources_flag, error_ellipse)
        # print(original_sources['separation'])
        
        # Setup and perform measurements
        forced_meas_cat = self._setup_and_perform_measurements(
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
                        error_ellipse._plot_error_ellipse(display=afw_display, wcs=wcs, x_offset=x_offset, y_offset=y_offset)

            if save_cutout and output_dir:
                # Save cutout
                filename = f"cutout_{cutout_index}_ra{ra_deg:.4f}_dec{dec_deg:.4f}.fits"
                output_path = os.path.join(output_dir, filename)
                target_img.writeFits(output_path)
                print(f"Saved cutout to: {output_path}")
        
        # Return results
        return self._prepare_results(forced_meas_cat, ra_list, dec_list,
                                   target_img, cutout_size)

    def _calculate_separations(self, table, ra_coord: float, dec_coord: float) -> afwTable.SourceTable:
        """
        Calculate angular separations between target coordinate and all sources,
        and add them to the source table.
        
        Parameters
        ----------
        table : astropy.table.Table
            Table containing source detections
        ra_coord : float
            Right ascension in degrees
        dec_coord : float
            Declination in degrees
        
        Returns
        -------
        astropy.table.Table
            Original table with added 'separation' column in arcseconds
        """
        target_coord = geom.SpherePoint(np.radians(ra_coord), np.radians(dec_coord), geom.radians)
        separations = [
            target_coord.separation(
                geom.SpherePoint(ra, dec, geom.radians)
            ).asArcseconds()  # Convert to arcseconds
            for ra, dec in zip(table['coord_ra'], table['coord_dec'])
        ]
        
        # Add separation column to table
        table['separation'] = separations
        return table

    
    def find_sources(self, calexp, ra_coord, dec_coord, error_ellipse: Optional[ErrorEllipse] = None):
        """
        Find sources on the fits or cutout, then find sources within error ellipse if provided.
        
        Parameters
        ----------
        calexp : lsst.afw.image.ExposureF
            The calibrated exposure
        ra_coord : float
            Right ascension in degrees
        dec_coord : float
            Declination in degrees
        error_ellipse : ErrorEllipse, optional
            Error ellipse parameters for filtering sources
        
        Returns
        -------
        astropy.table.Table
            Table of detected sources within the error ellipse
        """
        print('Starting source detection and measurement')
        
        # Setup detection and measurement
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("coord_raErr", type="F")
        schema.addField("coord_decErr", type="F")
        
        config = SourceDetectionTask.ConfigClass()
        config.thresholdValue = self.detection_threshold
        config.thresholdType = "stdev"
        
        sourceDetectionTask = SourceDetectionTask(schema=schema, config=config)
        sourceMeasurementTask = SingleFrameMeasurementTask(
            schema=schema,
            config=SingleFrameMeasurementTask.ConfigClass(),
            algMetadata=dafBase.PropertyList()
        )
        
        # Run detection and measurement
        tab = afwTable.SourceTable.make(schema)
        result = sourceDetectionTask.run(tab, calexp)
        sources = result.sources
        sourceMeasurementTask.run(measCat=sources, exposure=calexp)
        
        # Get all sources and add separations
        sources_copy = sources.copy(True)
        table = sources_copy.asAstropy()
        table_with_sep = self._calculate_separations(table, ra_coord, dec_coord)
        
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
            
            if len(filtered_table) == 0:
                print(f"No sources inside the error-ellipse.")
            else:
                print(f"Number of sources inside the error-ellipse: {len(filtered_table)}")
                for row in filtered_table:
                    print(f"""Found source in ellipse:
                        RA = {np.rad2deg(row['coord_ra'])} deg
                        Dec = {np.rad2deg(row['coord_dec'])} deg
                        PSF flux = {row['base_PsfFlux_instFlux']}
                        Separation = {row['separation']:.2f} arcsec""")
            
            return filtered_table


    def _setup_and_perform_measurements(self, target_img, ra_list, dec_list,
                                        x_offset, y_offset, psf_only):
        """Setup measurement configuration and perform measurements."""
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
        
        # Configure measurement task
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
        
        if cutout_size <= 0:
            return calexp, None, (0, 0)
            
        print(f"Creating cutout with size {cutout_size} pixels")
        
        # Create bounding box for cutout
        half_size = cutout_size // 2
        x_center, y_center = pixel_coord.getX(), pixel_coord.getY()
        
        bbox = geom.Box2I()
        bbox.include(geom.Point2I(int(x_center - half_size), int(y_center - half_size)))
        bbox.include(geom.Point2I(int(x_center + half_size), int(y_center + half_size)))
        
        # Create cutout
        target_img = calexp.Factory(calexp, bbox, origin=afwImage.LOCAL, deep=False)
        
        return target_img, bbox, (bbox.getMinX(), bbox.getMinY())
        
    def _initialize_coordinates(self, target_img, ra_deg, dec_deg, find_sources_flag,
                              error_ellipse: Optional[ErrorEllipse] = None):
        """Initialize coordinate lists and find sources if requested."""
        ra_list = [ra_deg]
        dec_list = [dec_deg]
        
        if find_sources_flag:
            sources = self.find_sources(target_img, ra_deg, dec_deg, error_ellipse)
            for source in sources:
                ra_list.append(np.rad2deg(source['coord_ra']))
                dec_list.append(np.rad2deg(source['coord_dec']))
        
        return ra_list, dec_list, sources

    def _prepare_results(self, forced_meas_cat, ra_list, dec_list, target_img, cutout_size):
        """Prepare the results dictionary."""
        # Prepare forced photometry results for the target
        target_result = {
            'ra': ra_list[0],
            'dec': dec_list[0],
            'flux': forced_meas_cat[0].get('base_PsfFlux_instFlux'),
            'flux_err': forced_meas_cat[0].get('base_PsfFlux_instFluxErr'),
        }
        target_result['snr'] = target_result['flux'] / target_result['flux_err'] if target_result['flux_err'] > 0 else 0
        target_result['mag'] = -2.5 * np.log10(target_result['flux']) + 31.4 if target_result['flux'] > 0 else np.nan
        target_result['mag_err'] = 2.5 / np.log(10) * target_result['flux_err'] / target_result['flux'] if target_result['flux'] > 0 else np.nan

        # Prepare results for additional sources (if any)
        additional_sources = []
        for i in range(1, len(ra_list)):
            source_result = {
                'ra': ra_list[i],
                'dec': dec_list[i],
                'flux': forced_meas_cat[i].get('base_PsfFlux_instFlux'),
                'flux_err': forced_meas_cat[i].get('base_PsfFlux_instFluxErr'),
            }
            source_result['snr'] = source_result['flux'] / source_result['flux_err'] if source_result['flux_err'] > 0 else 0
            source_result['mag'] = -2.5 * np.log10(source_result['flux']) + 31.4 if source_result['flux'] > 0 else np.nan
            source_result['mag_err'] = 2.5 / np.log(10) * source_result['flux_err'] / source_result['flux'] if source_result['flux'] > 0 else np.nan
            additional_sources.append(source_result)

        return {
            'target': target_result,
            'sources_in_ellipse': additional_sources,
            'cutout': target_img if cutout_size > 0 else None
        }


def main():
    # Create error ellipse parameters
    error_ellipse = ErrorEllipse(
        smaa_3sig=40.0,  # 10 arcseconds
        smia_3sig=20.0,   # 5 arcseconds
        theta=90.0,      # 45 degrees
        center_coord=(71.1496951, -30.5820904)  # RA, Dec in degrees
    )
    
    # Initialize services
    butler = Butler('dp02', collections='2.2i/runs/DP0.2')
    service = PhotometryService(detection_threshold=4)
    
    # Get calibrated exposure
    calexp = butler.get('calexp', dataId={'visit': 512055, 'detector': 75})
    
    # Perform photometry with error ellipse
    # In your main function:
    results = service.perform_batch_photometry(
        calexp,
        [(71.1496951, -30.5820904)],
        cutout_size=0,
        display=True,
        error_ellipse=error_ellipse
    )

    # Access the results
    for result in results:
        # Target forced photometry
        target = result['target']
        print(f"\nTarget Forced Photometry:")
        print(f"RA: {target['ra']:.6f}, Dec: {target['dec']:.6f}")
        print(f"Flux: {target['flux']:.2f} ± {target['flux_err']:.2f}")
        print(f"SNR: {target['snr']:.2f}")
        print(f"Magnitude: {target['mag']:.2f} ± {target['mag_err']:.2f}")

        # Sources in error ellipse
        print(f"\nSources in Error Ellipse:")
        for i, source in enumerate(result['sources_in_ellipse'], 1):
            print(f"\nSource {i}:")
            print(f"RA: {source['ra']:.6f}, Dec: {source['dec']:.6f}")
            # print(f"Separation: {source['separation']:.2f}")
            print(f"Flux: {source['flux']:.2f} ± {source['flux_err']:.2f}")
            print(f"SNR: {source['snr']:.2f}")
            print(f"Magnitude: {source['mag']:.2f} ± {source['mag_err']:.2f}")

    # print(results)

if __name__ == "__main__":
    main()
