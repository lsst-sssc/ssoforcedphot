"""
Utils module for forcedphot package.

This module provides utility functions for forcedphot package.

Main Components:
    - Linear interpolation of coordinates
    - Ephemeris data handling
    - Search parameters Dataclass
    - Image metadata Dataclass
    - Error ellipse Dataclass
    - Photometry result Dataclass
    - End result Dataclass
"""

from dataclasses import dataclass
import csv
import os
from pathlib import Path
from typing import List, Union
from astropy.time import Time

import astropy.units as u
import lsst.geom as geom
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astropy.time import Time
from ephemeris.data_model import QueryResult


def interpolate_coordinates(ra1, dec1, ra2, dec2, time1, time2, target_time):
    """
    Interpolate between two positions (RA, DEC) for a given time.

    Parameters
    ----------
    ra1, ra2 : float or Quantity
        Right Ascension for the first and second position
    dec1, dec2 : float or Quantity
        Declination for the first and second position
    time1, time2 : float
        Times corresponding to the first and second position
    target_time : float
        Time at which to interpolate

    Returns
    -------
    tuple
        (ra, dec) at the interpolated time
    """
    # Calculate the fraction of the time interval
    fraction = (target_time - time1) / (time2 - time1)

    # If inputs aren't Quantities, assume they're in degrees
    if not isinstance(ra1, u.Quantity):
        ra1 = ra1 * u.deg
    if not isinstance(dec1, u.Quantity):
        dec1 = dec1 * u.deg
    if not isinstance(ra2, u.Quantity):
        ra2 = ra2 * u.deg
    if not isinstance(dec2, u.Quantity):
        dec2 = dec2 * u.deg

    # Handle RA wrap-around (if needed)
    # If the difference between RAs is more than 180 degrees,
    # adjust one of them by 360 degrees
    if abs(ra2 - ra1) > 180 * u.deg:
        if ra2 > ra1:
            ra1 += 360 * u.deg
        else:
            ra2 += 360 * u.deg

    # Linear interpolation
    ra_interp = ra1 + (ra2 - ra1) * fraction
    dec_interp = dec1 + (dec2 - dec1) * fraction

    # Normalize RA to [0, 360) degrees
    ra_interp = ra_interp % (360 * u.deg)

    return ra_interp.value, dec_interp.value, target_time


@dataclass
class EphemerisDataCompressed:
    """
    Represents and manages ephemeris data with functionality for loading, filtering,
    and processing ephemeris information.

    Attributes:
        datetime (Time): Observation datetime
        ra_deg (float): Right Ascension in degrees
        dec_deg (float): Declination in degrees
        ra_rate (float): RA rate in arcsec/hour
        dec_rate (float): DEC rate in arcsec/hour
        uncertainty (Dict[str, float]): Dictionary containing uncertainty measurements
            - rss: Root sum square uncertainty in arcsec
            - smaa: Semi-major axis of the error-ellipse in arcsec
            - smia: Semi-minor axis of the error-ellipse in arcsec
            - theta: Position angle of the smaa in degrees
    """

    datetime: Time
    ra_deg: float
    dec_deg: float
    ra_rate: float
    dec_rate: float
    uncertainty: dict[str, float]

    @staticmethod
    def load_ephemeris(file_path: str) -> list["EphemerisDataCompressed"]:
        """
        Loads ephemeris data from an ECSV file and converts it to EphemerisDataCompressed objects.

        Args:
            file_path (str): Path to the ECSV file containing ephemeris data.

        Returns:
            List[EphemerisDataCompressed]: List of parsed ephemeris data rows.

        Raises:
            ValueError: If there's an error loading or parsing the ephemeris file.
        """
        try:
            data = Table.read(file_path, format="ascii.ecsv")
            rows = []

            for row in data:
                ephemeris_row = EphemerisDataCompressed(
                    datetime=Time(row["datetime"], scale="utc", format="jd"),
                    ra_deg=float(row["RA_deg"]),
                    dec_deg=float(row["DEC_deg"]),
                    ra_rate=float(row["RA_rate_arcsec_per_h"]),
                    dec_rate=float(row["DEC_rate_arcsec_per_h"]),
                    uncertainty={
                        "rss": float(row["RSS_3sigma_arcsec"]),
                        "smaa": float(row["SMAA_3sigma_arcsec"]),
                        "smia": float(row["SMIA_3sigma_arcsec"]),
                        "theta": float(row["Theta_3sigma_deg"]),
                    },
                )
                rows.append(ephemeris_row)

            return rows

        except Exception as e:
            raise ValueError(f"Error loading ephemeris file: {str(e)}") from e

    @staticmethod
    def compress_ephemeris(query_result: QueryResult) -> list["EphemerisDataCompressed"]:
        """
        Compresses ephemeris data into a list of EphemerisDataCompressed objects.

        Args:
            query_result (QueryResult): QueryResult object containing ephemeris data.

        Returns:
            List[EphemerisDataCompressed]: List of EphemerisDataCompressed objects.
        """
        ephemeris_data = query_result.ephemeris

        # Create a list of EphemerisDataCompressed objects
        compressed_ephemeris = []
        for i in range(len(ephemeris_data.datetime)):
            row = EphemerisDataCompressed(
                datetime=ephemeris_data.datetime[i],
                ra_deg=ephemeris_data.RA_deg[i],
                dec_deg=ephemeris_data.DEC_deg[i],
                ra_rate=ephemeris_data.RA_rate_arcsec_per_h[i],
                dec_rate=ephemeris_data.DEC_rate_arcsec_per_h[i],
                uncertainty={
                    "rss": ephemeris_data.RSS_3sigma_arcsec[i],
                    "smaa": ephemeris_data.SMAA_3sigma_arcsec[i],
                    "smia": ephemeris_data.SMIA_3sigma_arcsec[i],
                    "theta": ephemeris_data.Theta_3sigma_deg[i],
                },
            )
            compressed_ephemeris.append(row)

        return compressed_ephemeris

    @staticmethod
    def get_relevant_rows(
        ephemeris_rows: list["EphemerisDataCompressed"], t_min: Time, t_max: Time
    ) -> list["EphemerisDataCompressed"]:
        """
        Retrieves 2-3 rows of ephemeris data suitable for interpolation within a time range.

        Args:
            ephemeris_rows (List[EphemerisDataCompressed]): Complete list of ephemeris rows.
            t_min (Time): Start time of the desired range.
            t_max (Time): End time of the desired range.

        Returns:
            List[EphemerisDataCompressed]: 2-3 rows containing the relevant time period and
            adjacent data points.
        """
        within_range_idx = None
        for i, row in enumerate(ephemeris_rows):
            if t_min.mjd <= row.datetime.mjd <= t_max.mjd:
                within_range_idx = i
                break

        if within_range_idx is None:
            mid_time = Time((t_min.mjd + t_max.mjd) / 2, format="mjd")
            time_diffs = [abs(row.datetime.mjd - mid_time.mjd) for row in ephemeris_rows]
            within_range_idx = np.argmin(time_diffs)

        relevant_rows = []
        if within_range_idx > 0:
            relevant_rows.append(ephemeris_rows[within_range_idx - 1])
        relevant_rows.append(ephemeris_rows[within_range_idx])
        if within_range_idx < len(ephemeris_rows) - 1:
            relevant_rows.append(ephemeris_rows[within_range_idx + 1])

        return relevant_rows

    @staticmethod
    def get_time_range(rows: list["EphemerisDataCompressed"]) -> tuple[Time, Time]:
        """
        Determines the complete time span covered by the ephemeris data.

        Args:
            rows (List[EphemerisDataCompressed]): List of ephemeris rows.

        Returns:
            Tuple[Time, Time]: Start and end times of the ephemeris data.
        """
        return rows[0].datetime, rows[-1].datetime

    @staticmethod
    def get_time_windows(rows: list["EphemerisDataCompressed"]) -> list[tuple[Time, Time, float, float]]:
        """
        Creates time windows for coordinate interpolation from ephemeris data.

        Args:
            rows (List[EphemerisDataCompressed]): List of ephemeris rows.

        Returns:
            List[Tuple[Time, Time, float, float]]: List of time windows with corresponding coordinates.
                Each tuple contains (start_time, end_time, ra_deg, dec_deg).
        """
        windows = []
        for i, row in enumerate(rows):
            start_time = rows[i - 1].datetime if i > 0 else row.datetime

            end_time = rows[i + 1].datetime if i < len(rows) - 1 else row.datetime

            windows.append((start_time, end_time, row.ra_deg, row.dec_deg))

        return windows


@dataclass
class SearchParameters:
    """
    Dataclass to hold user-defined search parameters.

    Attributes
    ----------
    bands : set[str]
        Set of filter bands to search for
    ephemeris_file : str
        Path to the ECSV file containing ephemeris data
    """

    bands: set[str]
    ephemeris_file: str


@dataclass
class SearchParametersPolygon:
    """
    Dataclass to hold user-defined search parameters for polygon search.

    Attributes
    ----------
    bands : set[str]
        Set of filter bands to search for
    ephemeris_file : str
        Path to the ECSV file containing ephemeris data
    time_interval: float
        The maximum duration for each polygon segment in days.
    widening: float
        The desired width of the polygon on either side of the path,
        in arcseconds.
    """

    bands: set[str]
    ephemeris_file: str
    time_interval: float
    widening: float

    
@dataclass
class ImageMetadata:
    """
    Dataclass to hold image metadata.

    Attributes
    ----------
    visit_id : int
        LSST visit ID
    detector_id : int
        LSST detector ID
    band : str
        LSST filter band
    coordinates : Tuple[float, float]
        RA and Dec coordinates in degrees
    t_min : Time
        Observation start time
    t_max : Time
        Observation end time
    ephemeris_data : List[EphemerisDataCompressed]
        List of ephemeris data
    exact_ephemeris : EphemerisDataCompressed
        Ephemeris data with interpolated coordinates
    """

    visit_id: int
    detector_id: int
    band: str
    coordinates_central: tuple[float, float]
    t_min: Time
    t_max: Time
    ephemeris_data: list[EphemerisDataCompressed]
    exact_ephemeris: EphemerisDataCompressed


@dataclass
class ErrorEllipse:
    """
    Represents and manages error ellipse calculations.

    This class handles various geometric operations related to error ellipses, including
    point containment testing, ellipse point generation, and visualization.

    Attributes:
        smaa_3sig (float): Semi-major axis (3-sigma) in arcseconds
        smia_3sig (float): Semi-minor axis (3-sigma) in arcseconds
        theta (float): Position angle in degrees, measured East of North
        center_coord (Tuple[float, float]): Center coordinates (RA, Dec) in degrees
    """

    smaa_3sig: float
    smia_3sig: float
    theta: float
    center_coord: tuple[float, float]

    def is_point_inside(self, ra: float, dec: float) -> bool:
        """
        Determines if a given point lies within the error ellipse.

        The calculation transforms the point to the ellipse coordinate system
        and checks if it falls within the normalized ellipse equation.

        Args:
            ra (float): Right Ascension of the point in degrees
            dec (float): Declination of the point in degrees

        Returns:
            bool: True if the point is inside the ellipse, False otherwise
        """
        center = SkyCoord(ra=self.center_coord[0] * u.deg, dec=self.center_coord[1] * u.deg)
        point = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        sep = center.separation(point).arcsec
        pa = center.position_angle(point) - Angle(self.theta, unit=u.deg)

        dx = sep * np.sin(pa)
        dy = sep * np.cos(pa)

        return (dx / self.smaa_3sig) ** 2 + (dy / self.smia_3sig) ** 2 <= 1

    def get_ellipse_points(self, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates points along the error ellipse boundary.

        Creates a set of points that define the error ellipse boundary,
        useful for plotting or visualization purposes.

        Args:
            n_points (int, optional): Number of points to generate. Defaults to 100.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of RA and Dec coordinates in degrees
                defining the ellipse boundary
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
        ellipse_points = center.spherical_offsets_by(-rotated_dx * u.arcsec, rotated_dy * u.arcsec)

        return ellipse_points.ra.deg, ellipse_points.dec.deg

    def _plot_error_ellipse(self, display, wcs, x_offset: float, y_offset: float, n_points: int = 100):
        """
        Plots the error ellipse.

        Renders the error ellipse on the image using the provided
        World Coordinate System (WCS) transformation.

        Args:
            display: Display object to plot on
            wcs: World Coordinate System transformation
            x_offset (float): X-axis offset in pixels
            y_offset (float): Y-axis offset in pixels
            n_points (int, optional): Number of points to use for the ellipse. Defaults to 100.
        """
        ra_deg, dec_deg = self.get_ellipse_points(n_points)

        pixel_coords = []
        for ra, dec in zip(ra_deg, dec_deg):
            coord = geom.SpherePoint(np.radians(ra), np.radians(dec), geom.radians)
            pixel_coord = wcs.skyToPixel(coord)
            pixel_coords.append((pixel_coord.getX() - x_offset, pixel_coord.getY() - y_offset))

        for i in range(len(pixel_coords)):
            j = (i + 1) % len(pixel_coords)
            display.line(
                [(pixel_coords[i][0], pixel_coords[i][1]), (pixel_coords[j][0], pixel_coords[j][1])],
                ctype="red",
            )


@dataclass
class PhotometryResult:
    """
    Dataclass to hold the results of photometric measurements.

    Attributes
    ----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    ra_err : float
        RA uncertainty in arcsec
    dec_err : float
        DEC uncertainty in arcsec
    x : float
        x coordinate in pixels
    y : float
        y coordinate in pixels
    x_err : float
        x uncertainty in pixels
    y_err : float
        y uncertainty in pixels
    SNR : float
        Signal-to-noise ratio
    flux : float
        Flux in nanojanskys
    flux_err : float
        Flux uncertainty in nanojanskys
    mag : float
        AB Magnitude
    mag_err : float
        AB Magnitude uncertainty
    separation : float
        Separation from target coordinates in arcsec
    sigma : float
        Sigma in arcsec based on distance from target coordinates
    flags : Dict[str, bool]
        Dictionary of flags (3 so far)
            - base_PsfFlux_flag_badCentroid
            - base_PsfFlux_flag_badCentroid_edge
            - slot_Centroid_flag_edge
    """

    ra: float
    dec: float
    ra_err: float
    dec_err: float
    x: float
    y: float
    x_err: float
    y_err: float
    snr: float
    flux: float
    flux_err: float
    mag: float
    mag_err: float
    separation: float
    sigma: float
    flags: dict[str, bool]


@dataclass
class EndResult:
    """
    Dataclass to hold the results of the end-to-end process.

    Attributes
    ----------
    target_name : str
        Name of the target
    target_type : str
        Type of the target
    image_type : str
        Type of the image (calexp, goodSeeingDiff_differenceExp)
    ephemeris_service :str
        Name of the ephemeris service
    visit_id : int
        LSST visit ID
    detector_id : int
        LSST detector ID
    band : str
        LSST filter band
    coordinates_central : Tuple[float, float]
        RA and Dec coordinates in degrees
    obs_time : Time
        Observation time (mid_exposure)
    cutout_size : int
        Size of the cutout in pixels
    image_name : str
        Name of the savedimage
    uncertainty : Dict[str, float]
        Dictionary containing uncertainty measurements
        rss : float
        smaa_3sig : float
        smia_3sig : float
        theta : float
    forced_phot_on_target : PhotometryResult
        Forced photometry on target coordinates
    phot_within_error_ellipse : List[PhotometryResult]
        Photometry of sources within the error ellipse
    """

    target_name: str
    target_type: str
    image_type: str
    ephemeris_service: str
    visit_id: int
    detector_id: int
    band: str
    coordinates_central: tuple[float, float]
    obs_time: Time
    cutout_size: int
    saved_image_name: str
    uncertainty: dict[str, float]
    forced_phot_on_target: PhotometryResult
    phot_within_error_ellipse: list[PhotometryResult]

    def to_csv_row(self) -> dict:
        """
        Convert EndResult instance to a flattened dictionary suitable for CSV export.
        
        Returns
        -------
        dict
            Flattened dictionary containing all EndResult data
        """
        # Base metadata
        row = {
            'target_name': self.target_name,
            'target_type': self.target_type,
            'image_type': self.image_type,
            'ephemeris_service': self.ephemeris_service,
            'visit_id': self.visit_id,
            'detector_id': self.detector_id,
            'band': self.band,
            'coordinates_central_ra': self.coordinates_central[0],
            'coordinates_central_dec': self.coordinates_central[1],
            'obs_time_mjd': self.obs_time.mjd if isinstance(self.obs_time, Time) else self.obs_time,
            'cutout_size': self.cutout_size,
            'saved_image_name': self.saved_image_name,
        }
        
        # Uncertainty data
        row.update({
            'uncertainty_rss': self.uncertainty.get('rss', None),
            'uncertainty_smaa': self.uncertainty.get('smaa', None),
            'uncertainty_smia': self.uncertainty.get('smia', None),
            'uncertainty_theta': self.uncertainty.get('theta', None),
        })
        
        # Forced photometry on target
        if self.forced_phot_on_target:
            forced_phot = self.forced_phot_on_target
            row.update({
                'forced_phot_ra': forced_phot.ra,
                'forced_phot_dec': forced_phot.dec,
                'forced_phot_ra_err': forced_phot.ra_err,
                'forced_phot_dec_err': forced_phot.dec_err,
                'forced_phot_x': forced_phot.x,
                'forced_phot_y': forced_phot.y,
                'forced_phot_x_err': forced_phot.x_err,
                'forced_phot_y_err': forced_phot.y_err,
                'forced_phot_snr': forced_phot.snr,
                'forced_phot_flux': forced_phot.flux,
                'forced_phot_flux_err': forced_phot.flux_err,
                'forced_phot_mag': forced_phot.mag,
                'forced_phot_mag_err': forced_phot.mag_err,
                'forced_phot_separation': forced_phot.separation,
                'forced_phot_sigma': forced_phot.sigma,
            })
            
            # Handle flags
            if forced_phot.flags:
                for flag_name, flag_value in forced_phot.flags.items():
                    row[f'forced_phot_flag_{flag_name}'] = flag_value
        
        # Sources within error ellipse (flatten to single best source or count)
        row['num_sources_in_ellipse'] = len(self.phot_within_error_ellipse)
        
        # If there are sources in the ellipse, add the first/best one
        if self.phot_within_error_ellipse:
            best_source = self.phot_within_error_ellipse[0]  # Assuming first is best
            row.update({
                'ellipse_source_ra': best_source.ra,
                'ellipse_source_dec': best_source.dec,
                'ellipse_source_ra_err': best_source.ra_err,
                'ellipse_source_dec_err': best_source.dec_err,
                'ellipse_source_x': best_source.x,
                'ellipse_source_y': best_source.y,
                'ellipse_source_x_err': best_source.x_err,
                'ellipse_source_y_err': best_source.y_err,
                'ellipse_source_snr': best_source.snr,
                'ellipse_source_flux': best_source.flux,
                'ellipse_source_flux_err': best_source.flux_err,
                'ellipse_source_mag': best_source.mag,
                'ellipse_source_mag_err': best_source.mag_err,
                'ellipse_source_separation': best_source.separation,
                'ellipse_source_sigma': best_source.sigma,
            })
            
            # Handle flags for best source
            if best_source.flags:
                for flag_name, flag_value in best_source.flags.items():
                    row[f'ellipse_source_flag_{flag_name}'] = flag_value
        
        return row
    
    
    def save_results_to_csv(
        results: Union[List['EndResult'], 'EndResult'], 
        output_file: Union[str, Path],
        include_all_ellipse_sources: bool = False
    ) -> None:
        """
        Save EndResult instances to CSV file.
        
        Parameters
        ----------
        results : List[EndResult] or EndResult
            Single EndResult or list of EndResult instances to save
        output_folder : str
            Output folder path
        include_all_ellipse_sources : bool, optional
            If True, create separate rows for each source within error ellipse.
            If False, only include the best source per result. Default is False.
        """
        # Ensure results is a list
        if not isinstance(results, list):
            results = [results]
        
        if not results:
            raise ValueError("No results provided")
        
        output_file = Path(output_file)
        
        # Collect all rows
        all_rows = []
        
        for result in results:
            if include_all_ellipse_sources and result.phot_within_error_ellipse:
                # Create separate row for each source in ellipse
                for i, source in enumerate(result.phot_within_error_ellipse):
                    row = result.to_csv_row()
                    
                    # Override the ellipse_source fields with current source
                    row.update({
                        'ellipse_source_index': i,
                        'ellipse_source_ra': source.ra,
                        'ellipse_source_dec': source.dec,
                        'ellipse_source_ra_err': source.ra_err,
                        'ellipse_source_dec_err': source.dec_err,
                        'ellipse_source_x': source.x,
                        'ellipse_source_y': source.y,
                        'ellipse_source_x_err': source.x_err,
                        'ellipse_source_y_err': source.y_err,
                        'ellipse_source_snr': source.snr,
                        'ellipse_source_flux': source.flux,
                        'ellipse_source_flux_err': source.flux_err,
                        'ellipse_source_mag': source.mag,
                        'ellipse_source_mag_err': source.mag_err,
                        'ellipse_source_separation': source.separation,
                        'ellipse_source_sigma': source.sigma,
                    })
                    
                    # Handle flags
                    if source.flags:
                        for flag_name, flag_value in source.flags.items():
                            row[f'ellipse_source_flag_{flag_name}'] = flag_value
                    
                    all_rows.append(row)
            else:
                # Standard mode - one row per result
                all_rows.append(result.to_csv_row())
        
        # Write to CSV
        if all_rows:
            fieldnames = all_rows[0].keys()
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
