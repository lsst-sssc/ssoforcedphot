from dataclasses import dataclass
from astropy.time import Time
from typing import List, Tuple, Set, Dict


@dataclass
class EphemerisRow:
    """
    Dataclass to hold a single row of ephemeris data.

    Attributes
    ----------
    datetime : Time
        Observation datetime
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    ra_rate : float
        RA rate in arcsec/hour
    dec_rate : float
        DEC rate in arcsec/hour
    uncertainty : Dict[str, float]
        Dictionary containing uncertainty measurements
    """
    datetime: Time
    ra_deg: float
    dec_deg: float
    ra_rate: float
    dec_rate: float
    uncertainty: Dict[str, float]


@dataclass
class SearchParameters:
    """
    Dataclass to hold user-defined search parameters.

    Attributes
    ----------
    bands : Set[str]
        Set of filter bands to search for
    ephemeris_file : str
        Path to the ECSV file containing ephemeris data
    """
    bands: Set[str]
    ephemeris_file: str


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
    obs_time : Time
        Observation time
    """
    visit_id: int
    detector_id: int
    ccdvisit: int
    band: str
    coordinates_central: Tuple[float, float]
    obs_time: Time


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
