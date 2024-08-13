from dataclasses import dataclass, field

import numpy as np
from astropy.time import Time


@dataclass
class QueryInput:
    """
    A data class representing the input parameters for an ephemeris query.

    Attributes:
        target (str): The name or identifier of the celestial target.
        target_type (str): The type of celestial target.
        start (Time): The start time of the query period.
        end (Time): The end time of the query period.
        step (str): The time step interval for the query results.
    """

    target: str
    target_type: str
    start: Time
    end: Time
    step: str


@dataclass
class QueryInputMiriade:
    """
    A data class representing the input parameters for the Miriade ephemeris query.

    Attributes:
        target (str): The name or identifier of the celestial target.
        objtype (str): The type of celestial target.
        start (Time): The start time of the query period.
        step (str): The time step interval for the query results.
        nsteps (int): The number of steps in the query.
    """

    target: str
    objtype: str
    start: Time
    step: str
    nsteps: int


@dataclass
class EphemerisData:
    """
    A data class representing the ephemeris data for a celestial object.

    Attributes:
        datetime (Time): Time for the ephemeris data points.
        RA_deg (np.ndarray): Right Ascension in degrees.
        DEC_deg (np.ndarray): Declination in degrees.
        RA_rate_arcsec_per_h (np.ndarray): Rate of change of Right Ascension in arcseconds per hour.
        DEC_rate_arcsec_per_h (np.ndarray): Rate of change of Declination in arcseconds per hour.
        AZ_deg (np.ndarray): Azimuth in degrees.
        EL_deg (np.ndarray): Elevation in degrees.
        r_au (np.ndarray): Heliocentric distance in astronomical units.
        delta_au (np.ndarray): Geocentric distance in astronomical units.
        V_mag (np.ndarray): Visual magnitude.
        alpha_deg (np.ndarray): Phase angle in degrees.
        RSS_3sigma_arcsec (np.ndarray): Root Sum Square 3-sigma positional uncertainty in arcseconds.
    """

    datetime: Time = field(default_factory=lambda: Time([], scale="utc", format="jd"))
    RA_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    DEC_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    RA_rate_arcsec_per_h: np.ndarray = field(default_factory=lambda: np.array([]))
    DEC_rate_arcsec_per_h: np.ndarray = field(default_factory=lambda: np.array([]))
    AZ_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    EL_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    r_au: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_au: np.ndarray = field(default_factory=lambda: np.array([]))
    V_mag: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    RSS_3sigma_arcsec: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class QueryResult:
    """
    A data class representing the result of an ephemeris query.

    Attributes:
        target (str): The name or identifier of the celestial target.
        start (Time): The start time of the query period.
        end (Time): The end time of the query period.
        ephemeris (EphemerisData): The ephemeris data for the celestial object.
    """

    target: str
    start: Time
    end: Time
    ephemeris: EphemerisData
