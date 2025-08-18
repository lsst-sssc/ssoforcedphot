# polygon.py
import astropy.units as u
from astropy.coordinates import SkyCoord
from image_photometry.utils import EphemerisDataCompressed, QueryResult


def calculate_polygons(ephemeris_data: list[EphemerisDataCompressed], time_interval: float, widening: float):
    """
    Calculates sky polygons for time-sliced segments of an ephemeris list.

    This function iterates through a list of ephemeris points, divides it into
    segments based on the specified time interval, and generates a covering
    rectangular polygon for each segment.

    Args:
        ephemeris_data (list[EphemerisDataCompressed]):
            A list of EphemerisDataCompressed objects or QueryResults object.
        time_interval (float):
            The maximum duration for each polygon segment in days.
        widening (float):
            The desired width of the polygon on either side of the path,
            in arcseconds.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a
        calculated polygon and contains:
            - "time_start" (str): ISO timestamp of the segment"s start.
            - "time_end" (str): ISO timestamp of the segment"s end.
            - "polygon_corners" (list): A list of (RA, Dec) tuples in degrees
              for the four corners of the polygon.
    """
    # --- 1. Input Validation ---
    if isinstance(ephemeris_data, QueryResult):
        ephemeris_rows = EphemerisDataCompressed.compress_ephemeris(ephemeris_data)
    else:
        ephemeris_rows = ephemeris_data

    if not ephemeris_data or len(ephemeris_rows) < 2:
        raise ValueError("Ephemeris data must contain at least two points.")

    all_polygons = []
    current_index = 0

    # --- 2. Iterate Through Ephemeris Data in Time Slices ---
    while current_index < len(ephemeris_rows):
        # --- 2a. Define the current time slice ---
        start_point = ephemeris_rows[current_index]
        start_time = start_point.datetime
        target_end_time = start_time + time_interval * u.day

        # Find all points within this time interval, starting from the current_index
        segment_points = [p for p in ephemeris_rows[current_index:] if p.datetime <= target_end_time]

        # If no points are found besides the start, it means we are at the end.
        # We take the last two points to form the final segment.
        if len(segment_points) < 2:
            if len(ephemeris_rows) - current_index < 2:
                break
            segment_points = ephemeris_rows[-2:]
            end_point = segment_points[-1]
            current_index = len(ephemeris_rows)
        else:
            end_point = segment_points[-1]
            # Find the index of the end_point to set up the next iteration
            # This is safe because segment_points is a slice of the main list
            end_point_global_index = ephemeris_rows.index(end_point)
            current_index = end_point_global_index + 1


        # --- 2b. Calculate Polygon for the current segment ---
        a = SkyCoord(ra=start_point.ra_deg, dec=start_point.dec_deg, unit="deg", frame="icrs")
        b = SkyCoord(ra=end_point.ra_deg, dec=end_point.dec_deg, unit="deg", frame="icrs")

        separation_ab = a.separation(b)
        pa_ab = a.position_angle(b) if separation_ab > 1e-10 * u.arcsec else 0.0 * u.deg

        # Extend the path segment slightly at both ends
        extension = widening * u.arcsec
        a_ext = a.directional_offset_by(pa_ab - 180 * u.deg, extension)
        b_ext = b.directional_offset_by(pa_ab, extension)

        separation_ext = a_ext.separation(b_ext)
        pa_rect = a_ext.position_angle(b_ext) if separation_ext > 1e-10 * u.arcsec else 0.0 * u.deg

        corner1 = a_ext.directional_offset_by(pa_rect + 90 * u.deg, extension)
        corner2 = a_ext.directional_offset_by(pa_rect - 90 * u.deg, extension)
        corner3 = b_ext.directional_offset_by(pa_rect - 90 * u.deg, extension)
        corner4 = b_ext.directional_offset_by(pa_rect + 90 * u.deg, extension)

        polygon_corners = [
            (corner1.ra.deg, corner1.dec.deg),
            (corner4.ra.deg, corner4.dec.deg),
            (corner3.ra.deg, corner3.dec.deg),
            (corner2.ra.deg, corner2.dec.deg),
        ]

        # --- 2c. Store the calculated polygon ---
        all_polygons.append(
            {
                "time_start": start_point.datetime.iso,
                "time_end": end_point.datetime.iso,
                "polygon_corners": polygon_corners
            }
        )

    return all_polygons
