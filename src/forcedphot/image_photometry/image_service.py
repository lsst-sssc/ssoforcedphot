"""
Image Service Module

This module provides functionality for searching images in the ObsCore catalog with support
for ECSV ephemeris data and photometric bands. It handles coordinate interpolation (linear)
and time-based filtering of astronomical observations.

Main Components:
    - EphemerisDataCompressed: Handles loading and processing of ephemeris data
    - ImageService: Manages image searching and result processing
"""

import logging
import time
from typing import Optional

from astropy.time import Time
from forcedphot.ephemeris.data_model import QueryResult
from forcedphot.image_photometry.utils import (
    EphemerisDataCompressed,
    ImageMetadata,
    interpolate_coordinates,
)
from lsst.rsp import get_tap_service


class ImageService:
    """
    Service for searching and processing images based on ephemeris data.
    Interfaces with the ObsCore catalog through TAP service.
    """
    def __init__(self):
        """Initializes the ImageService with logging configuration and TAP service connection."""
        self.logger = logging.getLogger("image_service")
        self.service = get_tap_service("tap")

    def search_images(self, bands, ephemeris_data):
        """
        Performs an image search based on provided ephemeris data and search parameters.

        Parameters
        ----------
        bands : list[str]
            list of photometric bands to search for in the image
        ephemeris_data : EphemerisDataCompressed or ephemeris file path
            Ephemeris data to search for in the image

        Returns
        -------
        Optional[list[ImageMetadata]]
            list of matching image metadata or None if no results found
        """
        print("Begin the image search based on ephemeris data.")
        # Check if ephemeris data is a string or a list of EphemerisDataCompressed objects
        try:
            time_start = time.time()
            # Load ephemeris data if it's a string path
            if isinstance(ephemeris_data, str):
                ephemeris_rows = EphemerisDataCompressed.load_ephemeris(ephemeris_data)
            # If it's a QueryResult object, convert it to a list of EphemerisDataCompressed objects
            elif isinstance(ephemeris_data, QueryResult):
                ephemeris_rows = EphemerisDataCompressed.compress_ephemeris(ephemeris_data)
            else:
                raise ValueError(
                    "Ephemeris_data must be a string or a list of EphemerisDataCompressed objects"
                )

            if not ephemeris_rows:
                self.logger.error("No ephemeris data found")
                return None

            start_time, end_time = EphemerisDataCompressed.get_time_range(ephemeris_rows)
            time_windows = EphemerisDataCompressed.get_time_windows(ephemeris_rows)
            combined_results = self._execute_query(
                self._build_query(start_time, end_time, time_windows, bands)
            )

            if combined_results is None or combined_results.empty:
                self.logger.warning("No results found")
                return None
            else:
                self.logger.info(f"Image search is done. Time: {time.time() - time_start}")

            return self._process_results(combined_results, ephemeris_rows)

        except Exception as e:
            self.logger.error(f"Error loading ephemeris data: {str(e)}")
            return None

    def _execute_query(self, query: str) -> Optional[object]:
        """
        Executes a TAP query and returns the results.

        Parameters
        ----------
        query : str
            The TAP query to execute

        Returns
        -------
        Optional[object]
            Query results as a pandas DataFrame or None if query fails
        """
        try:
            job = self.service.submit_job(query)
            job.run()
            job.wait(phases=['COMPLETED', 'ERROR'])
            self.logger.info(f'Job phase is {job.phase}')
            job.raise_if_error()

            return job.fetch_result().to_table().to_pandas()

        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            return None

    def _build_query(self, start_time: Time, end_time: Time,
                     time_windows: list[tuple[Time, Time, float, float]],
                     bands: set[str]) -> str:
        """
        Constructs a TAP query for image search based on time windows and bands.

        Parameters
        ----------
        start_time : Time
            Overall start time for the search
        end_time : Time
            Overall end time for the search
        time_windows : list[Tuple]
            list of time windows with coordinates
        bands : set[str]
            set of photometric bands to search for

        Returns
        -------
        str
            Constructed TAP query string
        """
        bands_clause = " OR ".join([f"lsst_band = '{band}'" for band in bands])

        coordinate_conditions = []
        for time_before, time_after, ra, dec in time_windows:
            condition = f"""
                (t_min >= {time_before.mjd}
                AND t_max <= {time_after.mjd}
                AND CONTAINS(POINT('ICRS', {ra}, {dec}), s_region) = 1)
            """
            coordinate_conditions.append(condition)

        coordinate_clause = " OR ".join(coordinate_conditions)

        query = f"""
        SELECT
            lsst_visit,
            lsst_detector,
            lsst_ccdvisitid,
            lsst_band,
            s_ra,
            s_dec,
            t_min,
            t_max,
            s_region
        FROM ivoa.ObsCore
        WHERE calib_level = 2
        AND ({bands_clause})
        AND t_max >= {start_time.mjd}
        AND t_min <= {end_time.mjd}
        AND ({coordinate_clause})
        """
        return query

    def _process_results(self, results, ephemeris_rows: list[EphemerisDataCompressed]):
        """
        Processes query results and creates image metadata objects.

        Parameters
        ----------
        results : pandas.DataFrame
            Query results as a pandas DataFrame
        ephemeris_rows : list[EphemerisDataCompressed]
            list of ephemeris data rows

        Returns
        -------
        list[ImageMetadata]
            list of processed image metadata objects
        """

        metadata_list = []

        for _, row in results.iterrows():
            t_min = Time(row['t_min'], format='mjd')
            t_max = Time(row['t_max'], format='mjd')
            t_mid = (t_min.mjd + t_max.mjd) / 2

            # Get relevant ephemeris rows for this image
            relevant_ephemeris = EphemerisDataCompressed.get_relevant_rows(ephemeris_rows, t_min, t_max)
            interpolated_ra, interpolated_dec, target_time = interpolate_coordinates(
                relevant_ephemeris[0].ra_deg, relevant_ephemeris[0].dec_deg,
                relevant_ephemeris[-1].ra_deg, relevant_ephemeris[-1].dec_deg,
                relevant_ephemeris[0].datetime.mjd, relevant_ephemeris[-1].datetime.mjd,
                t_mid
            )

            interpolated_row = EphemerisDataCompressed(
                datetime=Time(t_mid, format='mjd'),
                ra_deg=interpolated_ra,
                dec_deg=interpolated_dec,
                ra_rate=relevant_ephemeris[1].ra_rate,
                dec_rate=relevant_ephemeris[1].dec_rate,
                uncertainty={
                    'rss': relevant_ephemeris[1].uncertainty['rss'],
                    'smaa': relevant_ephemeris[1].uncertainty['smaa'],
                    'smia': relevant_ephemeris[1].uncertainty['smia'],
                    'theta': relevant_ephemeris[1].uncertainty['theta']
                }
            )

            metadata = ImageMetadata(
                visit_id=int(row['lsst_visit']),
                detector_id=int(row['lsst_detector']),
                ccdvisit=int(row['lsst_ccdvisitid']),
                band=row['lsst_band'],
                coordinates_central=(float(row['s_ra']), float(row['s_dec'])),
                t_min=t_min,
                t_max=t_max,
                ephemeris_data=relevant_ephemeris,
                exact_ephemeris=interpolated_row
            )
            metadata_list.append(metadata)

        metadata_unique = list({item.ccdvisit: item for item in metadata_list}.values())
        self.logger.info(f"Found {len(metadata_unique)} image(s) ")

        print("-" * 40)
        print("Image Service is done.")
        print("-" * 40)

        return metadata_unique
