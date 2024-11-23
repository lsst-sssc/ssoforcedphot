"""
Image Service Module provides functionality for searching in the ObsCore catalog
with support for ECSV ephemeris data and photometric bands.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Dict
from astropy.time import Time
from astropy.io import ascii
from astropy.table import Table
import numpy as np
from lsst.rsp import get_tap_service
import pandas as pd
import time
from utils import EphemerisRow, SearchParameters, ImageMetadata


class EphemerisData:
    """Class to handle ephemeris data from ECSV files."""

    @staticmethod
    def load_ephemeris(file_path: str) -> List[EphemerisRow]:
        """
        Load ephemeris data from ECSV file.

        Parameters
        ----------
        file_path : str
            Path to the ECSV file

        Returns
        -------
        List[EphemerisRow]
            List of ephemeris data rows
        """
        try:
            data = Table.read(file_path, format="ascii.ecsv")
            rows = []
            
            for row in data:
                ephemeris_row = EphemerisRow(
                    datetime=Time(row['datetime'], scale="utc", format="mjd"),
                    ra_deg=float(row['RA_deg']),
                    dec_deg=float(row['DEC_deg']),
                    ra_rate=float(row['RA_rate_arcsec_per_h']),
                    dec_rate=float(row['DEC_rate_arcsec_per_h']),
                    uncertainty={
                        'rss': float(row['RSS_3sigma_arcsec']),
                        'smaa': float(row['SMAA_3sigma_arcsec']),
                        'smia': float(row['SMIA_3sigma_arcsec']),
                        'theta': float(row['Theta_3sigma_deg'])
                    }
                )
                rows.append(ephemeris_row)
            
            return rows
            
        except Exception as e:
            raise ValueError(f"Error loading ephemeris file: {str(e)}")

    

    
    @staticmethod
    def get_full_time_range(rows: List[EphemerisRow]) -> Tuple[Time, Time]:
        """
        Get full time range from first to last ephemeris entry.

        Parameters
        ----------
        rows : List[EphemerisRow]
            List of ephemeris rows

        Returns
        -------
        Tuple[Time, Time]
            Start and end times for full range
        """
        return rows[0].datetime, rows[-1].datetime

    @staticmethod
    def get_time_windows(rows: List[EphemerisRow]) -> List[Tuple[Time, Time, float, float]]:
        """
        Get time windows and coordinates for each ephemeris row.

        Parameters
        ----------
        rows : List[EphemerisRow]
            List of ephemeris rows

        Returns
        -------
        List[Tuple[Time, Time, float, float]]
            List of (start_time, end_time, ra, dec) for each row
        """
        windows = []
        for i, row in enumerate(rows):
            # Get start time from previous row or current row if first
            if i > 0:
                start_time = rows[i-1].datetime
            else:
                start_time = row.datetime
                
            # Get end time from next row or current row if last
            if i < len(rows) - 1:
                end_time = rows[i+1].datetime
            else:
                end_time = row.datetime
                
            windows.append((start_time, end_time, row.ra_deg, row.dec_deg))
        
        return windows


class ImageService:
    """
    A service class for searching and preparing astronomical image data.
    
    This class provides methods for querying the LSST RSP service using
    ephemeris data and user-defined bands.
    """

    def __init__(self):
        """Initialize the ImageService."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.service = get_tap_service("tap")

    def search_images(self, params: SearchParameters) -> Optional[List[ImageMetadata]]:
        """
        Search for images based on ephemeris data and user parameters.

        Parameters
        ----------
        params : SearchParameters
            User-defined search parameters

        Returns
        -------
        Optional[List[ImageMetadata]]
            List of image metadata objects, or None if no results found
        """
        try:
            # Load ephemeris data
            ephemeris_rows = EphemerisData.load_ephemeris(params.ephemeris_file)
            if not ephemeris_rows:
                self.logger.error("No ephemeris data found")
                return None

            # Get full time range and time windows
            start_time, end_time = EphemerisData.get_full_time_range(ephemeris_rows)
            time_windows = EphemerisData.get_time_windows(ephemeris_rows)
            
            time_start = time.time()
            combined_results = self._execute_query(
                self._build_combined_query(start_time, end_time, time_windows, params.bands)
            )
            
            if combined_results is None or combined_results.empty:
                self.logger.warning("No results found")
                return None
            else:
                self.logger.info(f"Search is done. Time: {time.time() - time_start}")
            
            return self._process_results(combined_results)

        except Exception as e:
            self.logger.error(f"Error during image search: {str(e)}")
            return None

    def _execute_query(self, query: str) -> Optional[object]:
        """
        Execute a TAP query and return results.

        Parameters
        ----------
        query : str
            Query string to execute

        Returns
        -------
        Optional[object]
            Query results or None if error occurs
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

    
    def _build_combined_query(self, start_time: Time, end_time: Time, time_windows: List[Tuple[Time, Time, float, float]],
                              bands: Set[str]) -> str:
        """
        Build the TAP query string to filter by time range (whole dataset), bands and coordinates for each time window.

        Parameters
        ----------
        start_time : Time
            Start of observation time range
        end_time : Time
            End of observation time range
        bands : Set[str]
            Set of filter bands to search for
        time_windows : List[Tuple[Time, Time, float, float]]
            List of time windows and coordinates
        preliminary_results : object
            Results from preliminary query

        Returns
        -------
        str
            Formatted final query string
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
        AND t_max >= {start_time.mjd} 
        AND t_min <= {end_time.mjd}
        AND ({bands_clause})
        AND ({coordinate_clause})
        """
        # print(query)
        return query
        
    
    def _process_results(self, results) -> List[ImageMetadata]:
        """
        Process query results into ImageMetadata objects.

        Parameters
        ----------
        results : pandas.DataFrame
            Query results

        Returns
        -------
        List[ImageMetadata]
            List of processed image metadata
        """
        metadata_list = []
        
        for _, row in results.iterrows():
            metadata = ImageMetadata(
                visit_id=int(row['lsst_visit']),
                detector_id=int(row['lsst_detector']),
                ccdvisit=int(row['lsst_ccdvisitid']),
                band=row['lsst_band'],
                coordinates_central=(float(row['s_ra']), float(row['s_dec'])),
                obs_time=Time(row['t_min'], format='mjd')
            )
            metadata_list.append(metadata)
            
            # self.logger.info(
            #     f"Found image: Visit={metadata.visit_id}, "
            #     f"Detector={metadata.detector_id}, Band={metadata.band}, "
            #     f"Time={metadata.obs_time.iso}"
            # )
        df = pd.DataFrame(metadata_list)
        # df_unique = df.drop_duplicates(subset=['ccdvisit'], inplace=False)
        print(df)

        metadata_unique = list({item.ccdvisit: item for item in metadata_list}.values())
        # print(metadata_unique)
        return df


def main():
    """Example usage of the ImageService with ECSV data."""
    # Example search parameters
    params = SearchParameters(
        bands={'r', 'i', 'g', 'u', 'z', 'y'},
        ephemeris_file='./test_eph2.ecsv'
    )
    start_time = time.time()
    
    # Initialize service and perform search
    image_service = ImageService()
    results = image_service.search_images(params)

    print(f"Time: {time.time() - start_time}")
    
    # Display results
    # if results:
    #     print("\nFound Images:")
    #     print("-" * 70)
    #     for img in results:
    #         print(f"Visit ID: {img.visit_id}")
    #         print(f"Detector ID: {img.detector_id}")
    #         print(f"Band: {img.band}")
    #         print(f"Coordinates: (RA={img.coordinates[0]:.6f}, Dec={img.coordinates[1]:.6f})")
    #         print(f"Observation Time: {img.obs_time.iso}")
    #         print("-" * 70)


if __name__ == "__main__":
    main()