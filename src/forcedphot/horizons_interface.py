import numpy as np
import logging
import time
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from local_dataclasses import QueryInput, EphemerisData, QueryResult

class HorizonsInterface:
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Rubin location
    DEFAULT_OBSERVER_LOCATION = 'X05'

    def __init__(self, observer_location=DEFAULT_OBSERVER_LOCATION):
        """
        Initialize the HorizonsInterface with an optional observer location.

        Parameters:
        observer_location (str): The observer location code. Default is 'X05' (Rubin location).
        """
        self.observer_location = observer_location


    def query_single_range(self, query: QueryInput) -> QueryResult:
        """
        Query ephemeris for a single time range.

        Parameters:
        query (QueryInput): The query parameters.

        Returns:
        QueryResult: The queried ephemeris data.
        """
        try:
            start_time = time.time()
            obj = Horizons(id_type='smallbody', id=query.target, location=self.observer_location, 
                           epochs={'start': query.start.iso, 'stop': query.end.iso, 'step': query.step})
            ephemeris = obj.ephemerides()
            end_time = time.time()
            self.logger.info(f"Query for range {query.start} to {query.end} successful for target \
                                          {query.target}. Time taken: {end_time - start_time:.2f} seconds.")

            ephemeris_data = EphemerisData(
                datetime_jd=Time(ephemeris['datetime_jd'], format='jd'),
                datetime_iso=Time(Time(ephemeris['datetime_jd'], format='jd').iso, format='iso'),
                RA_deg=np.array(ephemeris['RA']),
                DEC_deg=np.array(ephemeris['DEC']),
                RA_rate_arcsec_per_h=np.array(ephemeris['RA_rate']),
                DEC_rate_arcsec_per_h=np.array(ephemeris['DEC_rate']),
                AZ_deg=np.array(ephemeris['AZ']),
                EL_deg=np.array(ephemeris['EL']),
                r_au=np.array(ephemeris['r']),
                delta_au=np.array(ephemeris['delta']),
                V_mag=np.array(ephemeris['V']),
                alpha_deg=np.array(ephemeris['alpha']),
                RSS_3sigma_arcsec=np.array(ephemeris['RSS_3sigma'])
            )

            return QueryResult(query.target, query.start, query.end, ephemeris_data)
        except Exception as e:
            self.logger.error(f"An error occurred during query for range {query.start} to {query.end} for target {query.target}: {e}")
            return None
    
    @classmethod
    def query_ephemeris_from_csv(cls, csv_filename: str, observer_location=DEFAULT_OBSERVER_LOCATION):
        """
        Query ephemeris for multiple celestial objects from JPL Horizons based on a CSV file and save the data to ECSV files.

        Parameters:
        csv_filename (str): The filename of the input CSV file containing target, start time, end time, and step.
        observer_location (str): The observer location code. Default is 'X05' (Rubin location).
        """
        try:
            total_start_time = time.time()
            # Read the CSV file
            df = pd.read_csv(csv_filename)

            # Create HorizonsInterface instance with the specified observer location
            horizons_interface = cls(observer_location)

            # Process each row in the CSV file
            for _index, row in df.iterrows():
                query = QueryInput(
                    target=row.iloc[0],
                    start=Time(row.iloc[1], scale='utc'),
                    end=Time(row.iloc[2], scale='utc'),
                    step=row.iloc[3]
                )

                # Calculate the total number of instances
                total_days = (query.end - query.start).jd
                step_hours = float(query.step[:-1])  
                step_days = step_hours / 24.0  
                max_instances = 10000
                step_instances = total_days / step_days

                # Check if multiple queries are needed
                if step_instances > max_instances:
                    cls.logger.info(f"Total instances exceed 10,000 for target {query.target}. Splitting the queries.")

                    time_splits = int(step_instances // max_instances) + 1
                    time_ranges = [(query.start + i * (total_days / time_splits) * u.day,
                                    query.start + (i + 1) * (total_days / time_splits) * u.day)
                                   for i in range(time_splits)]
                else:
                    time_ranges = [(query.start, query.end)]

                all_ephemeris = EphemerisData()

                # Run queries sequentially
                for start, end in time_ranges:
                    result = horizons_interface.query_single_range(QueryInput(query.target, start, end, query.step))
                    if result is not None:
                        all_ephemeris.datetime_jd = Time(np.concatenate((all_ephemeris.datetime_jd.jd, result.ephemeris.datetime_jd.jd)), format='jd')
                        all_ephemeris.datetime_iso = Time(np.concatenate((all_ephemeris.datetime_iso.iso, result.ephemeris.datetime_iso.iso)), format='iso')
                        all_ephemeris.RA_deg = np.concatenate((all_ephemeris.RA_deg, result.ephemeris.RA_deg))
                        all_ephemeris.DEC_deg = np.concatenate((all_ephemeris.DEC_deg, result.ephemeris.DEC_deg))
                        all_ephemeris.RA_rate_arcsec_per_h = np.concatenate((all_ephemeris.RA_rate_arcsec_per_h, result.ephemeris.RA_rate_arcsec_per_h))
                        all_ephemeris.DEC_rate_arcsec_per_h = np.concatenate((all_ephemeris.DEC_rate_arcsec_per_h, result.ephemeris.DEC_rate_arcsec_per_h))
                        all_ephemeris.AZ_deg = np.concatenate((all_ephemeris.AZ_deg, result.ephemeris.AZ_deg))
                        all_ephemeris.EL_deg = np.concatenate((all_ephemeris.EL_deg, result.ephemeris.EL_deg))
                        all_ephemeris.r_au = np.concatenate((all_ephemeris.r_au, result.ephemeris.r_au))
                        all_ephemeris.delta_au = np.concatenate((all_ephemeris.delta_au, result.ephemeris.delta_au))
                        all_ephemeris.V_mag = np.concatenate((all_ephemeris.V_mag , result.ephemeris.V_mag ))
                        all_ephemeris.alpha_deg = np.concatenate((all_ephemeris.alpha_deg, result.ephemeris.alpha_deg))
                        all_ephemeris.RSS_3sigma_arcsec = np.concatenate((all_ephemeris.RSS_3sigma_arcsec, result.ephemeris.RSS_3sigma_arcsec))

                # Convert to pandas DataFrame
                relevant_data = pd.DataFrame({
                    'datetime_jd': all_ephemeris.datetime_jd.jd,
                    'datetime_iso': all_ephemeris.datetime_iso.iso,
                    'RA': all_ephemeris.RA_deg,
                    'DEC': all_ephemeris.DEC_deg,
                    'RA_rate': all_ephemeris.RA_rate_arcsec_per_h,
                    'DEC_rate': all_ephemeris.DEC_rate_arcsec_per_h,
                    'AZ': all_ephemeris.AZ_deg,
                    'EL': all_ephemeris.EL_deg,
                    'r': all_ephemeris.r_au,
                    'delta': all_ephemeris.delta_au,
                    'V': all_ephemeris.V_mag,
                    'alpha': all_ephemeris.alpha_deg,
                    'RSS_3sigma': all_ephemeris.RSS_3sigma_arcsec
                })

                # Generate output filename
                output_filename = f"{query.target}_{query.start.iso}_{query.end.iso}.csv".replace(":", "-").replace(" ", "_")

                # Save the data to a CSV file
                relevant_data.to_csv(output_filename, index=False)
                cls.logger.info(f"Ephemeris data successfully saved to {output_filename}")

            total_end_time = time.time()
            cls.logger.info(f"Total time taken for processing the CSV file: {total_end_time - total_start_time:.2f} seconds.")

        except Exception as e:
            cls.logger.error(f"An error occurred while processing the CSV file: {e}")

# Example usage
if __name__ == "__main__":
    HorizonsInterface.query_ephemeris_from_csv('targets.csv')

    # Different observer location
    # HorizonsInterface.query_ephemeris_from_csv('targets.csv', observer_location='500')  # Example: Geocentric