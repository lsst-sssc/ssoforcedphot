import logging
import time

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from forcedphot.ephemeris.local_dataclasses import EphemerisData, QueryInput, QueryResult


class HorizonsInterface:
    """
    A class for querying ephemeris data from JPL Horizons for celestial objects.

    This class provides methods to query ephemeris data for single time ranges
    or multiple objects from a CSV file. It uses the astropy Horizons module
    to interact with the JPL Horizons system.

    Attributes:
        DEFAULT_OBSERVER_LOCATION (str): The default observer location code ("X05" for Rubin).
        logger (logging.Logger): Logger for the class.
        observer_location (str): The observer location code used for queries.

    Methods:
        splitting_query(query: QueryInput) -> List[time_ranges]:
            Split a query into multiple time ranges to avoid exceeding the limit.

        query_single_range(query: QueryInput) -> QueryResult:
            Query ephemeris for a single time range.

        query_ephemeris_from_csv(csv_filename: str, observer_location=DEFAULT_OBSERVER_LOCATION):
            Query ephemeris for multiple celestial objects from a CSV file and save results to ECSV files.

    The class handles large queries by splitting them into smaller time ranges
    when necessary to avoid exceeding JPL Horizons limit of 10,000 instances per query."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Rubin location
    DEFAULT_OBSERVER_LOCATION = "X05"

    def __init__(self, observer_location=DEFAULT_OBSERVER_LOCATION):
        """
        Initialize the HorizonsInterface with an optional observer location.

        Parameters:
        ----------
        observer_location : str, optional
            The observer location code. Default is "X05" (Rubin location).
        """
        self.observer_location = observer_location

    def splitting_query(self, query: QueryInput, max_instances=10000):
        """
        Split a query into multiple time ranges if the total number of instances exceeds a specified maximum.

        This method calculates the total number of instances based on the query"s start and end dates
        and the step size. If the number of instances exceeds the specified maximum, it splits the
        query into multiple time ranges.

        Parameters:
        -----------
        query : QueryInput
            The input query containing start date, end date, step size, and target information.
        max_instances : int, optional
            The maximum number of instances allowed in a single query (default is 10000).

        Returns:
        --------
        list of tuple
            A list of time ranges, where each range is a tuple of (start_time, end_time).
            If splitting is not necessary, the list contains a single tuple with the original
            start and end times.

        Raises:
        -------
        ValueError
            If the step unit is not recognized ("s", "m", "h", or "d").

        Notes:
        ------
        - The method assumes that the query object has attributes: start, end, step, and target.
        - The step size is expected to be a string with a number followed by a unit (e.g., "1h" for 1 hour).
        - Supported step units are:
            "s" for seconds
            "m" for minutes
            "h" for hours
            "d" for days
        - Time ranges are calculated using astropy's Time objects and are returned as such.
        - The method uses astropy units (u) for time calculations.
        """
        # Define the step frequency with astropy units
        value, unit = int(query.step[:-1]), query.step[-1]
        if unit == "s":
            step_freqency = value * u.s
        elif unit == "m":
            step_freqency = value * u.min
        elif unit == "h":
            step_freqency = value * u.hour
        elif unit == "d":
            step_freqency = value * u.day
        else:
            raise ValueError("Error in the input field.")

        # Calculate the total number of instances
        total_days = (query.end - query.start).jd
        step_instances = int(total_days * 86400 / step_freqency.to(u.s).value)

        # Check if multiple queries are needed
        if step_instances > max_instances:
            self.logger.info(
                f"Total instances exceed {max_instances} for target {query.target}. Number"
                f"of instances: {step_instances}. Splitting the queries."
            )

            time_splits = int(step_instances // max_instances) + 1
            time_ranges = [
                (
                    query.start + i * (total_days / time_splits) * u.day,
                    query.start + (i + 1) * (total_days / time_splits) * u.day,
                )
                for i in range(time_splits)
            ]
        else:
            time_ranges = [(query.start, query.end)]

        return time_ranges

    def query_single_range(self, query: QueryInput) -> QueryResult:
        """
        Query ephemeris for a single time range.

        Parameters
        ----------
        query : QueryInput
            The query parameters containing target, start time, end time, and step.

        Returns
        -------
        QueryResult or None
            The queried ephemeris data wrapped in a QueryResult object if successful,
            or None if an error occurs.

        Raises
        ------
        Exception
            If an error occurs during the query process. The error is logged,
            but not re-raised.

        Notes
        ------
        - If a query would exceed 10,000 instances, it is automatically split into multiple queries.
        """
        try:
            # Split the query into smaller time ranges if necessary
            time_ranges = self.splitting_query(query, max_instances=10000)

            all_ephemeris = []

            for start, end in time_ranges:
                start_time = time.time()
                obj = Horizons(
                    id_type="smallbody",
                    id=query.target,
                    location=self.observer_location,
                    epochs={"start": start.iso, "stop": end.iso, "step": query.step},
                )
                ephemeris = obj.ephemerides()
                if ephemeris is not None:
                    all_ephemeris.append(ephemeris)

                end_time = time.time()

                self.logger.info(
                    f"Query for range {start} to {end} successful for target "
                    f"{query.target}. Time taken: {end_time - start_time:.2f} seconds."
                )
            # Combine the results if multiple queries were made
            combined_ephemeris = vstack(all_ephemeris)

            ephemeris_data = EphemerisData(
                datetime_jd=Time(combined_ephemeris["datetime_jd"], format="jd"),
                RA_deg=np.array(combined_ephemeris["RA"]),
                DEC_deg=np.array(combined_ephemeris["DEC"]),
                RA_rate_arcsec_per_h=np.array(combined_ephemeris["RA_rate"]),
                DEC_rate_arcsec_per_h=np.array(combined_ephemeris["DEC_rate"]),
                AZ_deg=np.array(combined_ephemeris["AZ"]),
                EL_deg=np.array(combined_ephemeris["EL"]),
                r_au=np.array(combined_ephemeris["r"]),
                delta_au=np.array(combined_ephemeris["delta"]),
                V_mag=np.array(combined_ephemeris["V"]),
                alpha_deg=np.array(combined_ephemeris["alpha"]),
                RSS_3sigma_arcsec=np.array(combined_ephemeris["RSS_3sigma"]),
            )

            return QueryResult(query.target, query.start, query.end, ephemeris_data)
        except Exception as e:
            self.logger.error(
                f"An error occurred during query for range {query.start} to {query.end}"
                f"for target {query.target}"
            )
            self.logger.error(f"Error details: {str(e)}")

            return None

    @classmethod
    def query_ephemeris_from_csv(
        cls, csv_filename: str, observer_location=DEFAULT_OBSERVER_LOCATION,save_data=False
    ):
        """
        Query ephemeris for multiple celestial objects from JPL Horizons based on a CSV file and save
        the data to CSV files.

        Parameters
        ----------
        csv_filename : str
            The filename of the input CSV file containing target, start time, end time, and step.
        observer_location : str, optional
            The observer location code. Default is "X05" (Rubin location).
        save_data : bool, optional
            Whether to save the data to ECSV files. Default is False.

        Returns
        -------
        List of QueryResult or None
            The queried ephemeris data wrapped in a QueryResult object if successful,
            or None if an error occurs. Also, the method saves the data to ECSV files.

        Raises
        ------
        Exception
            If an error occurs during the ECSV processing or querying. The error is logged,
            but not re-raised.

        Notes
        -----
        - The input CSV file should have columns for target, start time, end time, and step.
        - The method creates a separate ECSV file for each target in the input file.
        - The method logs information about the query process and any errors that occur.
        """
        try:
            total_start_time = time.time()

            # Create an empty list to store the results
            results = []
            # Read the CSV file
            df = pd.read_csv(csv_filename)

            # Create HorizonsInterface instance with the specified observer location
            horizons_interface = cls(observer_location)

            # Process each row in the CSV file
            for _index, row in df.iterrows():
                query = QueryInput(
                    target=row.iloc[0],
                    start=Time(row.iloc[1], scale="utc"),
                    end=Time(row.iloc[2], scale="utc"),
                    step=row.iloc[3],
                )

                # Initialze the query
                query_result = horizons_interface.query_single_range(query)

                if query_result is not None:
                    # Append the result to the list
                    results.append(query_result)

                if save_data:
                    # Convert to pandas DataFrame
                    relevant_data = pd.DataFrame(
                        {
                            "datetime_jd": query_result.ephemeris.datetime_jd.jd,
                            "RA": query_result.ephemeris.RA_deg,
                            "DEC": query_result.ephemeris.DEC_deg,
                            "RA_rate": query_result.ephemeris.RA_rate_arcsec_per_h,
                            "DEC_rate": query_result.ephemeris.DEC_rate_arcsec_per_h,
                            "AZ": query_result.ephemeris.AZ_deg,
                            "EL": query_result.ephemeris.EL_deg,
                            "r": query_result.ephemeris.r_au,
                            "delta": query_result.ephemeris.delta_au,
                            "V": query_result.ephemeris.V_mag,
                            "alpha": query_result.ephemeris.alpha_deg,
                            "RSS_3sigma": query_result.ephemeris.RSS_3sigma_arcsec,
                        }
                    )

                    # Generate output filename
                    output_filename = f"{query.target}_{query.start.iso}_{query.end.iso}.ecsv".replace(
                        ":", "-"
                    ).replace(" ", "_")

                    # Save the data to an ECSV file
                    result_table = Table.from_pandas(relevant_data)
                    result_table.write("./data/" + output_filename, format="ascii.ecsv", overwrite=True)
                    cls.logger.info(f"Ephemeris data successfully saved to {output_filename}")

            total_end_time = time.time()
            cls.logger.info(
                f"Total time taken for processing the ECSV file:"
                f"{total_end_time - total_start_time:.2f} seconds."
            )
            return results

        except Exception as e:
            cls.logger.error(f"An error occurred while processing the ECSV file: {e}")


# Example usage
if __name__ == "__main__":
    # HorizonsInterface.query_ephemeris_from_csv("./data/targets.csv")

    # Define the target query parameters
    target_query = QueryInput(
        target="Ceres",
        start=Time("2024-01-01 00:00"),
        end=Time("2025-12-31 23:59"),
        step="1h",
    )
    horizons = HorizonsInterface()
    result = horizons.query_single_range(query=target_query)
