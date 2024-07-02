import logging
import time
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons

""" 
Setting up dataclasses for the query input and the ephemeris data.
"""


@dataclass
class QueryInput:
    target: str
    start: Time
    end: Time
    step: str


@dataclass
class EphemerisData:
    datetime_jd: Time = field(default_factory=lambda: Time([], format="jd"))
    datetime_iso: Time = field(default_factory=lambda: Time([], format="iso"))
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
    target: str
    start: Time
    end: Time
    ephemeris: EphemerisData


class HorizonsInterface:
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Rubin location
    OBSERVER_LOCATION = "X05"

    @staticmethod
    def query_single_range(query: QueryInput) -> QueryResult:
        """
        Executes a query to the JPL Horizons system to retrieve ephemeris data for a specified target within a given date range.

        Parameters:
        -----------
        query : QueryInput
            An instance of QueryInput containing the following attributes:
            - target: The identifier of the astronomical object.
            - start: The start date of the query range (as a datetime object).
            - end: The end date of the query range (as a datetime object).
            - step: The step size for the query epochs (as a string, e.g., '1d' for 1 day).

        Returns:
        --------
        QueryResult
            An instance of QueryResult containing the following attributes:
            - target: The identifier of the astronomical object.
            - start: The start date of the query range.
            - end: The end date of the query range.
            - ephemeris_data: An instance of EphemerisData containing the ephemeris information:
                - datetime_jd: Julian Date times of the ephemeris.
                - datetime_iso: ISO format times of the ephemeris.
                - RA_deg: Right Ascension in degrees.
                - DEC_deg: Declination in degrees.
                - RA_rate_arcsec_per_h: Right Ascension rate in arcseconds per hour.
                - DEC_rate_arcsec_per_h: Declination rate in arcseconds per hour.
                - AZ_deg: Azimuth in degrees.
                - EL_deg: Elevation in degrees.
                - r_au: Heliocentric distance in astronomical units.
                - delta_au: Observer-centric distance in astronomical units.
                - V_mag: Visual magnitude.
                - alpha_deg: Phase angle in degrees.
                - RSS_3sigma_arcsec: 3-sigma RSS uncertainty in arcseconds.

        Raises:
        -------
        Exception
            If an error occurs during the query, the error is logged and None is returned.
        """
        try:
            start_time = time.time()
            obj = Horizons(
                id_type="smallbody",
                id=query.target,
                location=HorizonsInterface.OBSERVER_LOCATION,
                epochs={"start": query.start.iso, "stop": query.end.iso, "step": query.step},
            )
            ephemeris = obj.ephemerides()
            end_time = time.time()
            HorizonsInterface.logger.info(
                f"Query for range {query.start} to {query.end} successful for target {query.target}. Time taken: {end_time - start_time:.2f} seconds."
            )

            ephemeris_data = EphemerisData(
                datetime_jd=Time(ephemeris["datetime_jd"], format="jd"),
                datetime_iso=Time(Time(ephemeris["datetime_jd"], format="jd").iso, format="iso"),
                RA_deg=np.array(ephemeris["RA"]),
                DEC_deg=np.array(ephemeris["DEC"]),
                RA_rate_arcsec_per_h=np.array(ephemeris["RA_rate"]),
                DEC_rate_arcsec_per_h=np.array(ephemeris["DEC_rate"]),
                AZ_deg=np.array(ephemeris["AZ"]),
                EL_deg=np.array(ephemeris["EL"]),
                r_au=np.array(ephemeris["r"]),
                delta_au=np.array(ephemeris["delta"]),
                V_mag=np.array(ephemeris["V"]),
                alpha_deg=np.array(ephemeris["alpha"]),
                RSS_3sigma_arcsec=np.array(ephemeris["RSS_3sigma"]),
            )

            return QueryResult(query.target, query.start, query.end, ephemeris_data)
        except Exception as e:
            HorizonsInterface.logger.error(
                f"An error occurred during query for range {query.start} to {query.end} for target {query.target}: {e}"
            )
            return None

    @staticmethod
    def query_ephemeris_from_csv(csv_filename: str):
        """
        Processes a CSV file to query ephemeris data for multiple astronomical targets using the JPL Horizons system.
        The ephemeris data is then saved to individual CSV files for each target and date range specified in the input CSV.

        Parameters:
        -----------
        csv_filename : str
            The path to the input CSV file. The CSV should have the following columns:
            - target: The identifier of the astronomical object.
            - start: The start date of the query range (ISO format).
            - end: The end date of the query range (ISO format).
            - step: The step size for the query epochs (e.g., '1h' for 1 hour).

        Returns:
        --------
        None
            The function saves the queried ephemeris data to CSV files in the current directory.
            The output filenames are generated based on the target name and the date range.

        Raises:
        -------
        Exception
            If an error occurs during the processing of the CSV file, the error is logged.
        """

        try:
            total_start_time = time.time()
            # Read the CSV file
            df = pd.read_csv(csv_filename)

            # Process each row in the CSV file
            for index, row in df.iterrows():
                query = QueryInput(
                    target=row.iloc[0],
                    start=Time(row.iloc[1], scale="utc"),
                    end=Time(row.iloc[2], scale="utc"),
                    step=row.iloc[3],
                )

                # Calculate the total number of instances
                total_days = (query.end - query.start).jd
                step_hours = float(query.step[:-1])
                step_days = step_hours / 24.0
                max_instances = 10000
                step_instances = total_days / step_days

                # Check if multiple queries are needed
                if step_instances > max_instances:
                    HorizonsInterface.logger.info(
                        f"Total instances exceed 10,000 for target {query.target}. Splitting the queries."
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

                all_ephemeris = EphemerisData()

                # Run queries sequentially
                for start, end in time_ranges:
                    result = HorizonsInterface.query_single_range(
                        QueryInput(query.target, start, end, query.step)
                    )
                    if result is not None:
                        all_ephemeris.datetime_jd = Time(
                            np.concatenate((all_ephemeris.datetime_jd.jd, result.ephemeris.datetime_jd.jd)),
                            format="jd",
                        )
                        all_ephemeris.datetime_iso = Time(
                            np.concatenate(
                                (all_ephemeris.datetime_iso.iso, result.ephemeris.datetime_iso.iso)
                            ),
                            format="iso",
                        )
                        all_ephemeris.RA_deg = np.concatenate((all_ephemeris.RA_deg, result.ephemeris.RA_deg))
                        all_ephemeris.DEC_deg = np.concatenate(
                            (all_ephemeris.DEC_deg, result.ephemeris.DEC_deg)
                        )
                        all_ephemeris.RA_rate_arcsec_per_h = np.concatenate(
                            (all_ephemeris.RA_rate_arcsec_per_h, result.ephemeris.RA_rate_arcsec_per_h)
                        )
                        all_ephemeris.DEC_rate_arcsec_per_h = np.concatenate(
                            (all_ephemeris.DEC_rate_arcsec_per_h, result.ephemeris.DEC_rate_arcsec_per_h)
                        )
                        all_ephemeris.AZ_deg = np.concatenate((all_ephemeris.AZ_deg, result.ephemeris.AZ_deg))
                        all_ephemeris.EL_deg = np.concatenate((all_ephemeris.EL_deg, result.ephemeris.EL_deg))
                        all_ephemeris.r_au = np.concatenate((all_ephemeris.r_au, result.ephemeris.r_au))
                        all_ephemeris.delta_au = np.concatenate(
                            (all_ephemeris.delta_au, result.ephemeris.delta_au)
                        )
                        all_ephemeris.V_mag = np.concatenate((all_ephemeris.V_mag, result.ephemeris.V_mag))
                        all_ephemeris.alpha_deg = np.concatenate(
                            (all_ephemeris.alpha_deg, result.ephemeris.alpha_deg)
                        )
                        all_ephemeris.RSS_3sigma_arcsec = np.concatenate(
                            (all_ephemeris.RSS_3sigma_arcsec, result.ephemeris.RSS_3sigma_arcsec)
                        )

                # Convert to pandas DataFrame
                relevant_data = pd.DataFrame(
                    {
                        "datetime_jd": all_ephemeris.datetime_jd.jd,
                        "datetime_iso": all_ephemeris.datetime_iso.iso,
                        "RA": all_ephemeris.RA_deg,
                        "DEC": all_ephemeris.DEC_deg,
                        "RA_rate": all_ephemeris.RA_rate_arcsec_per_h,
                        "DEC_rate": all_ephemeris.DEC_rate_arcsec_per_h,
                        "AZ": all_ephemeris.AZ_deg,
                        "EL": all_ephemeris.EL_deg,
                        "r": all_ephemeris.r_au,
                        "delta": all_ephemeris.delta_au,
                        "V": all_ephemeris.V_mag,
                        "alpha": all_ephemeris.alpha_deg,
                        "RSS_3sigma": all_ephemeris.RSS_3sigma_arcsec,
                    }
                )

                # Generate output filename
                output_filename = f"{query.target}_{query.start.iso}_{query.end.iso}.csv".replace(
                    ":", "-"
                ).replace(" ", "_")

                # Save the data to a CSV file
                relevant_data.to_csv(output_filename, index=False)
                HorizonsInterface.logger.info(f"Ephemeris data successfully saved to {output_filename}")

            total_end_time = time.time()
            HorizonsInterface.logger.info(
                f"Total time taken for processing the CSV file: {total_end_time - total_start_time:.2f} seconds."
            )

        except Exception as e:
            HorizonsInterface.logger.error(f"An error occurred while processing the CSV file: {e}")


# Example usage
# if __name__ == "__main__":
#     HorizonsInterface.query_ephemeris_from_csv("targets.csv")
