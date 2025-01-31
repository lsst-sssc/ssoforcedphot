import logging
import time

import astropy.units as u
import numpy as np
from astropy.table import Table, vstack
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from forcedphot.ephemeris.data_model import EphemerisData, QueryInput, QueryResult


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
        self.logger = logging.getLogger("horizons_interface")

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
        - Time ranges are calculated using astropy Time objects and are returned as such.
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
                f" of instances: {step_instances}. Splitting the queries."
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

    def save_horizons_data_to_ecsv(self, query_input, ephemeris_data):
        """
        Save queried ephemeris data to an ECSV file.

        This method takes the query input and result, converts the relevant data into an astropy Table,
        rewrites the unit and the descriptions of the columns, and then saves it as an ECSV file.
        The filename is generated based on the target and time range of the query.

        Parameters
        ----------
        query_input : QueryInput
            The input query parameters containing target, start time, end time, and step.
        ephemeris_data : EphemerisData
            The result of the ephemeris query containing the ephemeris data.

        Returns
        -------
        None

        Notes
        -----
        - The Table is created with columns for datetime in Julian Date, RA, DEC, RA rate, DEC rate,
          elevation, heliocentric distance, geocentric distance, V magnitude, phase angle, and RSS 3-sigma.
        - The output filename is generated by combining the target name and the start and end times
         of the query, with colons and spaces replaced to create a valid filename.
        - The data is saved in ECSV format.
        """

        output_filename = f"{query_input.target}_{query_input.start.iso}_{query_input.end.iso}.ecsv".replace(
            ":", "-"
        ).replace(" ", "_").replace("/", "_")

        # Save the data to an ECSV file

        result_table = Table(
            {
                "datetime": ephemeris_data.datetime.jd,
                "RA_deg": ephemeris_data.RA_deg,
                "DEC_deg": ephemeris_data.DEC_deg,
                "RA_rate_arcsec_per_h": ephemeris_data.RA_rate_arcsec_per_h,
                "DEC_rate_arcsec_per_h": ephemeris_data.DEC_rate_arcsec_per_h,
                "AZ_deg": ephemeris_data.AZ_deg,
                "EL_deg": ephemeris_data.EL_deg,
                "r_au": ephemeris_data.r_au,
                "delta_au": ephemeris_data.delta_au,
                "V_mag": ephemeris_data.V_mag,
                "alpha_deg": ephemeris_data.alpha_deg,
                "RSS_3sigma_arcsec": ephemeris_data.RSS_3sigma_arcsec,
                "SMAA_3sigma_arcsec": ephemeris_data.SMAA_3sigma_arcsec,
                "SMIA_3sigma_arcsec": ephemeris_data.SMIA_3sigma_arcsec,
                "Theta_3sigma_deg": ephemeris_data.Theta_3sigma_deg,
            }
        )

        result_table["datetime"].unit = u.day
        result_table["datetime"].description = "Time for the ephemeris data points."
        result_table["RA_deg"].unit = u.deg
        result_table["RA_deg"].description = "Right Ascension in degrees"
        result_table["DEC_deg"].unit = u.deg
        result_table["DEC_deg"].description = "Declination in degrees"
        result_table["RA_rate_arcsec_per_h"].unit = u.arcsec / u.hour
        result_table["RA_rate_arcsec_per_h"].description = "Rate of change in Right Ascension"
        "in arcseconds per hour"
        result_table["DEC_rate_arcsec_per_h"].unit = u.arcsec / u.hour
        result_table["DEC_rate_arcsec_per_h"].description = "Rate of change in Declination"
        "in arcseconds per hour"
        result_table["AZ_deg"].unit = u.deg
        result_table["AZ_deg"].description = "Azimuth in degrees"
        result_table["EL_deg"].unit = u.deg
        result_table["EL_deg"].description = "Elevation in degrees"
        result_table["r_au"].unit = u.au
        result_table["r_au"].description = "Heliocentric distance in astronomical units"
        result_table["delta_au"].unit = u.au
        result_table["delta_au"].description = "Geocentric distance in astronomical units"
        result_table["V_mag"].description = "Visual magnitude"
        result_table["alpha_deg"].unit = u.deg
        result_table["alpha_deg"].description = "Phase angle in degrees"
        result_table["RSS_3sigma_arcsec"].unit = u.arcsec
        result_table["RSS_3sigma_arcsec"].description = "3-sigma uncertainty in arcseconds"
        result_table["SMAA_3sigma_arcsec"].unit = u.arcsec
        result_table["SMAA_3sigma_arcsec"].description = "Semi-major axis of error ellipse in arcseconds"
        result_table["SMIA_3sigma_arcsec"].unit = u.arcsec
        result_table["SMIA_3sigma_arcsec"].description = "Semi-minor axis of error ellipse in arcseconds"
        result_table["Theta_3sigma_deg"].unit = u.deg
        result_table["Theta_3sigma_deg"].description = "Position angle of error ellipse in degrees"

        result_table.write("./" + output_filename, format="ascii.ecsv", overwrite=True)
        self.logger.info(f"Ephemeris data successfully saved to {output_filename}")

    def query_single_range(self, query: QueryInput, save_data: bool = False) -> QueryResult:
        """
        Query ephemeris for a single time range.

        Parameters
        ----------
        query : QueryInput
            The query parameters containing target, start time, end time, and step.
        save_data : bool, optional
            Whether to save the queried ephemeris data to an ECSV file. Default is False.

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
                    id=query.target,
                    id_type=query.target_type,
                    location=self.observer_location,
                    epochs={"start": start.iso, "stop": end.iso, "step": query.step},
                )

                if query.target_type == "comet_name":
                    jd_mid = (start + (end - start) / 2).jd
                    mag_type = "Tmag"
                    ephemeris = obj.ephemerides(
                        closest_apparition=f"<{jd_mid:.0f}",
                        no_fragments=True,
                        skip_daylight=False,
                    )

                else:
                    # TODO Some object only run with smallbody type, but the mag_type remains Tmag
                    mag_type = "V"
                    ephemeris = obj.ephemerides(skip_daylight=False)

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
                datetime=Time(combined_ephemeris["datetime_jd"], scale="utc", format="jd"),
                RA_deg=np.array(combined_ephemeris["RA"]),
                DEC_deg=np.array(combined_ephemeris["DEC"]),
                RA_rate_arcsec_per_h=np.array(combined_ephemeris["RA_rate"]),
                DEC_rate_arcsec_per_h=np.array(combined_ephemeris["DEC_rate"]),
                AZ_deg=np.array(combined_ephemeris["AZ"]),
                EL_deg=np.array(combined_ephemeris["EL"]),
                r_au=np.array(combined_ephemeris["r"]),
                delta_au=np.array(combined_ephemeris["delta"]),
                V_mag=np.array(combined_ephemeris[mag_type]),
                alpha_deg=np.array(combined_ephemeris["alpha"]),
                RSS_3sigma_arcsec=np.array(combined_ephemeris["RSS_3sigma"]),
                SMAA_3sigma_arcsec=np.array(combined_ephemeris["SMAA_3sigma"]),
                SMIA_3sigma_arcsec=np.array(combined_ephemeris["SMIA_3sigma"]),
                Theta_3sigma_deg=np.array(combined_ephemeris["Theta_3sigma"]),
            )

            # Save the data to an ECSV file
            if save_data:
                self.save_horizons_data_to_ecsv(query, ephemeris_data)

            return QueryResult(query.target, query.start, query.end, ephemeris_data)
        except Exception as e:
            self.logger.error(
                f"An error occurred during query for range {query.start} to {query.end}"
                f" for target {query.target}"
            )
            self.logger.error(f"Error details: {str(e)}")

            return None


# Example usage
if __name__ == "__main__":

    target_query = QueryInput(
        target="65803",
        target_type="smallbody",
        start=Time("2022-09-26 00:00"),
        end=Time("2022-09-29 23:59"),
        step="1h",
    )
    horizons = HorizonsInterface()
    result = horizons.query_single_range(query=target_query, save_data=False)
