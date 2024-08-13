import logging
import time

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time
from astroquery.esasky import ESASky
from astroquery.imcce import Miriade

from forcedphot.ephemeris.data_model import EphemerisData, QueryInput, QueryInputMiriade, QueryResult


class MiriadeInterface:
    """
    A class for querying ephemeris data for celestial objects using the Miriade service.

    This class provides methods to query ephemeris data for celestial objects using the Miriade service.
    It supports single range queries and batch processing from CSV files.

    Attributes:
        DEFAULT_OBSERVER_LOCATION (str): Default location code for the observer (set to "X05" for
        Rubin Observatory).
        logger (logging.Logger): Logger for the class.
        observer_location (str): The observer location code used for queries.

    Methods:
        convert_to_miriade_query(query: QueryInput) -> QueryInputMiriade:
            Calculate the number of steps for a Miriade query based on the input parameters.

        query_single_range(query: QueryInput):
            Query Miriade for ephemeris data within a single time range.

        query_ephemeris_from_csv(csv_file: str, observer_location=DEFAULT_OBSERVER_LOCATION):
            Process multiple queries from a CSV file and save results as ECSV files.

    The class uses Astroquery library to interact with the Miriade service. It supports querying
    ephemeris data for celestial objects within a specified time range. The class also provides a
    method to process multiple queries from a CSV file and save the results as ECSV files.
    TODO To check if there is a row limit at the Miriade service, because so far it seems that it can
    handle more than 5000 rows
    """

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Rubin locaion
    DEFAULT_OBSERVER_LOCATION = "X05"

    def __init__(self, observer_location=DEFAULT_OBSERVER_LOCATION):
        """
        Initialize the MiriadeInterface with an optional observer location.

        Parameters:
        ----------
        observer_location : str, optional
            The observer location code. Default is 'X05' (Rubin location).
        """
        self.observer_location = observer_location

    def set_target_type(self, target_type):
        """
        Set the target type for the miriade query based on the input target type.

        This method maps the input target type to the corresponding target type
        used by the Horizons system. The supported target types are "smallbody",
        "comet_name", and "asteroid_name".

        Parameters
        ----------
        target_type : str
            The input target type, which can be one of the following:
            - "smallbody": Represents small bodies (asteroids, comets).
            - "comet_name": Represents comets.
            - "asteroid_name": Represents asteroids by name.
            - "designation": Represents comets by designation.

        Returns
        -------
        str
            The corresponding target type used by the Horizons system:
            - "asteroid" for "smallbody" and "asteroid_name".
            - "comet" for "comet_name".
        """
        if target_type == "smallbody":
            return "asteroid"
        elif target_type == "comet_name":
            return "comet"
        elif target_type == "asteroid_name":
            return "asteroid"
        elif target_type == "designation":
            return "comet"
        else:
            raise ValueError(
                f"Unsupported target type: {target_type}. Please chose"
                f"from 'smallbody', 'comet_name', 'asteroid_name', or 'designation'."
            )

    def convert_to_miriade_query(self, query: QueryInput) -> QueryInputMiriade:
        """
        Calculate the number of steps for a Miriade query and prepare the query input.

        This method processes the input query, calculates the number of steps based on
        the given time range and step frequency, and prepares a QueryInputMiriade object
        for use with the Miriade service.

        Parameters:
        -----------
        query : QueryInput
            A dataclass containing the query parameters, including:
            - target: The celestial object name or ID
            - start: The start time of the query
            - end: The end time of the query
            - step: The time step for the query (e.g., '1h' for 1 hour)

        Returns:
        --------
        QueryInputMiriade
            A dataclass containing the parameters for the Miriade query, including:
            - target: The celestial object name or ID
            - start: The start time of the query
            - step: The time step for the query
            - nsteps: The calculated number of steps for the query

        Raises:
        -------
        ValueError
        If the step unit in the input query is not recognized (valid units are "s", "m", "h", "d").

        Notes:
        ------
        - The method supports step units in seconds (s), minutes (m), hours (h), and days (d).
        - For comet targets, it resolves the comet name using the ESASky.find_sso method.
        - The number of steps is calculated based on the total duration and step frequency.
        - The target type is set using the self.set_target_type method.
        """

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

        total_days = (query.end - query.start).jd
        nsteps = int(total_days * 86400 / step_freqency.to(u.s).value)
        print(f"Total days: {total_days}, nsteps: {nsteps}")

        if query.target_type == "comet_name":
            resolved_target = ESASky.find_sso(sso_name=query.target, sso_type="COMET")
            sso_name = resolved_target[0].get("sso_name")
        else:
            sso_name = query.target

        query_miriade = QueryInputMiriade(
            target=sso_name,
            objtype=self.set_target_type(query.target_type),
            start=query.start,
            step=query.step,
            nsteps=nsteps,
        )

        return query_miriade

    def save_miriade_data_to_ecsv(self, query_input, ephemeris_data):
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
        ).replace(" ", "_")

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

        result_table.write("./" + output_filename, format="ascii.ecsv", overwrite=True)
        self.logger.info(f"Ephemeris data successfully saved to {output_filename}")

    def query_single_range(self, query: QueryInput, save_data: bool = False):
        """
        Query Miriade for ephemeris data within a single time range.

        This method calculates the necessary parameters for a Miriade query,
        fetches the ephemeris data, and returns relevant astronomical information.

        Parameters:
        -----------
        query : QueryInput
            An object containing the query parameters, including:
            - target: The celestial object name or ID
            - start: The start time of the query
            - end: The end time of the query
            - step: The time step for the query (e.g., '1h' for 1 hour)
        save_data : bool, optional
            - A flag indicating whether to save the queried data to a file.

        Returns:
        --------
        QueryResult or None
            The queried ephemeris data wrapped in a QueryResult object if successful,
            or None if an error occurs.

        Raises:
        -------
        Exception
            Any exception that occurs during the query process is caught,
            logged, and results in a None return value.
        """

        try:
            start_time = time.time()
            # Calculate nsteps for query
            query_miriade = self.convert_to_miriade_query(query)

            # Query Miriade
            ephemeris = Miriade.get_ephemerides(
                targetname=query_miriade.target,
                objtype=query_miriade.objtype,
                location=self.observer_location,
                epoch=query_miriade.start,
                epoch_step=query_miriade.step,
                epoch_nsteps=query_miriade.nsteps,
                coordtype=5,
            )

            end_time = time.time()
            self.logger.info(
                f"Query for range {query_miriade.start} with {query_miriade.nsteps}"
                f" completed in {end_time - start_time} seconds."
            )

            # Selecting relevant columns
            relevant_columns = [
                "epoch",
                "RAJ2000",
                "DECJ2000",
                "RAcosD_rate",
                "DEC_rate",
                "AZ",
                "EL",
                "heldist",
                "delta",
                "V",
                "alpha",
                "posunc",
            ]
            relevant_data = ephemeris[relevant_columns]

            ephemeris_data = EphemerisData(
                datetime=Time(relevant_data["epoch"], scale="utc", format="jd"),
                RA_deg=np.array(relevant_data["RAJ2000"]),
                DEC_deg=np.array(relevant_data["DECJ2000"]),
                RA_rate_arcsec_per_h=np.array(relevant_data["RAcosD_rate"]),
                DEC_rate_arcsec_per_h=np.array(relevant_data["DEC_rate"]),
                AZ_deg=np.array(relevant_data["AZ"]),
                EL_deg=np.array(relevant_data["EL"]),
                r_au=np.array(relevant_data["heldist"]),
                delta_au=np.array(relevant_data["delta"]),
                V_mag=np.array(relevant_data["V"]),
                alpha_deg=np.array(relevant_data["alpha"]),
                RSS_3sigma_arcsec=np.array(relevant_data["posunc"]),
            )

            if save_data:
                # Save the queried ephemeris data to ECSV file
                self.save_miriade_data_to_ecsv(query, ephemeris_data)

            return QueryResult(query.target, query.start, query.end, ephemeris_data)

        except Exception as e:
            self.logger.error(
                f"An error occurred during query for range {query_miriade.start}"
                f"with {query_miriade.nsteps} for target {query_miriade.target}"
            )
            self.logger.error(f"Error details: {str(e)}")

            return None

    @classmethod
    def query_ephemeris_from_csv(
        cls, csv_file: str, observer_location=DEFAULT_OBSERVER_LOCATION, save_data: bool = False
    ):
        """
        Process multiple ephemeris queries from a CSV file and save results as ECSV files.

        This class method reads query parameters from a CSV file, performs ephemeris queries
        for each row, and saves the results in individual ECSV files.

        Parameters:
        -----------
        csv_file : str
            Path to the input CSV file containing query parameters.
            The CSV should have columns in the order: target, start_time, end_time, step.

        observer_location : str, optional
            The observer location code to use for all queries.
            Defaults to the class's DEFAULT_OBSERVER_LOCATION.

        save_data : bool, optional
            Whether to save the queried ephemeris data to ECSV files.

        Returns:
        --------
        QueryResult or None
            The queried ephemeris data wrapped in a QueryResult object if successful,
            or None if an error occurs. Also, the method saves the data to ECSV files.

        Raises:
        -------
        Exception
            Any exception during the process is caught and logged, but not re-raised.

        Notes:
        ------
        - The input CSV file should have no header and contain four columns:
        target, start_time, end_time, step.
        - Each query's results are saved in a separate ECSV file.
        - The method logs the total processing time for the entire CSV file.
        - If an error occurs for a single query, it's logged and the method continues
        with the next row in the CSV.
        """

        try:
            total_start_time = time.time()
            # Create an empty list to store the results
            results = []
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Create Miriade interface instance with the specified observer location
            miriade_interface = cls(observer_location)

            # Process each row in the CSV file
            for _index, row in df.iterrows():
                query = QueryInput(
                    target=row.iloc[0],
                    target_type=row.iloc[1],
                    start=Time(row.iloc[2], scale="utc"),
                    end=Time(row.iloc[3], scale="utc"),
                    step=row.iloc[4],
                )

                # Query Miriade for the current row
                query_result = miriade_interface.query_single_range(query)

                if query_result is not None:
                    # Append the result to the list
                    results.append(query_result)

                if save_data:
                    # Save the queried ephemeris data to ECSV file
                    miriade_interface.save_miriade_data_to_ecsv(query, query_result.ephemeris)

            total_end_time = time.time()
            cls.logger.info(
                f"Total time taken for processing the ECSV file:"
                f"{total_end_time - total_start_time:.2f} seconds."
            )
            return results

        except Exception as e:
            cls.logger.error(f"An error occurred during query for CSV file {csv_file}")
            cls.logger.error(f"Error details: {str(e)}")


if __name__ == "__main__":
    # MiriadeInterface.query_ephemeris_from_csv("./data/targets.csv")

    # Different observer location
    # MiriadeInterface.query_ephemeris_from_csv("./targets.csv", observer_location="500")

    # Define the target query parameters
    target_query = QueryInput(
        target="Encke",
        target_type="comet_name",
        start=Time("2024-01-01 00:00"),
        end=Time("2025-12-31 23:59"),
        step="1h",
    )
    miriade = MiriadeInterface()
    result = miriade.query_single_range(query=target_query, save_data=True)
