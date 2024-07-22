import numpy as np
from astropy.io import ascii
from astropy.time import Time
from astropy.table import Table
from typing import List
from forcedphot.ephemeris.local_dataclasses import EphemerisData

class DataLoader:
    @staticmethod
    def load_ephemeris_from_ecsv(file_path: str) -> EphemerisData:
        """
        Load ephemeris data from an ECSV file and return it as an EphemerisData object.

        Parameters:
        -----------
        file_path : str
            Path to the ECSV file containing ephemeris data.

        Returns:
        --------
        EphemerisData
            An EphemerisData object populated with the data from the ECSV file.

        Raises:
        -------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the ECSV file is missing required columns.
        """
        try:
            # Read the ECSV file
            table = Table.read(file_path, format='ascii.ecsv')

            # Check if all required columns are present
            required_columns = [
                'datetime_jd', 'RA_deg', 'DEC_deg', 'RA_rate_arcsec_per_h',
                'DEC_rate_arcsec_per_h', 'AZ_deg', 'EL_deg', 'r_au', 'delta_au',
                'V_mag', 'alpha_deg', 'RSS_3sigma_arcsec'
            ]
            missing_columns = [col for col in required_columns if col not in table.colnames]
            if missing_columns:
                raise ValueError(f"Missing columns in ECSV file: {', '.join(missing_columns)}")

            # Create and populate the EphemerisData object
            ephemeris_data = EphemerisData(
                datetime_jd=Time(table['datetime_jd'], format='jd'),
                RA_deg=np.array(table['RA_deg']),
                DEC_deg=np.array(table['DEC_deg']),
                RA_rate_arcsec_per_h=np.array(table['RA_rate_arcsec_per_h']),
                DEC_rate_arcsec_per_h=np.array(table['DEC_rate_arcsec_per_h']),
                AZ_deg=np.array(table['AZ_deg']),
                EL_deg=np.array(table['EL_deg']),
                r_au=np.array(table['r_au']),
                delta_au=np.array(table['delta_au']),
                V_mag=np.array(table['V_mag']),
                alpha_deg=np.array(table['alpha_deg']),
                RSS_3sigma_arcsec=np.array(table['RSS_3sigma_arcsec'])
            )

            print(f"Loaded ephemeris data with {len(ephemeris_data.datetime_jd)} points from {file_path}.")

            return ephemeris_data

        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} was not found.")
        except Exception as e:
            raise ValueError(f"Error loading ECSV file: {str(e)}")

    @staticmethod
    def load_multiple_ephemeris_files(file_paths: List[str]) -> List[EphemerisData]:
        """
        Load multiple ephemeris files and return a list of EphemerisData objects.

        Parameters:
        -----------
        file_paths : List[str]
            A list of paths to ECSV files containing ephemeris data.

        Returns:
        --------
        List[EphemerisData]
            A list of EphemerisData objects, each populated with data from one ECSV file.
        """
        return [DataLoader.load_ephemeris_from_ecsv(file_path) for file_path in file_paths]


if __name__ == "__main__":
    # Example usage
    # file_path = "./Ceres_2024-01-01_00-00-00.000_2025-12-31_23-59-00.000.ecsv"
    # try:
    #     ephemeris_data = DataLoader.load_ephemeris_from_ecsv(file_path)
    # except Exception as e:
    #     print(f"Error: {str(e)}")

    # Example of loading multiple files
    file_paths = ["./Ceres_2024-01-01_00-00-00.000_2025-12-31_23-59-00.000.ecsv", "./Encke_2024-01-01_00-00-00.000_2024-06-30_23-59-00.000.ecsv"]
    try:
        ephemeris_list = DataLoader.load_multiple_ephemeris_files(file_paths)
        print(f"Loaded {len(ephemeris_list)} ephemeris files.")
    except Exception as e:
        print(f"Error: {str(e)}")
