import logging
import os
from dataclasses import dataclass
from urllib.request import urlretrieve

from astropy.time import Time
from lsst.rsp import get_tap_service
from pyvo.dal.adhoc import DatalinkResults


@dataclass
class SearchInput:
    """
    Dataclass to hold search input parameters.

    Attributes:
        ra (str): Right Ascension in degrees.
        dec (str): Declination in degrees.
        obs_time (Time): Observation time.
        band (str): Filter band.
    """
    ra: str
    dec: str
    obs_time: Time
    band: str

class ImageServiceRsp:
    """
    A class to search for and retrieve astronomical images using the LSST RSP service.

    Attributes:
        DEFAULT_SAVE_IMAGES (bool): Default setting for saving images.
        service: TAP service object for querying.
        auth_session: Authenticated session for the TAP service.
        logger (logging.Logger): Logger for the class.
    """

    DEFAULT_SAVE_IMAGES = False

    def __init__(self):
        """Initialize the ImageServiceRsp class."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.service = get_tap_service("tap")
        self.auth_session = self.service._session

    def search_for_images(self, query_input: SearchInput, save_data: bool = False):
        """
        Search for images based on the given input parameters.

        Args:
            query_input (SearchInput): Input parameters for the search.
            save_data (bool, optional): Whether to save the retrieved images. Defaults to False.

        Returns:
            DataFrame containing the search results, or None if no results found.
        """
        query = f"""
        SELECT * FROM ivoa.ObsCore
        WHERE CONTAINS(POINT('ICRS', {query_input.ra}, {query_input.dec}), s_region) = 1
        AND calib_level = 2
        AND t_max > {query_input.obs_time.mjd} AND t_min < {query_input.obs_time.mjd}
        AND lsst_band = '{query_input.band}'
        """

        job = self.service.submit_job(query)
        job.run()
        job.wait(phases=['COMPLETED', 'ERROR'])
        self.logger.info(f'Job phase is {job.phase}')
        job.raise_if_error()

        results = job.fetch_result().to_table().to_pandas()
        self.logger.info(f"Number of hits: {len(results)}")

        if results.empty:
            self.logger.warning("No results found.")
            return None

        if save_data:
            results_raw = job.fetch_result()
            for result in results_raw:
                self.save_image(result)

        return results

    def save_image(self, result):
        """
        Save an image from the search results.

        Args:
            result: A single result row from the TAP query.
        """
        obs_id = result["obs_id"]
        obs_date = Time(result["t_min"], format="mjd")
        dl_results = DatalinkResults.from_result_url(result['access_url'], session=self.auth_session)
        image_url = dl_results['access_url'][0]

        output_filename = f"{obs_id}_{obs_date.iso}.fits".replace(":", "-").replace(" ", "_")
        filename = os.path.join(os.getenv("HOME"),
                                "WORK/ssoforcedphot/src/forcedphot/image_service", output_filename)
        urlretrieve(image_url, filename)
        self.logger.info(f"Saved image: {filename}")


if __name__ == "__main__":
    test_search = SearchInput(
        ra="62.0",
        dec="-37.0",
        obs_time=Time(60611.1762, format='mjd'),
        band="r",
    )
    image_search = ImageServiceRsp()
    results = image_search.search_for_images(query_input=test_search, save_data=True)
    if results is not None:
        print(results)
