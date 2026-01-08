"""
Ephemeris Refinement Service.

This module provides precise ephemeris refinement by querying JPL Horizons or Miriade
at exact observation times extracted from discovered images, eliminating interpolation errors.
"""

import logging
from typing import Optional

from astropy.time import Time
from ephemeris.miriade_interface import MiriadeInterface
from image_photometry.utils import EphemerisDataCompressed, ImageMetadata


class EphemerisRefinementService:
    """
    Refines ephemeris data by querying JPL Horizons or Miriade at exact
    observation times extracted from discovered images.

    This service eliminates interpolation errors by obtaining precise ephemeris
    data at the exact midpoint times of image exposures, rather than relying
    on linear interpolation between coarse ephemeris points.
    """

    def __init__(self, observer_location: str = "X05"):
        """
        Initialize the refinement service.

        Parameters
        ----------
        observer_location : str
            Observatory code (default: X05 for Rubin Observatory)
        """
        self.observer_location = observer_location
        self.logger = logging.getLogger("ephemeris_refinement")

    def refine_ephemeris(
        self,
        image_metadata_list: list[ImageMetadata],
        target: str,
        target_type: str,
        ephemeris_service: str,
        cache_folder: Optional[str] = "./output/refined_ephemeris",
        use_cache: bool = True,
    ) -> dict[str, EphemerisDataCompressed]:
        """
        Extract exact observation times from images and query precise ephemeris.

        Parameters
        ----------
        image_metadata_list : list[ImageMetadata]
            Images found in Stage 1 image search
        target : str
            Target designation (e.g., "2024 TN57")
        target_type : str
            Object type (smallbody, majorbody, etc.)
        ephemeris_service : str
            "Horizons" or "Miriade"
        cache_folder : str, optional
            Directory to store cached refined ephemeris
        use_cache : bool
            Whether to use cached data if available

        Returns
        -------
        dict[str, EphemerisDataCompressed]
            Mapping of image_id → precise ephemeris at observation time
            where image_id = f"{visit_id}_{detector_id}"
        """
        self.logger.info(
            f"Refining ephemeris for {len(image_metadata_list)} images "
            f"using {ephemeris_service}"
        )

        # Extract observation times
        obs_times = self._extract_observation_times(image_metadata_list)

        # Check cache
        if use_cache:
            cached_data = self._load_from_cache(target, cache_folder)
            if cached_data:
                self.logger.info("Using cached refined ephemeris data")
                return self._match_cached_data(image_metadata_list, cached_data)

        # Query refined ephemeris
        if ephemeris_service.lower() == "horizons":
            refined_ephemeris = self._query_horizons_batch(
                target, target_type, obs_times
            )
        elif ephemeris_service.lower() == "miriade":
            refined_ephemeris = self._query_miriade_batch(
                target, target_type, obs_times
            )
        else:
            raise ValueError(
                f"Invalid ephemeris service: {ephemeris_service}. "
                "Use 'Horizons' or 'Miriade'."
            )

        # Map to image IDs
        result = self._map_to_image_ids(
            image_metadata_list, refined_ephemeris
        )

        # Cache results
        if use_cache and cache_folder:
            self._save_to_cache(target, result, cache_folder)

        return result

    def _extract_observation_times(
        self, image_metadata_list: list[ImageMetadata]
    ) -> list[Time]:
        """
        Extract midpoint observation times from image metadata.

        Parameters
        ----------
        image_metadata_list : list[ImageMetadata]
            List of image metadata objects

        Returns
        -------
        list[Time]
            List of unique observation midpoint times
        """
        obs_times = []
        for metadata in image_metadata_list:
            # Calculate midpoint time
            t_begin = metadata.datetime_begin
            t_end = metadata.datetime_end
            t_mid = Time((t_begin.mjd + t_end.mjd) / 2.0, format="mjd", scale="utc")
            obs_times.append(t_mid)

        # Remove duplicates (same visit/detector might appear multiple times)
        unique_times = sorted(set(t.iso for t in obs_times))
        unique_times = [Time(t, format="iso", scale="utc") for t in unique_times]

        self.logger.info(f"Extracted {len(unique_times)} unique observation times")
        return unique_times

    def _query_horizons_batch(
        self,
        target: str,
        target_type: str,
        obs_times: list[Time],
    ) -> list[EphemerisDataCompressed]:
        """
        Query JPL Horizons for multiple observation times using batch approach.

        Strategy:
        - Horizons accepts list of Julian Dates via epochs parameter
        - However, URI limit is ~2000 characters
        - Batch into groups of ~50 times to stay under limit

        Parameters
        ----------
        target : str
            Target designation
        target_type : str
            Object type
        obs_times : list[Time]
            List of observation times

        Returns
        -------
        list[EphemerisDataCompressed]
            Refined ephemeris data for each observation time
        """
        from astroquery.jplhorizons import Horizons

        # Convert to Julian Dates
        jd_times = [t.jd for t in obs_times]

        # Batch into groups (to avoid URI length limit)
        batch_size = 50  # Conservative estimate
        batches = [
            jd_times[i:i + batch_size]
            for i in range(0, len(jd_times), batch_size)
        ]

        self.logger.info(
            f"Querying Horizons in {len(batches)} batches "
            f"({len(jd_times)} total epochs)"
        )

        all_ephemeris = []

        for i, batch in enumerate(batches):
            self.logger.info(
                f"Querying batch {i+1}/{len(batches)} "
                f"({len(batch)} epochs)"
            )

            # Query Horizons with list of epochs
            obj = Horizons(
                id=target,
                id_type=target_type,
                location=self.observer_location,
                epochs=batch,  # List of JD times
            )

            try:
                ephemeris = obj.ephemerides(skip_daylight=False)

                # Convert to EphemerisDataCompressed format
                for row in ephemeris:
                    ephem_data = self._convert_horizons_row(row)
                    all_ephemeris.append(ephem_data)

                self.logger.info(
                    f"Successfully retrieved {len(ephemeris)} ephemeris points"
                )

            except Exception as e:
                self.logger.error(
                    f"Error querying Horizons batch {i+1}: {str(e)}"
                )
                # Continue with other batches
                continue

        return all_ephemeris

    def _query_miriade_batch(
        self,
        target: str,
        target_type: str,
        obs_times: list[Time],
    ) -> list[EphemerisDataCompressed]:
        """
        Query Miriade for multiple observation times.

        Note: Miriade doesn't support arbitrary epoch lists like Horizons.
        Strategy: Query individually for each observation time.

        Parameters
        ----------
        target : str
            Target designation
        target_type : str
            Object type
        obs_times : list[Time]
            List of observation times

        Returns
        -------
        list[EphemerisDataCompressed]
            Refined ephemeris data for each observation time
        """
        from astroquery.imcce import Miriade

        self.logger.info(
            f"Querying Miriade for {len(obs_times)} individual epochs "
            "(sequential queries)"
        )

        # Map target type
        miriade_interface = MiriadeInterface(self.observer_location)
        objtype = miriade_interface.set_target_type(target_type)

        all_ephemeris = []

        for i, obs_time in enumerate(obs_times):
            try:
                # Query single epoch
                ephemeris = Miriade.get_ephemerides(
                    targetname=target,
                    objtype=objtype,
                    location=self.observer_location,
                    epoch=obs_time.jd,
                    epoch_step='1d',  # Only need 1 point
                    epoch_nsteps=1,
                    coordtype=5,
                )

                # Convert to EphemerisDataCompressed format
                if len(ephemeris) > 0:
                    ephem_data = self._convert_miriade_row(ephemeris[0])
                    all_ephemeris.append(ephem_data)

                if (i + 1) % 10 == 0:
                    self.logger.info(
                        f"Progress: {i+1}/{len(obs_times)} queries completed"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error querying Miriade for time {obs_time.iso}: {str(e)}"
                )
                # Continue with other times
                continue

        self.logger.info(
            f"Successfully retrieved {len(all_ephemeris)}/{len(obs_times)} "
            "ephemeris points"
        )

        return all_ephemeris

    def _convert_horizons_row(self, row) -> EphemerisDataCompressed:
        """Convert Horizons table row to EphemerisDataCompressed."""
        return EphemerisDataCompressed(
            datetime=Time(row["datetime_jd"], scale="utc", format="jd"),
            ra_deg=float(row["RA"]),
            dec_deg=float(row["DEC"]),
            ra_rate=float(row["RA_rate"]),
            dec_rate=float(row["DEC_rate"]),
            uncertainty={
                "rss": float(row["RSS_3sigma"]),
                "smaa": float(row["SMAA_3sigma"]),
                "smia": float(row["SMIA_3sigma"]),
                "theta": float(row["Theta_3sigma"]),
            },
        )

    def _convert_miriade_row(self, row) -> EphemerisDataCompressed:
        """Convert Miriade table row to EphemerisDataCompressed."""
        return EphemerisDataCompressed(
            datetime=Time(row["epoch"], scale="utc", format="jd"),
            ra_deg=float(row["RAJ2000"]),
            dec_deg=float(row["DECJ2000"]),
            ra_rate=float(row["RAcosD_rate"]),
            dec_rate=float(row["DEC_rate"]),
            uncertainty={
                "rss": float(row.get("posunc", 0.001)),
                "smaa": 0.0,  # Not available in Miriade
                "smia": 0.0,
                "theta": 0.0,
            },
        )

    def _map_to_image_ids(
        self,
        image_metadata_list: list[ImageMetadata],
        refined_ephemeris: list[EphemerisDataCompressed],
    ) -> dict[str, EphemerisDataCompressed]:
        """
        Map refined ephemeris to image IDs by matching observation times.

        Parameters
        ----------
        image_metadata_list : list[ImageMetadata]
            Original image metadata
        refined_ephemeris : list[EphemerisDataCompressed]
            Refined ephemeris data

        Returns
        -------
        dict[str, EphemerisDataCompressed]
            Mapping of image_id → refined ephemeris
        """
        result = {}

        for metadata in image_metadata_list:
            # Calculate midpoint time
            t_mid = Time(
                (metadata.datetime_begin.mjd + metadata.datetime_end.mjd) / 2.0,
                format="mjd",
                scale="utc"
            )

            # Find closest ephemeris point
            closest_ephem = min(
                refined_ephemeris,
                key=lambda e: abs((e.datetime - t_mid).sec)
            )

            # Create image ID
            image_id = f"{metadata.visit_id}_{metadata.detector_id}"
            result[image_id] = closest_ephem

        return result

    def _load_from_cache(
        self, target: str, cache_folder: str
    ) -> Optional[dict]:
        """Load cached refined ephemeris if available."""
        # TODO: Implement caching logic
        # Format: refined_ephemeris_{target}_{timestamp}.pkl
        return None

    def _save_to_cache(
        self, target: str, data: dict, cache_folder: str
    ) -> None:
        """Save refined ephemeris to cache."""
        # TODO: Implement caching logic
        pass

    def _match_cached_data(
        self, image_metadata_list: list[ImageMetadata], cached_data: dict
    ) -> dict[str, EphemerisDataCompressed]:
        """Match cached data to current image list."""
        # TODO: Implement cache matching
        pass
