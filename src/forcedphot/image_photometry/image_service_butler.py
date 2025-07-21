import logging
from lsst.daf.butler import Butler, Timespan
import lsst.sphgeom as sphgeom
from astropy.time import Time, TimeDelta
import lsst.geom as geom
from typing import Optional
from ephemeris.data_model import QueryResult
from image_photometry.utils import (
    EphemerisDataCompressed,
    ImageMetadata,
    interpolate_coordinates,
)

class ImageServiceButler:
    """
    Service for searching and processing images based on ephemeris data.
    Interfaces DP1 through Butler.
    """

    def __init__(self):
        self.butler = Butler('dp1', collections='LSSTComCam/DP1')
        self.logger = logging.getLogger("ImageServiceButler")

    def search_images_polygon(self, polygons: list, 
                                 bands: list[str], 
                                 ephemeris: list[EphemerisDataCompressed]) -> list[ImageMetadata]:
        """
        Searches for images (visit_image) that intersect with given polygons and time ranges,
        then enriches them with metadata and ephemeris information.

        Args:
            polygons: List of dicts, each with 'time_start', 'time_end', 'polygon_corners'.
            bands: List of band names (strings) to search for.
            ephemeris: A list of EphemerisDataCompressed objects.

        Returns:
            A list of unique ImageMetadata objects.
        """
        all_dataset_refs = []
        self.logger.info(f"Processing {len(polygons)} polygons for bands {bands}.")

        for i, item in enumerate(polygons):
            try:
                time_start_str = item['time_start']
                time_end_str = item['time_end']

                time_start = Time(time_start_str, format="iso", scale="tai")
                time_end = Time(time_end_str, format="iso", scale="tai")
                timespan_search = Timespan(time_start, time_end) 

                corners = item['polygon_corners']
                                
                polygon_string = "POLYGON " + " ".join([f"{c[0]} {c[1]}" for c in corners])
                region_search = sphgeom.Region.from_ivoa_pos(polygon_string)

                bands_clause = "band.name IN (" + ", ".join([f"'{b}'" for b in bands]) + ")"
                where_clause = f"{bands_clause} AND visit.timespan OVERLAPS timespan AND visit_detector_region.region OVERLAPS region"

                self.logger.debug(f"Polygon {i+1}: Querying with timespan {timespan_search} and region from {len(corners)} corners.")
                self.logger.debug(f"Polygon {i+1}: Where clause: {where_clause}")

                try:
                    dataset_refs_for_item = list(self.butler.query_datasets(
                        "visit_image",
                        where=where_clause,
                        bind={"timespan": timespan_search, "region": region_search} 
                    ))
                    self.logger.info(f"Polygon {i+1}: Found {len(dataset_refs_for_item)} dataset_refs.")
                    all_dataset_refs.extend(dataset_refs_for_item)
                except Exception as butler_error:
                    # Handle EmptyQueryResultError and other butler query exceptions
                    if "EmptyQueryResultError" in str(type(butler_error)):
                        self.logger.info(f"Polygon {i+1}: No datasets found for this polygon (EmptyQueryResultError). Continuing to next polygon.")
                    else:
                        self.logger.warning(f"Polygon {i+1}: Butler query failed with error: {butler_error}. Continuing to next polygon.")
                    continue
                    
                # self.logger.info(f"Polygon {i+1}: Found {len(dataset_refs_for_item)} dataset_refs.")
                all_dataset_refs.extend(dataset_refs_for_item)
            except Exception as e:
                self.logger.error(f"Error processing polygon item {i} ({item}): {e}", exc_info=True)
                continue

        self.logger.info(f"Total {len(all_dataset_refs)} dataset_refs found before initial deduplication.")
        unique_refs = list(set(all_dataset_refs))
        self.logger.info(f"Found {len(unique_refs)} unique dataset_refs after initial deduplication.")

        image_metadata_list = []
        if not unique_refs:
            self.logger.info("No unique dataset references found, returning empty list of ImageMetadata.")
            return []

        self.logger.info(f"Fetching metadata for {len(unique_refs)} unique dataset_refs.")
        for ref_idx, ref in enumerate(unique_refs):
            try:
                self.logger.debug(f"Processing ref {ref_idx + 1}/{len(unique_refs)}: {ref.dataId}")
                visit_id = ref.dataId['visit']
                detector_id = ref.dataId['detector']

                band_name = ref.dataId['band']

                visit_info = self.butler.get("visit_image.visitInfo", visit=visit_id, detector=detector_id)
    
                t_min = visit_info.date.toAstropy()
                exp_time = visit_info.exposureTime
                t_max = t_min + TimeDelta(exp_time, format='sec')
                central_ra = visit_info.boresightRaDec.getRa().asDegrees()
                central_dec = visit_info.boresightRaDec.getDec().asDegrees()
                
                relevant_eph_for_image = []
                exact_eph_for_image = None

                if ephemeris:
                    relevant_eph_for_image = EphemerisDataCompressed.get_relevant_rows(
                        ephemeris, t_min, t_max
                    )
                    self.logger.debug(f"Ref {ref.dataId}: Found {len(relevant_eph_for_image)} relevant ephem rows for exposure {t_min.isot} - {t_max.isot}")

                    if relevant_eph_for_image:
                        t_mid_mjd = (t_min.mjd + t_max.mjd) / 2.0
                        
                        if len(relevant_eph_for_image) == 1:
                            interp_ra = relevant_eph_for_image[0].ra_deg
                            interp_dec = relevant_eph_for_image[0].dec_deg
                            exact_eph_for_image = relevant_eph_for_image[0] # Use the single point as "exact"
                            self.logger.debug(f"Ref {ref.dataId}: Using single ephem point for exact_eph: {exact_eph_for_image}")
                        else: # len > 1, interpolate
                            # Use first and last points from the *relevant subset* for interpolation span
                            p_start = relevant_eph_for_image[0]
                            p_end = relevant_eph_for_image[-1]
                            
                            self.logger.debug(f"Ref {ref.dataId}: Interpolating using start={p_start.datetime.isot}, end={p_end.datetime.isot} for t_mid={Time(t_mid_mjd, format='mjd', scale='tai').isot}")

                            interp_ra, interp_dec, _ = interpolate_coordinates(
                                p_start.ra_deg, p_start.dec_deg,
                                p_end.ra_deg, p_end.dec_deg,
                                p_start.datetime.mjd, p_end.datetime.mjd,
                                t_mid_mjd
                            )
                            
                            # Find the point in relevant_eph_for_image closest to t_mid_mjd for rates/uncertainty
                            closest_eph_for_aux_data = min(relevant_eph_for_image, key=lambda eph: abs(eph.datetime.mjd - t_mid_mjd))
                            exact_eph_for_image = EphemerisDataCompressed(
                                datetime=Time(t_mid_mjd, format='mjd', scale='tai'),
                                ra_deg=interp_ra, dec_deg=interp_dec,
                                ra_rate=closest_eph_for_aux_data.ra_rate,
                                dec_rate=closest_eph_for_aux_data.dec_rate,
                                # uncertainty=closest_eph_for_aux_data.uncertainty
                                uncertainty={         
                                    "rss": closest_eph_for_aux_data.uncertainty["rss"],
                                    "smaa": closest_eph_for_aux_data.uncertainty["smaa"],
                                    "smia": closest_eph_for_aux_data.uncertainty["smia"],
                                    "theta": closest_eph_for_aux_data.uncertainty["theta"],
                                },
                            )
                            self.logger.debug(f"Ref {ref.dataId}: Interpolated exact_eph: {exact_eph_for_image}, aux from: {closest_eph_for_aux_data.datetime.isot}")
                    else:
                        self.logger.warning(f"Ref {ref.dataId}: No relevant ephemeris data found for exposure {t_min.isot} - {t_max.isot}. Exact ephemeris will be None.")
                else:
                    self.logger.info("No global ephemeris data provided ('ephemeris' is empty). Skipping ephemeris processing for images.")

                image_meta = ImageMetadata(
                    visit_id=visit_id,
                    detector_id=detector_id,
                    band=band_name,
                    coordinates_central=(central_ra, central_dec),
                    t_min=t_min,
                    t_max=t_max,
                    ephemeris_data=relevant_eph_for_image,
                    exact_ephemeris=exact_eph_for_image
                )
                image_metadata_list.append(image_meta)
                self.logger.debug(f"Successfully created ImageMetadata for visit_id {visit_id}")

            except Exception as e:
                self.logger.error(f"Error processing dataset ref {ref.dataId if ref else 'N/A'} (index {ref_idx}): {e}", exc_info=True)
                # Continue to next ref if one fails
                continue
        
        self.logger.info(f"Created {len(image_metadata_list)} ImageMetadata objects before final deduplication.")

        # Deduplicate ImageMetadata objects based on visit_id + detector_id

        if not image_metadata_list:
            return []
        
        unique_metadata_dict = {(item.visit_id, item.detector_id): item for item in image_metadata_list}
        unique_metadata_list = list(unique_metadata_dict.values())
        
        self.logger.info(f"Returning {len(unique_metadata_list)} unique ImageMetadata objects after (visit_id, detector_id) deduplication.")
        return unique_metadata_list
