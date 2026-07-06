"""
Cutout Service Module

Provides a pluggable cutout system with two backends:
- Butler (local, in-memory): Extracts sub-images from already-loaded ExposureF objects.
- SODA (remote, server-side): Requests cutouts from the RSP SODA image cutout service.

Both backends produce an lsst.afw.image.ExposureF that the photometry pipeline
can consume without modification.

Main Components:
    - CutoutResult: Standardized result dataclass from any cutout provider.
    - CutoutProvider: Protocol defining the cutout interface.
    - ButlerCutoutProvider: In-memory pixel slicing from a pre-loaded ExposureF.
    - SodaCutoutProvider: Remote SODA-based cutout via DataLink + SodaQuery.
    - CutoutService: Factory/manager that delegates to the active provider.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import lsst.afw.image as afwimage
import lsst.geom as geom
import numpy as np


@dataclass
class CutoutResult:
    """Standardized result from any cutout provider.

    Attributes
    ----------
    exposure : ExposureF or None
        The cutout image. None if the cutout failed.
    bbox : geom.Box2I or None
        Bounding box in parent coordinates (Butler only, None for SODA).
    x_offset : float
        Pixel X offset of the cutout origin relative to the parent image.
        0.0 for SODA cutouts (standalone image).
    y_offset : float
        Pixel Y offset of the cutout origin relative to the parent image.
        0.0 for SODA cutouts (standalone image).
    cutout_size_px : int
        Actual cutout size in pixels.
    cutout_size_arcsec : float or None
        Angular size in arcseconds, if applicable.
    provider : str
        Name of the provider that created this cutout ("butler" or "soda").
    success : bool
        Whether the cutout creation succeeded.
    message : str
        Status or error message.
    """

    exposure: Any  # Optional[ExposureF] - use Any to avoid import issues at class level
    bbox: Any  # Optional[geom.Box2I]
    x_offset: float
    y_offset: float
    cutout_size_px: int
    cutout_size_arcsec: Optional[float]
    provider: str
    success: bool
    message: str


# SODA needs an explicit angular cutout size. When the caller requests the full image
# (cutout_size == 0) we pass an intentionally over-large CIRCLE radius: 1560 arcsec
# (0.433 deg) comfortably exceeds the half-diagonal of a single LSSTCam/ComCam science
# CCD (~580 arcsec at 0.2 arcsec/px), so SODA clips the circle to the detector bounds
# and returns the entire detector image. Any value above ~580 arcsec works; 1560 is a
# safe round margin, not a physically derived constant.
DEFAULT_SODA_FULL_RADIUS_ARCSEC = 1560


class CutoutProvider(Protocol):
    """Abstract interface for cutout providers."""

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

    def get_cutout(
        self,
        ra_deg: float,
        dec_deg: float,
        cutout_size: float,
        **kwargs: Any,
    ) -> CutoutResult:
        """Create a cutout at the given coordinates."""
        ...


class ButlerCutoutProvider:
    """Cutout provider using Butler in-memory pixel slicing.

    Extracts a sub-image from an already-loaded ExposureF using WCS
    sky-to-pixel conversion and bounding box construction. Preserves
    the original pixel coordinate system via PARENT origin.
    """

    def __init__(self):
        self._name = "butler"
        self.logger = logging.getLogger("cutout_service.butler")

    @property
    def name(self) -> str:
        """Return the provider name."""
        return self._name

    def get_cutout(
        self,
        ra_deg: float,
        dec_deg: float,
        cutout_size: float,
        **kwargs: Any,
    ) -> CutoutResult:
        """Create cutout from an already-loaded ExposureF.

        Parameters
        ----------
        ra_deg : float
            Right Ascension in degrees.
        dec_deg : float
            Declination in degrees.
        cutout_size : float
            Cutout size in pixels.
        **kwargs
            Must include ``exposure`` (ExposureF): the pre-loaded full image.

        Returns
        -------
        CutoutResult
            The cutout result with preserved WCS and pixel offsets.
        """
        exposure = kwargs.get("exposure")
        if exposure is None:
            return CutoutResult(
                exposure=None,
                bbox=None,
                x_offset=0,
                y_offset=0,
                cutout_size_px=0,
                cutout_size_arcsec=None,
                provider=self.name,
                success=False,
                message="Butler provider requires 'exposure' kwarg",
            )

        cutout_size = int(cutout_size)
        wcs = exposure.getWcs()
        coord = geom.SpherePoint(np.radians(ra_deg), np.radians(dec_deg), geom.radians)

        pixel_coord = wcs.skyToPixel(coord)
        x_center, y_center = pixel_coord.getX(), pixel_coord.getY()
        half_size = cutout_size // 2

        # Edge cutoff (pixels): reject targets closer than this to any image edge
        # so the cutout / PSF isn't truncated.
        x_cutoff = y_cutoff = 10

        min_x, max_x = 0, exposure.getWidth()
        min_y, max_y = 0, exposure.getHeight()

        # Check if target is within 10 pixels of any edge
        if (
            x_center < x_cutoff
            or x_center > (max_x - x_cutoff)
            or y_center < y_cutoff
            or y_center > (max_y - y_cutoff)
        ):
            self.logger.warning(
                f"Target is within {x_cutoff} px (x) / {y_cutoff} px (y) of the image edge "
                "or outside the boundaries. Skipping image."
            )
            return CutoutResult(
                exposure=None,
                bbox=None,
                x_offset=0,
                y_offset=0,
                cutout_size_px=0,
                cutout_size_arcsec=None,
                provider=self.name,
                success=False,
                message="Target within 10px of image edge or outside boundaries",
            )

        if cutout_size <= 0:
            self.logger.info("cutout_size <= 0; using the complete image.")
            return CutoutResult(
                exposure=exposure,
                bbox=None,
                x_offset=0,
                y_offset=0,
                cutout_size_px=max(max_x, max_y),
                cutout_size_arcsec=None,
                provider=self.name,
                success=True,
                message="Using complete image (cutout_size <= 0)",
            )

        if (
            (x_center - half_size) < min_x
            or (x_center + half_size) > max_x
            or (y_center - half_size) < min_y
            or (y_center + half_size) > max_y
        ):
            self.logger.info("Cutout boundaries fall outside the image; using the complete image.")
            return CutoutResult(
                exposure=exposure,
                bbox=None,
                x_offset=0,
                y_offset=0,
                cutout_size_px=max(max_x, max_y),
                cutout_size_arcsec=None,
                provider=self.name,
                success=True,
                message="Cutout extends beyond image bounds, using complete image",
            )

        self.logger.info(f"Creating cutout with size {cutout_size} pixels")

        # Create bounding box for cutout
        bbox = geom.Box2I()
        bbox.include(geom.Point2I(float(x_center - half_size), float(y_center - half_size)))
        bbox.include(geom.Point2I(float(x_center + half_size), float(y_center + half_size)))

        # Create cutout with PARENT origin to preserve WCS
        target_img = exposure.Factory(exposure, bbox, origin=afwimage.PARENT, deep=False)

        return CutoutResult(
            exposure=target_img,
            bbox=bbox,
            x_offset=bbox.getMinX(),
            y_offset=bbox.getMinY(),
            cutout_size_px=cutout_size,
            cutout_size_arcsec=None,
            provider=self.name,
            success=True,
            message="Butler cutout created successfully",
        )


class SodaCutoutProvider:
    """Cutout provider using the RSP SODA image cutout service.

    Requests server-side cutouts via the IVOA SODA protocol. The full image
    never enters client memory -- only the cutout region is transferred.
    Cutout size is specified in arcseconds (angular units).
    """

    def __init__(self, dr: str = "dp1"):
        self._name = "soda"
        self._dr = dr
        self._session = None
        self.logger = logging.getLogger("cutout_service.soda")

    @property
    def name(self) -> str:
        """Return the provider name."""
        return self._name

    def _get_session(self):
        """Get or create the PyVO authentication session."""
        if self._session is None:
            from lsst.rsp.utils import get_pyvo_auth

            self._session = get_pyvo_auth()
        return self._session

    def _find_image_access_url(
        self,
        visit_id: int,
        detector_id: int,
        band: str,
    ) -> Optional[str]:
        """Find the SODA-compatible access_url for a specific image via ObsTAP.

        Parameters
        ----------
        visit_id : int
            LSST visit identifier.
        detector_id : int
            LSST detector identifier.
        band : str
            Photometric band.

        Returns
        -------
        str or None
            The access_url for the image, or None if not found.
        """
        from lsst.rsp import get_tap_service

        tap = get_tap_service("tap")
        query = f"""
        SELECT access_url
        FROM ivoa.ObsCore
        WHERE lsst_visit = {visit_id}
          AND lsst_detector = {detector_id}
          AND lsst_band = '{band}'
          AND calib_level = 2
        """
        job = tap.submit_job(query)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR"])
        job.raise_if_error()
        results = job.fetch_result().to_table()

        if len(results) == 0:
            return None
        return str(results[0]["access_url"])

    def _fits_bytes_to_exposure(self, raw_bytes: bytes):
        """Convert raw FITS bytes from SODA into an lsst.afw.image.ExposureF.

        First tries the native LSST reader (MemFileManager). If that fails
        (e.g. the SODA service returns a standard FITS rather than the
        multi-extension LSST format), falls back to reading with
        astropy.io.fits and constructing an ExposureF from the image data
        and WCS header.

        Parameters
        ----------
        raw_bytes : bytes
            Raw FITS file content from the SODA cutout response.

        Returns
        -------
        lsst.afw.image.ExposureF
            The cutout as an LSST ExposureF object with WCS set.
        """
        # Attempt 1: native LSST reader (works if SODA returns LSST format)
        try:
            from lsst.afw.fits import MemFileManager
            from lsst.afw.image import ExposureF

            mem = MemFileManager(len(raw_bytes))
            mem.setData(raw_bytes, len(raw_bytes))
            return ExposureF(mem)
        except Exception:
            self.logger.info(
                "SODA FITS is not in LSST multi-extension format, falling back to astropy reader."
            )

        # Attempt 2: standard FITS via astropy -> ExposureF
        import io

        from astropy.io import fits as astropy_fits
        from lsst.afw.geom import makeSkyWcs
        from lsst.afw.image import ExposureF, MaskedImageF
        from lsst.daf.base import PropertyList

        hdulist = astropy_fits.open(io.BytesIO(raw_bytes))

        # Find the image HDU (primary or first image extension)
        image_data = None
        image_header = None
        for hdu in hdulist:
            if hdu.data is not None and hdu.data.ndim == 2:
                image_data = hdu.data
                image_header = hdu.header
                break

        if image_data is None:
            raise ValueError("No 2D image data found in SODA FITS response")

        # Create MaskedImageF from the pixel data
        ny, nx = image_data.shape
        masked_image = MaskedImageF(nx, ny)
        masked_image.image.array[:] = image_data.astype(np.float32)

        # Create ExposureF
        exposure = ExposureF(masked_image)

        # Set WCS from FITS header
        import contextlib

        metadata = PropertyList()
        for key in image_header:
            if key and key not in ("COMMENT", "HISTORY", ""):
                with contextlib.suppress(Exception):
                    metadata.set(key, image_header[key])
        try:
            wcs = makeSkyWcs(metadata)
            exposure.setWcs(wcs)
        except Exception as wcs_err:
            self.logger.warning(f"Could not set WCS from SODA FITS header: {wcs_err}")

        hdulist.close()
        return exposure

    def get_cutout(
        self,
        ra_deg: float,
        dec_deg: float,
        cutout_size: float,
        **kwargs: Any,
    ) -> CutoutResult:
        """Create cutout via the RSP SODA service.

        Parameters
        ----------
        ra_deg : float
            Right Ascension in degrees.
        dec_deg : float
            Declination in degrees.
        cutout_size : float
            Cutout radius in arcseconds.
        **kwargs
            Optional keys:
            - ``access_url`` (str): Direct SODA access URL.
            - ``visit_id`` (int), ``detector_id`` (int), ``band`` (str):
              Used to look up access_url via ObsTAP if not provided directly.

        Returns
        -------
        CutoutResult
            The cutout result. offsets are (0, 0) and bbox is None since
            SODA cutouts are standalone images.
        """
        import astropy.units as u

        try:
            session = self._get_session()

            # Step 1: Resolve access_url if not provided
            access_url = kwargs.get("access_url")
            if access_url is None:
                visit_id = kwargs.get("visit_id")
                detector_id = kwargs.get("detector_id")
                band = kwargs.get("band")
                if not all([visit_id is not None, detector_id is not None, band is not None]):
                    return CutoutResult(
                        exposure=None,
                        bbox=None,
                        x_offset=0,
                        y_offset=0,
                        cutout_size_px=0,
                        cutout_size_arcsec=cutout_size,
                        provider=self.name,
                        success=False,
                        message="SODA requires access_url or (visit_id, detector_id, band)",
                    )
                access_url = self._find_image_access_url(visit_id, detector_id, band)
                if access_url is None:
                    return CutoutResult(
                        exposure=None,
                        bbox=None,
                        x_offset=0,
                        y_offset=0,
                        cutout_size_px=0,
                        cutout_size_arcsec=cutout_size,
                        provider=self.name,
                        success=False,
                        message=(
                            f"No access_url found for visit={visit_id}, detector={detector_id}, band={band}"
                        ),
                    )

            # Step 2: Build DataLink
            from pyvo.dal.adhoc import DatalinkResults, SodaQuery

            dl_result = DatalinkResults.from_result_url(access_url, session=session)

            # Step 3: Build and configure SODA query
            sq = SodaQuery.from_resource(
                dl_result,
                dl_result.get_adhocservice_by_id("cutout-sync-exposure"),
                session=session,
            )
            # Convert cutout_size (arcsec) to degrees for SODA CIRCLE parameter
            radius_deg = DEFAULT_SODA_FULL_RADIUS_ARCSEC / 3600.0 if cutout_size == 0 else cutout_size / 3600.0
            sq.circle = (ra_deg * u.deg, dec_deg * u.deg, radius_deg * u.deg)

            # Step 4: Execute and get FITS bytes
            raw_bytes = sq.execute_stream().read()

            # Step 5: Deserialize to ExposureF
            cutout_exposure = self._fits_bytes_to_exposure(raw_bytes)

            # Step 6: Check PSF preservation
            if not cutout_exposure.hasPsf():
                self.logger.warning(
                    "SODA cutout does NOT include PSF model. "
                    "Forced photometry accuracy may be degraded. "
                    "Consider using Butler cutout for PSF-dependent measurements."
                )

            # Estimate pixel size from the cutout
            cutout_size_px_approx = int(cutout_exposure.getWidth())

            return CutoutResult(
                exposure=cutout_exposure,
                bbox=None,
                x_offset=0.0,
                y_offset=0.0,
                cutout_size_px=cutout_size_px_approx,
                cutout_size_arcsec=cutout_size,
                provider=self.name,
                success=True,
                message="SODA cutout created successfully",
            )

        except Exception as e:
            self.logger.error(f"SODA cutout failed: {e}", exc_info=True)
            return CutoutResult(
                exposure=None,
                bbox=None,
                x_offset=0,
                y_offset=0,
                cutout_size_px=0,
                cutout_size_arcsec=cutout_size,
                provider=self.name,
                success=False,
                message=f"SODA cutout failed: {e!s}",
            )


class CutoutService:
    """Factory/manager for cutout operations.

    Delegates to the configured CutoutProvider. Supports switching
    between providers at runtime via ``set_provider()``.

    Parameters
    ----------
    provider : str
        Initial provider name: "butler" or "soda".
    dr : str
        Data release identifier for the SODA provider.
    """

    PROVIDERS = {"butler", "soda"}

    def __init__(
        self,
        provider: str = "butler",
        dr: str = "dp1",
    ):
        self.logger = logging.getLogger("cutout_service")
        self._dr = dr
        self._providers: dict[str, Any] = {
            "butler": ButlerCutoutProvider(),
            "soda": SodaCutoutProvider(dr=dr),
        }
        self._active_provider_name = provider

    @property
    def active_provider(self) -> str:
        """Return the name of the currently active provider."""
        return self._active_provider_name

    def set_provider(self, provider_name: str):
        """Switch the active cutout provider.

        Parameters
        ----------
        provider_name : str
            "butler" or "soda".
        """
        if provider_name not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider_name}. Choose from {self.PROVIDERS}")
        self._active_provider_name = provider_name
        self.logger.info(f"Cutout provider set to: {provider_name}")

    def get_cutout(self, ra_deg: float, dec_deg: float, cutout_size: float, **kwargs: Any) -> CutoutResult:
        """Delegate cutout creation to the active provider.

        Parameters
        ----------
        ra_deg : float
            Right Ascension in degrees.
        dec_deg : float
            Declination in degrees.
        cutout_size : float
            Size in pixels (butler) or arcseconds (soda).
        **kwargs
            Provider-specific arguments (e.g., exposure, visit_id, etc.).

        Returns
        -------
        CutoutResult
            Standardized cutout result.
        """
        provider = self._providers[self._active_provider_name]
        return provider.get_cutout(ra_deg, dec_deg, cutout_size, **kwargs)
