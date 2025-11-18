import lsst.geom as geom
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS


def create_diagnostic_plot(
    image_exposure,
    target_skycoord,
    x_offset,  # noqa: ARG001 - kept for API compatibility, no longer used
    y_offset,  # noqa: ARG001 - kept for API compatibility, no longer used
    error_ellipse_obj,
    nearby_skycoords,
    output_filepath,
    title=None,
):
    """
    Generates and saves a diagnostic plot for forced photometry.

    Parameters
    ----------
    image_exposure : lsst.afw.image.ExposureF
        The image exposure to display.
    target_skycoord : astropy.coordinates.SkyCoord
        The sky coordinates of the target object.
    x_offset : float
        Legacy parameter, no longer used. Kept for API compatibility.
    y_offset : float
        Legacy parameter, no longer used. Kept for API compatibility.
    error_ellipse_obj : image_photometry.utils.ErrorEllipse
        The error ellipse of the target object.
    nearby_skycoords : list[astropy.coordinates.SkyCoord]
        A list of sky coordinates for nearby objects.
    output_filepath : str
        The path to save the output PNG image.
    title : str, optional
        The title for the plot.
    """
    try:
        wcs = image_exposure.getWcs()

        if wcs is None:
            raise ValueError("WCS not found in image_exposure.")

        # Account for image XY0 origin
        # Get the FITS metadata from the original WCS
        astropy_wcs = WCS(wcs.getFitsMetadata())

        # Adjust CRPIX to account for XY0 offset
        xy0 = image_exposure.getXY0()
        astropy_wcs.wcs.crpix[0] -= xy0.getX()
        astropy_wcs.wcs.crpix[1] -= xy0.getY()

        fig = plt.figure()
        # Use the corrected WCS for the plot projection.
        ax = plt.subplot(projection=astropy_wcs)
        fig.add_axes(ax)

        # Display the image
        image_array = image_exposure.image.array
        norm = ImageNormalize(image_array, interval=ZScaleInterval())
        ax.imshow(image_array, origin="lower", cmap="gray", norm=norm)

        # Add WCS gridlines.
        ax.grid(color="black", linestyle="dotted", linewidth=0.5)

        # Plot target - convert using LSST WCS then to display coordinates
        target_sky_point = geom.SpherePoint(
            target_skycoord.ra.deg * geom.degrees, target_skycoord.dec.deg * geom.degrees
        )
        target_pixel_pos = wcs.skyToPixel(target_sky_point)
        target_display_x = target_pixel_pos.getX() - xy0.getX()
        target_display_y = target_pixel_pos.getY() - xy0.getY()

        ax.plot(
            target_display_x,
            target_display_y,
            "o",
            markersize=3,
            markerfacecolor="none",
            markeredgewidth=0.5,
            markeredgecolor="red",
            label="Target",
        )

        # Plot error ellipse - convert using LSST WCS
        ellipse_ra, ellipse_dec = error_ellipse_obj.get_ellipse_points()
        ellipse_pixel_x = []
        ellipse_pixel_y = []
        for ra, dec in zip(ellipse_ra, ellipse_dec):
            sky_point = geom.SpherePoint(ra * geom.degrees, dec * geom.degrees)
            pixel_pos = wcs.skyToPixel(sky_point)
            display_x = pixel_pos.getX() - xy0.getX()
            display_y = pixel_pos.getY() - xy0.getY()
            ellipse_pixel_x.append(display_x)
            ellipse_pixel_y.append(display_y)

        ax.plot(
            ellipse_pixel_x,
            ellipse_pixel_y,
            "r--",
            linewidth=0.5,
            label="Error Ellipse",
        )

        # Plot nearby sources
        # Convert RA/Dec to pixel coordinates using the ORIGINAL LSST WCS
        nearby_pixel_x = []
        nearby_pixel_y = []
        for nearby_coord in nearby_skycoords:
            sky_point = geom.SpherePoint(
                nearby_coord.ra.deg * geom.degrees, nearby_coord.dec.deg * geom.degrees
            )
            pixel_pos = wcs.skyToPixel(sky_point)
            # Adjust pixel position to display coordinates
            display_x = pixel_pos.getX() - xy0.getX()
            display_y = pixel_pos.getY() - xy0.getY()
            nearby_pixel_x.append(display_x)
            nearby_pixel_y.append(display_y)

        if nearby_pixel_x:
            ax.plot(
                nearby_pixel_x,
                nearby_pixel_y,
                "bo",
                markersize=3,
                mfc="none",
                markeredgewidth=0.5,
                label="Nearby Sources",
            )

        # Set labels and title
        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        if title:
            ax.set_title(title)
        ax.legend()

        # Save the plot
        plt.savefig(output_filepath, dpi=150)
        # print(f"Plot saved successfully to {output_filepath}")

    except Exception as e:
        print(f"Error creating or saving diagnostic plot: {e}")
        raise
    finally:
        # Ensure the figure is closed
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
