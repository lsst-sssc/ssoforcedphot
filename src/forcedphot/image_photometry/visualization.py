import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.coordinates import SkyCoord
import lsst.geom as geom
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS


def create_diagnostic_plot(image_exposure, target_skycoord, x_offset, y_offset, error_ellipse_obj, nearby_skycoords, output_filepath, title=None):
    """
    Generates and saves a diagnostic plot for forced photometry.

    Parameters
    ----------
    image_exposure : lsst.afw.image.ExposureF
        The image exposure to display.
    target_skycoord : astropy.coordinates.SkyCoord
        The sky coordinates of the target object.
    x_offset : int
        Pixel offset in case of cutout.
    y_offset : int
        Pixel offset in case of cutout.
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

        # Create an Astropy WCS object from the FITS metadata.
        astropy_wcs = WCS(wcs.getFitsMetadata())

        # Apply the pixel offsets to the WCS reference pixel.
        # This adjusts the WCS to the coordinate system of the cutout image.
        astropy_wcs.wcs.crpix[0] -= x_offset
        astropy_wcs.wcs.crpix[1] -= y_offset

        fig = plt.figure()
        # Use the corrected WCS for the plot projection.
        ax = plt.subplot(projection=astropy_wcs)
        fig.add_axes(ax)

        # Display the image
        image_array = image_exposure.image.array
        norm = ImageNormalize(image_array, interval=ZScaleInterval())
        ax.imshow(image_array, origin='lower', cmap='gray', norm=norm)

        # Plot target
        ax.plot(target_skycoord.ra.deg, target_skycoord.dec.deg, 'o', markersize=3, markerfacecolor='none', markeredgewidth=0.5, markeredgecolor='red',label='Target', transform=ax.get_transform('world'))

        # Plot error ellipse        
        ellipse_ra, ellipse_dec = error_ellipse_obj.get_ellipse_points()
        ax.plot(ellipse_ra, ellipse_dec, 'r--', linewidth = 0.5, label='Error Ellipse', transform=ax.get_transform('world'))

        # Plot nearby sources
        nearby_ra_degrees = []
        nearby_dec_degrees = []
        for nearby_coord in nearby_skycoords:
            nearby_ra_degrees.append(nearby_coord.ra.deg)
            nearby_dec_degrees.append(nearby_coord.dec.deg)
        
        if nearby_ra_degrees: # Only plot if there are nearby sources
            ax.plot(nearby_ra_degrees, nearby_dec_degrees, 'bo', markersize=3, mfc='none', markeredgewidth=0.5, label='Nearby Sources', transform=ax.get_transform('world'))

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
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
