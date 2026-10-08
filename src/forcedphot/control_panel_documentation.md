## Panel application
The ODC (Object Detection Controller) can also be controlled via a Panel-based ([https://panel.holoviz.org/]) web application, providing an interactive graphical user interface (GUI).
The application is divided into six main tabs, each handling a specific part of the detection and analysis process.

### Main Features
- Ephemeris data generation and management.
- Image search and filtering based on ephemeris.
- Photometric analysis on retrieved images, including PSF forced photometry and optional aperture photometry.
- Complete end-to-end pipeline execution combining all steps.
- Standalone photometry at arbitrary coordinates, without requiring ephemeris data.
- Real-time logging output directly within the application (with a note on potential display delays).

### Tab Descriptions

#### 1. Ephemeris Query Tab
This tab handles the generation and management of ephemeris data for your target object.

##### Key Parameters:
- **Ephemeris Source**: Choose between "Use Existing Data" (to perform a live query) or "Upload ECSV" (to use pre-computed ephemeris from a file).
- **Service**: Select the ephemeris service for live queries (Horizons or Miriade).
- **Target Name**: Enter the name of the astronomical target (e.g., "C/2020 F3 (NEOWISE)").
- **Target Type**: Choose the classification of your target (e.g., `smallbody`, `asteroid_name`, `comet_name`, `designation`).
- **Time Parameters**:
  - `Start Time`: Set the beginning of the observation period.
  - `Time Specification`: Choose between defining an `End Time` or a `Day Range` (number of days forward from the start time).
  - `Step Value` and `Unit`: Define the time resolution for the ephemeris data points (e.g., `1` `h` for hourly steps).
- **Save Ephemeris Data**: Check this box to save the queried ephemeris data to the specified `Output Folder`.

##### Output:
Displays a table with detailed ephemeris data, including:
- `Time (ISO)`: Observation time in ISO format.
- `RA (deg)`, `Dec (deg)`: Right Ascension and Declination coordinates in degrees.
- `RA Rate (arcsec/h)`, `Dec Rate (arcsec/h)`: Rates of change in RA and Dec in arcseconds per hour.
- `AZ (deg)`, `EL (deg)`: Azimuth and Elevation angles in degrees.
- `r (au)`: Heliocentric distance in Astronomical Units.
- `Delta (au)`: Geocentric distance in Astronomical Units.
- `V mag`: Visual magnitude.
- `Alpha (deg)`: Phase angle in degrees.
- `RSS 3σ (arcsec)`, `SMAA 3σ (arcsec)`, `SMIA 3σ (arcsec)`, `Theta 3σ (deg)`: 3-sigma uncertainty information including Root Sum Square, semi-major/minor axes, and position angle of the error ellipse.

#### 2. Image Search Tab
This tab handles searching for LSST images based on the ephemeris data.

##### Key Parameters:
- **Image Search Method**:
  - `Point`: Searches for images that overlap with individual ephemeris data points (can be slower for long ephemeris).
  - `Polygon`: Creates a polygon based on the ephemeris track and searches for images overlapping this area (generally faster for longer ephemeris data).
- **Filters (Bands)**: Select one or more photometric bands (`u`, `g`, `r`, `i`, `z`, `y`) for the image search.
- **Polygon Search Options** (visible only if "Polygon" method is selected):
  - `Widening (arcsec)`: Expands the polygon search area around the ephemeris track by this amount in arcseconds.
  - `Time Interval (days)`: Groups ephemeris points into segments for polygon creation within this time interval.

##### Output:
Displays a table summarizing the metadata of the found images:
- `Visit ID`: Unique identifier for the observation visit.
- `Detector ID`: Identifier for the specific detector used.
- `Filter`: The photometric band of the image.
- `Obs Start (MJD)`, `Obs End (MJD)`: The start and end times of the observation in Modified Julian Date (MJD), converted to ISO format for display.

#### 3. Photometry Analysis Tab
This tab configures and executes the photometric analysis on the images found by the "Image Search" tab.

##### Key Parameters:
- **Image Type**: Choose the type of image to process: `"visit_image"` (calibrated raw image) or `"difference_image"` (difference image).
- **Detection Threshold (SNR)**: Set the Signal-to-Noise Ratio (SNR) threshold for detecting sources.
- **Cutout Provider**: Choose the backend for image cutouts:
  - `Butler (local)`: Loads the full exposure in memory and slices it locally (default).
  - `SODA (remote)`: Requests server-side cutouts from the IVOA SODA service, avoiding loading the full image.
- **Cutout Size (pixels)** (visible when using Butler): Define the square size of the image cutout in pixels (e.g., `800` for 800×800). Set to `0` to use the entire image.
- **Cutout Radius (arcsec)** (visible when using SODA): Define the cutout radius in arcseconds for the SODA server-side cutout request.
- **Override Error Ellipse (arcsec)**: If a value greater than `0` is provided, the ephemeris-derived error ellipse will be overridden by a circular error region of this radius (in arcseconds) for source extraction.
- **Aperture Photometry**: Check to enable aperture photometry in addition to PSF forced photometry.
- **Aperture Radii (arcsec)** (enabled when Aperture Photometry is checked): List of aperture radii in arcseconds (e.g., `[3.0, 5.0, 7.0]`). Edit the list directly to customize the radii.
- **Refine Ephemeris at Observation Times**: When enabled, queries precise ephemeris at exact observation times instead of using linear interpolation. This eliminates interpolation errors and significantly improves accuracy for fast-moving objects like Near-Earth Objects (NEOs) and comets. Note that enabling this option increases processing time as it performs additional ephemeris queries.
- **Output Options**:
  - `Save Diagnostic Plots (PNG)`: Check to save PNG images marked with detected sources and the error ellipse.
  - `Save FITS Cutouts`: Check to save the FITS image cutouts used for photometry.
  - `Display Results in Firefly`: Check to attempt displaying the images and photometry results in a Firefly viewer.
  - `Save Results to JSON`: Check to save the final photometry results in JSON format.
  - `Save Results to CSV`: Check to save the final photometry results in CSV format.
  - `Include All Sources within Error Ellipse in CSV` (visible only if `Save Results to CSV` is checked): If enabled, creates separate rows in the CSV for each source detected within the error ellipse; otherwise, only the forced photometry target row is written.

##### Results Display:
The table summarizes photometry measurements for the target object in each processed image:
- `Visit ID`, `Detector ID`, `Band`: Identifiers for the image.
- `Flux (nJy)`, `Flux Error`: Measured flux and its uncertainty in nanoJansky.
- `Magnitude`, `Mag Error`: Measured AB magnitude and its uncertainty.
- `SNR`: Signal-to-Noise Ratio of the target detection.
- `Nearby Sources`: Count of additional sources detected within the error ellipse.

#### 4. Complete Run Tab
This tab allows you to execute the entire object detection pipeline from ephemeris query to photometry in a single, integrated workflow.

##### Features:
- Combines all parameters from the "Ephemeris Query", "Image Search", and "Photometry Analysis" tabs into one interface, organized into collapsible cards.
- Provides single-click execution of the entire end-to-end workflow using the "Run All" button.
- Supports all the same photometry options as the Photometry Analysis tab, including aperture photometry.

##### Key Parameters (Photometry Settings card):
- **Image Type**, **Detection Threshold**, **Cutout Provider**, **Cutout Size / Cutout Radius**: Same as Photometry Analysis tab.
- **Aperture Photometry** and **Aperture Radii (arcsec)**: Enable and configure aperture photometry alongside PSF forced photometry.
- **Override Error Ellipse**, **Refine Ephemeris**: Same as Photometry Analysis tab.
- **Output options**: Same save options as the Photometry Analysis tab.

##### Usage:
1. Configure all parameters across the Ephemeris Settings, Image Search Settings, and Photometry Settings cards.
2. Click the "Run All" button to start the complete pipeline execution.
3. The final photometry results will be displayed in the table at the bottom of the tab.

#### 5. Standalone Photometry Tab
This tab performs forced photometry at arbitrary sky coordinates without any ephemeris dependency. It is useful for re-measuring previously identified positions, measuring transients, or any non-SSO science case.

##### Input Modes:
Select the input mode from the **Input Mode** dropdown:

- **Single Coordinate**: Measure photometry at a single RA/Dec position in one specified image.
  - `Visit ID`, `Detector ID`, `Band`: Identify the image.
  - `RA (degrees)`, `Dec (degrees)`: Target coordinates.

- **Batch CSV**: Process a list of coordinates and images from an uploaded CSV file.
  - Upload a `.csv` file with the following columns:
    - Required: `visit_id`, `detector`, `band`, `ra`, `dec`
    - Optional: `error_radius`, `target_name`, `aperture_radii` (comma-separated values in a single cell, e.g. `3.0,5.0,7.0`)
  - When `aperture_radii` is provided per row in the CSV, those values are used instead of the global Aperture Radii setting.

- **Multiple in Image**: Measure photometry for multiple RA/Dec coordinates in the same image.
  - `Visit ID`, `Detector ID`, `Band`: Identify the shared image.
  - `Coordinates (RA, Dec - one per line)`: Enter coordinates as `ra, dec` pairs, one per line.

##### Common Parameters:
- **Error Radius (arcsec)**: Search radius around the target coordinate for nearby source detection.
- **Detection Threshold (SNR)**: SNR threshold for source detection within the error radius.
- **Image Type**: `"visit_image"` or `"difference_image"`.
- **Aperture Photometry**: Check to enable aperture photometry in addition to PSF forced photometry.
- **Aperture Radii (arcsec)** (enabled when Aperture Photometry is checked): List of aperture radii in arcseconds (e.g., `[3.0, 5.0, 7.0]`).
- **Cutout Provider**: `Butler (local)` or `SODA (remote)` — same as in the Photometry Analysis tab.
- **Cutout Size (pixels)** (Butler) / **Cutout Radius (arcsec)** (SODA): Size of the image cutout.

##### Output Options:
- `Save Diagnostic Plots`: Save PNG diagnostic images.
- `Save FITS Cutouts`: Save FITS cutout files.
- `Save Results CSV`: Save results to a CSV file.
- `Save Results JSON`: Save results to a JSON file.
- `Save all sources within error ellipse` (visible when `Save Results CSV` is checked): Include all detected sources within the error radius as separate rows in the CSV.
- `Output Folder`: Directory where all output files are saved.

##### Results Display:
An interactive table showing the photometry results for all processed coordinates.

### Using the Application

#### Basic Workflow:
1.  **Generate Ephemeris**: Navigate to the "Ephemeris Query" tab. Select your ephemeris source (live query or ECSV upload), enter target details, and define time parameters. Click "Run Query". The results will populate the table.
2.  **Search for Images**: Go to the "Image Search" tab. Select desired filters and the image search method. Click "Run Image Query". This step will automatically use the ephemeris data obtained in the previous step (or from the uploaded ECSV). Image metadata will appear in the table.
3.  **Perform Photometry**: Switch to the "Photometry Analysis" tab. Configure detection parameters, cutout size, aperture photometry options, and output preferences. Click "Run Photometry". This step will process the image metadata obtained in the previous "Image Search" step. Photometry results will be displayed.
4.  **Complete Pipeline**: For an integrated, hands-off workflow, use the "Complete Run" tab. Configure all parameters for ephemeris, image search, and photometry in one place. Then, execute the entire pipeline with a single click on "Run All". The final photometry results will be shown in the table at the bottom of this tab.
5.  **Standalone Photometry**: To measure flux at known coordinates without ephemeris, use the "Standalone Photometry" tab. Select an input mode, provide coordinates and image identifiers, configure parameters, and click "Run Standalone Photometry".

#### Additional Features:
-   **Terminal Output**: View application logs, progress messages, and system output in the terminal widget at the bottom of the application. Note that output in the web browser might be slightly delayed compared to immediate command-line execution.
-   **Test Functions**: Use the "Test Logging" and "Test Terminal" buttons to verify that the logging and terminal display functionality is working correctly.
-   **Clear Terminal**: Use the "Clear Terminal" button to clear the contents of the terminal display.
-   **Table Views**: All data output in tables (`pn.widgets.Tabulator`) are interactive, allowing for sorting columns and filtering data.

## Troubleshooting
-   If an ephemeris query fails, verify the target ID/name, target type, and ensure that the start/end times or day range are correctly specified and within reasonable limits.
-   Check the time format (should be `"YYYY-MM-DD HH:MM:SS"`) if you encounter time-related errors.
-   Ensure that the `Output Folder` exists and has write permissions if you are saving any output files.
-   For queries covering very large time ranges or requiring high time resolution, consider using larger `Step Value`s to reduce the amount of ephemeris data generated.
-   If the generated or loaded ephemeris data is excessively large, the subsequent image search step might become slow or fail due to memory constraints.
-   When running services within the Panel application, the terminal output may appear with a slight delay. Please be patient for messages to propagate to the browser.
-   If the "Test Terminal" button does not produce output, it might indicate that a long-running process is currently occupying the backend, preventing immediate updates. Wait for the ongoing process to complete.
