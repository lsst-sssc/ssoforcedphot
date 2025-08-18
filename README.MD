# Faint Solar System Object Detection Service

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

## Overview
The Faint Solar System Object Detection Service is a Python tool designed for moving solar system object detection and photometry within the Rubin Science Platform. It integrates multiple services including ephemeris queries, image searching, and photometry processing.

## Features
- Ephemeris data retrieval from multiple services (JPL Horizons, Miriade).
- Image search and processing using ObsCore.
- Forced photometry on target coordinates.
- Photometry on nearby sources detected within a specified error ellipse.
- Multiple filter selection for image searches (u, g, r, i, z, y).
- Flexible input options: single query for a target, or batch processing via CSV files.
- Multiple output options: JSON and CSV files for results, diagnostic plots (PNG), and FITS image cutouts.

## Installation Requirements

- Python 3.9 or higher
- Git
- Access to Rubin Science Platform (for full functionality)

On the Rubin Science Platform (RSP):

```bash
# Clone the repository in your RSP environment
git clone https://github.com/lsst-sssc/ssoforcedphot.git
cd ssoforcedphot

# Install in development mode with the RSP environment
pip install -e .
```

## Basic Usage

### Command Line Interface (CLI)
The basic structure of an ODC command is:
```bash
python odc.py [service options] [target options] [time options] [filter options] [output options]
```

### Example Command
This example demonstrates a full pipeline run, from ephemeris query to photometry, with various saving options.
```bash
python odc.py --service-selection all \
              --ephemeris-service Horizons \
              --image-type visit_image \
              --target "2024 TN57" \
              --target-type smallbody \
              --start-time "2024-12-01 00:00:00" \
              --day-range 10 \
              --step 12h \
              --image-search-method point \
              --save-ephem-data \
              --filters u g r i z y \
              --output-folder ./output \
              --save-diag-plots \
              --save-fits \
              --min-cutout-size 800 \
              --all-ellipse-sources \
              --save-json \
              --save-csv
```

## Command Line Options

### Service Selection
- `--service-selection`: Choose which services to use in the pipeline.
  - Options: `"all"`, `"ephemeris"`, `"image"`, `"photometry"`.
  - Default: `"ephemeris"`.

### Ephemeris Options
- `--ephemeris-service`: Select the ephemeris service provider.
  - Options: `"Horizons"`, `"Miriade"`.
  - Default: `"Horizons"`.
- `--target`: Target object identifier (e.g., "2024 TN57" for a single query).
- `--target-type`: Type of target object (e.g., "smallbody", "asteroid_name", "comet_name", "designation").

### Input Options
- `--csv`: Path to a CSV file for batch processing multiple targets. Used by the ephemeris service.
- `--ephem-ecsv`: Path to an ECSV file containing pre-computed ephemeris data. This bypasses the ephemeris query service.

### Time Range Options
- `--start-time`: Start time for the search (format: `"YYYY-MM-DD HH:MM:SS"`).
- `--end-time`: End time for the search (format: `"YYYY-MM-DD HH:MM:SS"`).
- `--day-range`: Number of days to search forward from the start time (alternative to `--end-time`).
- `--step`: Time step for ephemeris query (e.g., `"1h"`, `"10m"`, `"1d"`).
  - Default: `"1h"`.

### Filter and Image Options
- `--filters`: List of filters for image search (e.g., `--filters g r i`).
  - Available filters: `u`, `g`, `r`, `i`, `z`, `y`.
  - Default: `["r"]`.
- `--image-type`: Type of image to process for photometry.
  - Options: `"visit_image"`, `"goodSeeingDiff_differenceExp"`.
  - Default: `"visit_image"`.
- `--image-search-method`: Method to search for images.
  - Options: `"point"`, `"polygon"`.
  - Default: `"point"`.
- `--time-interval`: Time interval in days for grouping ephemeris points when using the `"polygon"` image search method.
  - Default: `5.0`.
- `--widening`: Widening factor in arcseconds for the polygon search area around the ephemeris track.
  - Default: `1.0`.

### Output Options
- `--output-folder`: Directory to save all output files (ephemeris data, diagnostic plots, FITS, JSON/CSV results).
  - Default: `./output`.
- `--save-ephem-data`: Save ephemeris query results to a file in the output folder.
- `--save-diag-plots`: Save diagnostic images (PNG) marked with sources and the error ellipse.
- `--save-fits`: Save the FITS image cutouts.
- `--min-cutout-size`: Minimum size of square image cutouts in pixels.
  - Default: `800`.
- `--save-json`: Save final photometry results as JSON.
- `--save-csv`: Save final photometry results as CSV.
- `--all-ellipse-sources`: If `True` and saving to CSV, create separate rows for each source detected within the error ellipse. If `False`, only include the best source per result.

### Additional Options
- `--location`: Observer location code (e.g., "X05" for Rubin Observatory).
  - Default: `"X05"`.
- `--threshold`: Signal-to-Noise Ratio (SNR) threshold for source detection in photometry.
  - Default: `3`.
- `--override-error`: If greater than 0, this value in arcseconds will be used as the radius for a circular error ellipse, overriding the ephemeris's reported uncertainty.
  - Default: `0` (no override).
- `--display`: If `True`, attempts to display results (images and error ellipses) in Firefly for visualization.
  - Default: `False`.
- `-h, --help`: Show help message and exit.

## Working with Input Files

### Using ECSV Ephemeris Data
You can use pre-computed ephemeris data in ECSV format to bypass live ephemeris queries:

```bash
python odc.py --service-selection all \
              --ephem-ecsv /path/to/ephemeris.ecsv \
              --filters g r i \
              --save-diag-plots \
              --save-fits
```

The ECSV file should contain the necessary columns with their respective units and descriptions:
- `datetime` (float64, unit: `d`, description: Time for the ephemeris data points.)
- `RA_deg` (float64, unit: `deg`, description: Right Ascension in degrees)
- `DEC_deg` (float64, unit: `deg`, description: Declination in degrees)
- `RA_rate_arcsec_per_h` (float64, unit: `arcsec / h`, description: Rate of change in Right Ascension)
- `DEC_rate_arcsec_per_h` (float64, unit: `arcsec / h`, description: Rate of change in Declination)
- `AZ_deg` (float64, unit: `deg`, description: Azimuth in degrees)
- `EL_deg` (float64, unit: `deg`, description: Elevation in degrees)
- `r_au` (float64, unit: `AU`, description: Heliocentric distance in astronomical units)
- `delta_au` (float64, unit: `AU`, description: Geocentric distance in astronomical units)
- `V_mag` (float64, optional, description: Visual magnitude)
- `alpha_deg` (float64, unit: `deg`, description: Phase angle in degrees)
- `RSS_3sigma_arcsec` (float64, unit: `arcsec`, description: 3-sigma Root Sum Square uncertainty in arcseconds)
- `SMAA_3sigma_arcsec` (float64, unit: `arcsec`, description: Semi-major axis of error ellipse)
- `SMIA_3sigma_arcsec` (float64, unit: `arcsec`, description: Semi-minor axis of error ellipse)
- `Theta_3sigma_deg` (float64, unit: `deg`, description: Position angle of error ellipse)

Example ECSV format:
```
# %ECSV 1.0
# ---
# datatype:
# - {name: datetime, unit: d, datatype: float64, description: Time for the ephemeris data points.}
# - {name: RA_deg, unit: deg, datatype: float64, description: Right Ascension in degrees}
# - {name: DEC_deg, unit: deg, datatype: float64, description: Declination in degrees}
# - {name: RA_rate_arcsec_per_h, unit: arcsec / h, datatype: float64, description: Rate of change in Right Ascension}
# - {name: DEC_rate_arcsec_per_h, unit: arcsec / h, datatype: float64, description: Rate of change in Declination}
# - {name: AZ_deg, unit: deg, datatype: float64, description: Azimuth in degrees}
# - {name: EL_deg, unit: deg, datatype: float64, description: Elevation in degrees}
# - {name: r_au, unit: AU, datatype: float64, description: Heliocentric distance in astronomical units}
# - {name: delta_au, unit: AU, datatype: float64, description: Geocentric distance in astronomical units}
# - {name: V_mag, datatype: float64, description: Visual magnitude}
# - {name: alpha_deg, unit: deg, datatype: float64, description: Phase angle in degrees}
# - {name: RSS_3sigma_arcsec, unit: arcsec, datatype: float64, description: 3-sigma uncertainty in arcseconds}
# - {name: SMAA_3sigma_arcsec, unit: arcsec, datatype: float64, description: Semi-major axis of error ellipse}
# - {name: SMIA_3sigma_arcsec, unit: arcsec, datatype: float64, description: Semi-minor axis of error ellipse}
# - {name: Theta_3sigma_deg, unit: deg, datatype: float64, description: Position angle of error ellipse}
# schema: astropy-2.0
datetime RA_deg DEC_deg RA_rate_arcsec_per_h DEC_rate_arcsec_per_h AZ_deg EL_deg r_au delta_au V_mag alpha_deg RSS_3sigma_arcsec SMAA_3sigma_arcsec SMIA_3sigma_arcsec Theta_3sigma_deg
2460610.054 70.7969 -30.0513 99.88192 17.15154 242.49141 5.849219 1.424217235644 2.38654504100278 18.18 6.2864 6.427 40.409 15.121 13.897
2460610.1762 70.9 -29.8 99.88192 17.15154 242.49141 5.849219 1.424217235644 2.38654504100278 18.18 6.2864 6.427 40.409 15.121 13.897
```

### Using CSV for Multiple Targets
For batch processing multiple targets through the ephemeris service, you can use a CSV file. This is currently only supported for the ephemeris query step.

```bash
python odc.py --service-selection all \
              --csv /path/to/targets.csv \
              --ephemeris-service Horizons \
              --filters g r i \
              --save-ephem-data
```

The CSV file should contain the following columns:
- `target_id`
- `target_type`
- `start_time` (format: `"YYYY-MM-DD HH:MM:SS"`)
- `end_time` (optional, if `day_range` is used instead)
- `day_range` (optional, if `end_time` is used instead, integer number of days)

Example CSV format:
```csv
target_id,target_type,start_time,day_range
65803,smallbody,2022-09-25 00:00:00,6
2014MU69,smallbody,2022-09-25 00:00:00,3
```

## API Usage
The Object Detection Controller (`odc.ObjectDetectionController`) can be used programmatically to integrate into other Python scripts or applications. The `api_connection` method provides a unified interface for all pipeline steps.

```python
from odc import ObjectDetectionController

controller = ObjectDetectionController()

# Example for a complete end-to-end run via API
input_data = {
    "ephemeris": {
        "target": "2024 TN57",
        "target_type": "smallbody",
        "ephemeris_service": "Horizons",
        "start": "2024-12-01 00:00:00",
        "end": "2024-12-11 00:00:00",
        "step": "10h",
        "observer_location": "X05",
        "save_ephem_data": True,
        "output_folder": "./output"
    },
    "image": {
        "filters": ["g", "r", "i"],
        # ephemeris_data will be populated by the controller from the "ephemeris" step if run first
        # For a standalone image query, you might pass a pre-loaded QueryResult object here:
        # "ephemeris_data": pre_loaded_ephem_query_result_object,
        "image_search_method": "point",
        "widening": 1.0, # Only relevant if image_search_method is "polygon"
        "time_interval": 5.0 # Only relevant if image_search_method is "polygon"
    },
    "photometry": {
        "image_type": "visit_image", # Corrected from "calexp"
        "threshold": 5,
        "save_diag_plots": True, # Corrected from "save_cutouts"
        "save_fits": True, # New option for FITS cutouts
        "display": False,
        "save_json": True,
        "save_csv": True,
        "save_error_sources": True, # New option for all ellipse sources in CSV
        "output_folder": "./api_output"
    }
}

results = controller.api_connection(input_data)

# 'results' will contain a dictionary with keys 'ephemeris', 'image', 'photometry'
# holding the respective output data, or an 'error' key if a step failed.
print(results)
```

## Common Use Cases

### 1. Basic Ephemeris Query (CLI)
Query and display ephemeris data for a single target for one day.
```bash
python odc.py --service-selection ephemeris \
              --target 2024 TN57 \
              --target-type smallbody \
              --start-time "2024-12-01 00:00:00" \
              --day-range 10
              --step 10h
```

### 2. Multi-filter Image Search (CLI)
Find images in multiple filters for a target, assuming ephemeris data is either pre-computed or will be queried based on target/time parameters.
```bash
python odc.py --service-selection image \
              --filters g r i \
              --target 2024 TN57 \
              --target-type smallbody \
              --start-time "2024-12-1 00:00:00" \
              --day-range 10 \
              --step 10h \
              --image-search-method polygon \
              --time-interval 5.0 \
              --widening 2.0
```

### 3. Full Pipeline with Photometry (CLI)
Execute the entire pipeline: ephemeris query, image search, and photometry, saving diagnostic plots and JSON/CSV results.
```bash
python odc.py --service-selection all \
              --target 2024 TN57 \
              --target-type smallbody \
              --start-time "2024-12-01 00:00:00" \
              --day-range 10 \
              --step 12h \
              --filters g r i \
              --save-diag-plots \
              --save-fits \
              --save-json \
              --save-csv \
              --all-ellipse-sources \
              --output-folder ./output
```

## Panel application
The package provides a Panel-based web interface for interactive use:
- Open `control_panel.py` in your RSP notebook environment.
- Start the Panel app by clicking the "Preview with Lumen" icon (usually in the top left corner of the code view, or by running `panel serve control_panel.py` in a terminal).
The application is divided into five main tabs, each handling a specific part of the detection and analysis process.

### Main Features
- Ephemeris generation and management.
- Image search and filtering.
- Photometric analysis.
- Complete pipeline execution (combining all steps).
- Real-time logging output (with some caveats, see "Tips").

### Tab Descriptions

#### 1. Ephemeris Query Tab
This tab handles the generation and management of ephemeris data.

##### Key Parameters:
- **Ephemeris Source**: Choose between "Use Generated Data" (live query) or "Upload ECSV" (use a local file).
- **Service**: Select between "Horizons" and "Miriade" services for live queries.
- **Target Name**: Enter the name of the target (e.g., "C/2020 F3 (NEOWISE)").
- **Target Type**: Choose between `smallbody`, `asteroid_name`, `comet_name`, or `designation`.
- **Time Parameters**:
  - `Start Time`: Set the beginning of the observation period.
  - `Time Specification`: Choose between defining an `End Time` or a `Day Range` from the start.
  - `Step Value` and `Unit`: Define the time resolution of the ephemeris (e.g., `1` `h`).
- **Save Ephemeris Data**: Option to save the queried ephemeris data to the `Output Folder`.

##### Output:
Displays a table with detailed ephemeris data including:
- DateTime (ISO format)
- RA/Dec coordinates (degrees)
- RA/Dec motion rates (arcsec/h)
- Azimuth/Elevation (degrees)
- Heliocentric distance (`r` in AU)
- Geocentric distance (`delta` in AU)
- Visual magnitude (`V mag`)
- Phase angle (`Alpha` in degrees)
- Uncertainty information (RSS 3σ, SMAA 3σ, SMIA 3σ, Theta 3σ in arcsec/deg)

#### 2. Image Search Tab
Handles image search operations based on ephemeris data provided by the "Ephemeris Query" tab or an ECSV file.

##### Key Parameters:
- **Image Search Method**: Choose `"Point"` (searches for images overlapping individual ephemeris points) or `"Polygon"` (creates a polygon from the ephemeris track and searches for images overlapping this area).
- **Filters (Bands)**: Select desired photometric bands (e.g., `u`, `g`, `r`, `i`, `z`, `y`). Multiple selections are allowed.
- **Polygon Search Options** (visible only when "Polygon" method is selected):
  - `Widening (arcsec)`: Expansion factor around the polygon track.
  - `Time Interval (days)`: Time window for grouping ephemeris points into polygons.

##### Output:
Displays a table with metadata for found images, including:
- Visit ID
- Detector ID
- Filter band
- Observation Start Time (`Obs Start`) and End Time (`Obs End`) in ISO format.

#### 3. Photometry Analysis Tab
Configures and executes photometric analysis on the image metadata retrieved from the "Image Search" tab.

##### Key Parameters:
- **Image Type**: Choose between `"visit_image"` (raw visit data) and `"goodSeeingDiff_differenceExp"` (difference images).
- **Detection Threshold (SNR)**: Set the Signal-to-Noise Ratio (SNR) sensitivity for source detection.
- **Cutout Size (pixels)**: Define the square analysis window in pixels (e.g., `800` for 800x800). Set to `0` to use the full image.
- **Override Error Ellipse (arcsec)**: If greater than 0, provides a user-defined circular radius to override the ephemeris's reported uncertainty for forced photometry.
- **Output Options**:
  - `Save Diagnostic Plots (PNG)`: Save plots showing sources and error ellipses.
  - `Save FITS Cutouts`: Save the FITS image cutouts.
  - `Display Results in Firefly`: Attempt to display images and results in a Firefly viewer.
  - `Save Results to JSON`: Save the final photometry results in JSON format.
  - `Save Results to CSV`: Save the final photometry results in CSV format.
  - `Include All Sources within Error Ellipse in CSV`: If `Save Results to CSV` is checked, this creates separate rows for each source detected within the error ellipse.

##### Results Display:
The table summarizes photometry measurements for the target object in each processed image, including:
- Visit ID, Detector ID, Band
- Flux measurements (nJy) and their errors
- Magnitude measurements (AB) and their errors
- Signal-to-Noise Ratio (SNR)
- Count of nearby sources detected within the error ellipse.

#### 4. Complete Run Tab
Executes the full object detection pipeline end-to-end, from ephemeris query, through image search, to final photometry.

##### Features:
- Consolidates all parameters from the previous tabs into a single, comprehensive interface.
- Provides collapsible parameter cards for better organization and navigation.
- Offers single-click execution of the entire workflow, streamlining the process.

##### Usage:
1. Configure all parameters across the Ephemeris, Image Search, and Photometry sections.
2. Click the "Run All Services" button to start the complete pipeline execution.
3. The final photometry results will be displayed in the table at the bottom of the tab.

### Using the Application

#### Basic Workflow:
1. **Generate Ephemeris**: Navigate to the "Ephemeris Query" tab. Select your source, enter target information, and define time parameters. Click "Run Ephemeris Query".
2. **Search for Images**: Go to the "Image Search" tab. Select desired filters and the image search method. Click "Run Image Search". This step will use the ephemeris data obtained in the previous step.
3. **Perform Photometry**: Switch to the "Photometry Analysis" tab. Configure detection and output parameters. Click "Run Photometry Analysis". This step will use the image metadata from the previous step.
4. **Complete Pipeline**: For an integrated workflow, use the "Complete Run" tab. Configure all parameters in one place and execute the entire pipeline with a single button press.

#### Additional Features:
- **Terminal Output**: View real-time (note: actual updates might be slightly delayed in the browser due to Panel's rendering mechanism) logging and system messages at the bottom of the application window.
- **Test Functions**: Use the "Test Logging" and "Test Terminal" buttons to verify the logging and terminal display functionality.
- **Clear Terminal**: Use the "Clear Terminal" button to reset the terminal display.
- **Table Views**: Interactive data tables (powered by Tabulator) with sorting and filtering capabilities for easy data exploration.

## Tips 
1. Always specify a target and time range for ephemeris queries.
2. Consider using `--save-ephem-data` for long ephemeris queries to avoid re-running them.
3. Adjust the `--threshold` (SNR) for photometry based on your desired detection sensitivity.
4. Use `--display` (in CLI) or `Display Results in Firefly` (in Panel) for visual inspection of results and cutouts.
5. When processing multiple targets or large time ranges, monitor memory usage.
6. Consider using pre-computed ECSV ephemeris data (`--ephem-ecsv` or "Upload ECSV" in Panel) to save processing time, especially for repetitive runs.
7. Use the Panel application for a more user-friendly, interactive interface.
8. Use the `--help` option for detailed command-line options and their descriptions.
9. If you run the different services within the Panel application, the terminal output may experience a slight delay in refreshing. Be patient.

## Troubleshooting
- If an ephemeris query fails, verify the target ID/name, target type, and ensure the start/end times or day range are valid.
- Check time format if getting time-related errors (ensure "YYYY-MM-DD HH:MM:SS").
- Ensure the specified output directory exists and is writable if saving files.
- For queries covering large time ranges or a high number of steps, consider using larger time steps (`--step`) to reduce the amount of ephemeris data.
- If the ephemeris data generated or loaded is exceptionally large, the subsequent image search step might be slow or fail due to memory constraints.
- If the code is running in the Panel application, the terminal output will be delayed, so be patient for messages to appear.
- Use the "Test Terminal" button in the Panel app to confirm the terminal is responsive. If it doesn't work, it might indicate the main application process is busy.
