# Faint Solar System Object Detection Service Guide
## Object Detection Controller 

### Overview
The Object Detection Controller (ODC) is a Python tool designed for solar system object detection and photometry. 
It integrates multiple services including ephemeris queries, image searching, and photometry processing.

### Features
- Ephemeris data retrieval from multiple services (JPL Horizons, Miriade)
- Image search and processing (ObsCore)
- Forced photometry on target coordinates
- Photomerty on nearby sources (within error ellipse)
- Multiple filters selection (u, g, r, i, z, y)
- Flexible input options (single query, batch processing via CSV)
- Multiple output options (The end results are JSON files)

### Installation Requirements
- Clone the repository
- Right now, it is only compatible with the RSP 

### Basic Usage

#### Command Line Interface
The basic structure of an ODC command is:
```bash
python odc.py [service options] [target options] [time options] [filter options] [output options]
```

#### Example Command
```bash
python odc.py --service-selection all \
              --ephemeris-service Horizons \
              --image-type goodSeeingDiff_differenceExp \
              --target 65803 \
              --target-type smallbody \
              --start-time "2022-09-25 00:00" \
              --day-range 6 \
              --step 1h \
              --save-data \
              --filters g r i \
              --output-dir ./output \
              --min-cutout-size 800
```

### Command Line Options

#### Service Selection
- `--service-selection`: Choose which services to use
  - Options: "all", "ephemeris", "image", "photometry"
  - Default: "ephemeris"

#### Ephemeris Options
- `--ephemeris-service`: Select the ephemeris service
  - Options: "Horizons", "Miriade"
  - Default: "Horizons"
- `--target`: Target object identifier
- `--target-type`: Type of target object (e.g., "smallbody")

#### Input Options
- `--csv`: CSV file for batch processing
- `--ecsv`: ECSV file for using pre-computed ephemeris data

#### Time Range Options
- `--start-time`: Start time (format: "YYYY-MM-DD HH:MM:SS")
- `--end-time`: End time (format: "YYYY-MM-DD HH:MM:SS")
- `--day-range`: Number of days to search forward from start time
- `--step`: Time step for ephemeris query (default: "1h")

#### Filter and Image Options
- `--filters`: List of filters to use (e.g., "g r i")
  - Available filters: u, g, r, i, z, y
  - Default: ["r"]
- `--image-type`: Type of image to process
  - Options: "calexp", "goodSeeingDiff_differenceExp"
  - Default: "calexp"

#### Output Options
- `--output-dir`: Directory for output files (default: "./output")
- `--save-data`: Save query results
- `--save-cutouts`: Save image cutouts
- `--min-cutout-size`: Minimum size of cutouts (default: 800)
- `--save_json`: Save results as JSON
- `--display`: Display images and error ellipses in Firefly

#### Additional Options
- `--location`: Observer location code (default: "X05" for Rubin Observatory)
- `--threshold`: SNR threshold for forced photometry (default: 3)
- `-h, --help`: Show help message and exit

### Working with Input Files

#### Using ECSV Ephemeris Data
You can use pre-computed ephemeris data in ECSV format:

```bash
python odc.py --service-selection all \
              --ephem-ecsv /path/to/ephemeris.ecsv \
              --filters g r i \
              --save-cutouts
```

The ECSV file should contain the necessary columns:
- datetime
- RA_deg (degrees)
- DEC_deg (degrees)
- RA_rate_arcsec_per_h (arcsec / h)
- DEC_rate_arcsec_per_h (arcsec / h)
- AZ_deg (degrees)
- EL_deg (degrees)
- r_au (AU)
- delta_au (AU)
- V_mag (optional)
- alpha_deg (degrees)
- RSS_3sigma_arcsec (arcsec)
- SMAA_3sigma_arcsec (arcsec)
- SMIA_3sigma_arcsec (arcsec)
- Theta_3sigma_deg (degrees)

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

#### Using CSV for Multiple Targets
For batch processing multiple targets, you can use a CSV file. 
This is currently only usable for the ephemeris service.

```bash
python odc.py --service-selection all \
              --csv /path/to/targets.csv \
              --ephemeris-service Horizons \
              --filters g r i \
              --save-data
```

The CSV file should contain the following columns:
- target_id
- target_type
- start_time
- end_time (optional if using day_range)
- day_range (optional if using end_time)

Example CSV format:
```csv
target_id,target_type,start_time,day_range
65803,smallbody,2022-09-25 00:00,6
2014MU69,smallbody,2022-09-25 00:00,3
```

#### API Usage for Single Target
You can also process a single target programmatically:

```python
from odc import ObjectDetectionController

controller = ObjectDetectionController()

# Using CSV file
input_data = {
    "ephemeris": {
        "service": "Horizons",
        "csv_file": "/path/to/targets.csv",
        "observer_location": "X05",
        "save_data": True
    },
    "image": {
        "filters": ["g", "r", "i"]
    },
    "photometry": {
        "image_type": "calexp",
        "threshold": 5
    }
}

results = controller.api_connection(input_data)

```

### Common Use Cases

#### 1. Basic Ephemeris Query
```bash
python odc.py --service-selection ephemeris \
              --target 65803 \
              --target-type smallbody \
              --start-time "2022-09-25 00:00" \
              --day-range 1
```

#### 2. Multi-filter Image Search
```bash
python odc.py --service-selection image \
              --filters g r i \
              --target 65803 \
              --start-time "2022-09-25 00:00" \
              --day-range 1
```

#### 3. Full Pipeline with Photometry
```bash
python odc.py --service-selection all \
              --target 65803 \
              --target-type smallbody \
              --start-time "2022-09-25 00:00" \
              --day-range 1 \
              --filters g r i \
              --save-cutouts \
              --save_json
```

### API Usage
The ODC can also be used programmatically through its API interface:

```python
from odc import ObjectDetectionController

controller = ObjectDetectionController()

input_data = {
    "ephemeris": {
        "target": "65803",
        "target_type": "smallbody",
        "service": "Horizons",
        "start": "2022-09-25 00:00",
        "end": "2022-09-26 00:00",
        "step": "1h"
    },
    "image": {
        "filters": ["g", "r", "i"]
    },
    "photometry": {
        "image_type": "calexp",
        "threshold": 5,
        "save_cutouts": True
    }
}

results = controller.api_connection(input_data)
```

## Panel application
The ODC can also be controlled via a Panel application. 
The application is divided into four main tabs, each handling a specific part of the detection and analysis process.

### Main Features
- Ephemeris generation and management
- Image search and filtering
- Photometric analysis
- Complete pipeline execution

### Tab Descriptions

#### 1. Ephemeris Tab
This tab handles the generation and management of ephemeris data.

##### Key Parameters:
- **Ephemeris Source**: Choose between using existing data or uploading an ECSV file
- **Service**: Select between Horizons and Miriade services
- **Target Name**: Enter the name of the target
- **Target Type**: Choose between 'smallbody' or 'comet_name'
- **Time Parameters**:
  - Start Time: Set the beginning of the observation period
  - Time Specification: Choose between End Time or Day Range
  - Step Value and Unit: Define the time resolution of the ephemeris

##### Output:
Displays a table with detailed ephemeris data including:
- DateTime
- RA/Dec coordinates (degrees)
- Motion rates
- Azimuth/Elevation
- Distance measurements
- Magnitude and uncertainty information

#### 2. Image Tab
Handles image search operations based on ephemeris data.

##### Key Features:
- Filter selection (u, g, r, i, z, y bands)
- Integration with ephemeris data
- Tabulated results showing:
  - Visit ID
  - Detector ID
  - Filter band
  - Time range (min/max)

#### 3. Photometry Tab
Configures and executes photometric analysis.

##### Key Parameters:
- **Image Type**: Choose between 'calexp' and 'goodSeeingDiff_differenceExp'
- **Detection Threshold**: Set sensitivity for detection
- **Cutout Size**: Define the analysis window in pixels
- **Output Options**:
  - Save Cutouts
  - Display Results
  - Save to JSON

##### Results Display:
- Visit and Detector IDs
- Band information
- Flux measurements (nJy)
- Magnitude measurements (AB)
- Signal-to-Noise Ratio (SNR)
- Nearby source counts

#### 4. Complete Run Tab
Executes the full detection pipeline from ephemeris to photometry.

##### Features:
- Combines all parameters from previous tabs
- Single-click execution of the entire workflow
- Collapsible parameter cards for better organization

### Using the Application

#### Basic Workflow:
1. **Generate Ephemeris**:
   - Select the appropriate service
   - Enter target information
   - Define time parameters
   - Run the ephemeris query

2. **Search for Images**:
   - Select desired filters
   - Use generated ephemeris or uploaded ECSV
   - Execute image search

3. **Perform Photometry**:
   - Configure detection parameters
   - Set output preferences
   - Run photometric analysis

4. **Complete Pipeline**:
   - Use the Complete Run tab for end-to-end processing
   - Configure all parameters in one place
   - Execute with single button press

#### Additional Features:
- **Terminal Output**: View real-time (not really) logging and system messages
- **Test Functions**: Verify system functionality
- **Clear Terminal**: Reset the terminal display
- **Table Views**: Interactive data tables with sorting and filtering capabilities


## Tips 
1. Always specify a target and time range
2. Consider using `--save-data` for long queries
3. Adjust `--threshold` based on your detection requirements
4. Use `--display` for visual inspection
5. When processing multiple targets, monitor memory usage for large datasets
6. Consider using ECSV format for pre-computed ephemeris data to save processing time
7. Use the Panel application for a more user-friendly interface
8. Use the `--help` option for detailed command-line options
9. If you run the different services in the panel application, the terminal output will be delayed. 

## Troubleshooting
- If ephemeris query fails, verify target ID and type
- Check time format if getting time-related errors
- Ensure output directory exists and is writable
- For large queries, consider using larger time steps
- If the ephemeris data is too large, the image search can be slow or fail
- If the code is running in the Panel application, the terminal output will be delayed, be patient.
- Check the terminal output with the Test Terminal button. If the code is running in the background, the button will not work.
