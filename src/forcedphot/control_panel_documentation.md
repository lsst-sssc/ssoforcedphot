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

## Troubleshooting
- If ephemeris query fails, verify target ID and type
- Check time format if getting time-related errors
- Ensure output directory exists and is writable
- For large queries, consider using larger time steps
- If the ephemeris data is too large, the image search can be slow or fail
- If the code is running in the Panel application, the terminal output will be delayed, be patient.
- Check the terminal output with the Test Terminal button. If the code is running in the background, the button will not work.