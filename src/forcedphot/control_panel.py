import datetime
import io
import logging
import sys

import astropy.units as u
import pandas as pd
import panel as pn
from astropy.table import Table
from astropy.time import Time
from odc import ObjectDetectionController

from forcedphot.ephemeris.data_model import QueryResult

# Set up logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Panel extensions and template
pn.extension('tabulator', 'terminal', design='material')
pn.config.theme = 'dark'

template = pn.template.MaterialTemplate(
    title='Faint Solar System Object Detection Service',
    logo='rubin_logo.svg'
)

# Initialize terminal and redirect
terminal = pn.widgets.Terminal(
    "Application Output:\n",
    options={"cursorBlink": True, "scrollback": 1000, "encoding": "utf-8", "fontSize": 11},
    height=300,
    sizing_mode='stretch_width'
)

# Stream handler for the terminal widget
class TerminalHandler(logging.Handler):
    """Custom logging handler that writes formatted log messages to a terminal widget with color coding.

    Args:
        terminal_widget (pn.widgets.Terminal): The terminal widget to display logs in.
    """
    def __init__(self, terminal_widget):
        super().__init__()
        self.terminal_widget = terminal_widget

    def emit(self, record):
        """Terminal widget output format"""
        try:
            msg = self.format(record)
            # Add color based on log level
            if record.levelno >= logging.ERROR:
                msg = f"\033[91m{msg}\033[0m"  # Red for errors
            elif record.levelno >= logging.WARNING:
                msg = f"\033[93m{msg}\033[0m"  # Yellow for warnings
            elif record.levelno >= logging.INFO:
                msg = f"\033[92m{msg}\033[0m"  # Green for info

            self.terminal_widget.write(msg + "\n")
        except Exception:
            self.handleError(record)

terminal_handler = TerminalHandler(terminal)
terminal_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))

root_logger.addHandler(terminal_handler)

# Custom stdout/stderr redirector
class StreamToLogger:
    """Redirects standard output/error streams to a logging handler.

    Args:
        terminal_widget (pn.widgets.Terminal): Terminal widget for log display.
        is_error (bool): Whether to treat the stream as stderr (default: False).
    """
    def __init__(self, terminal_widget, is_error=False):
        self.terminal = terminal_widget
        self.is_error = is_error
        self.logger = logging.getLogger('stdout' if not is_error else 'stderr')
        self.logger.setLevel(logging.INFO)

    def write(self, buf):
        """Write function for terminal widget"""
        if buf.rstrip():
            if self.is_error:
                self.logger.error(buf.rstrip())
            else:
                self.logger.info(buf.rstrip())

    def flush(self):
        """Flush function for terminal widget"""
        pass


sys.stdout = StreamToLogger(terminal)
sys.stderr = StreamToLogger(terminal, is_error=True)

def serialize_query_result(result):
    """Convert QueryResult and EphemerisData objects to a JSON-serializable dictionary.

    Args:
        result (QueryResult): The query result object to serialize.

    Returns:
        dict: Serialized data structure compatible with JSON.
    """
    def serialize_time(time_obj):
        return time_obj.iso if isinstance(time_obj, Time) else str(time_obj)

    def serialize_ephemeris(ephem):
        """Convert EphemerisData to JSON-serializable format"""
        return {
            'datetime': [serialize_time(t) for t in ephem.datetime],
            'RA_deg': ephem.RA_deg.tolist(),
            'DEC_deg': ephem.DEC_deg.tolist(),
            'RA_rate_arcsec_per_h': ephem.RA_rate_arcsec_per_h.tolist(),
            'DEC_rate_arcsec_per_h': ephem.DEC_rate_arcsec_per_h.tolist(),
            'AZ_deg': ephem.AZ_deg.tolist(),
            'EL_deg': ephem.EL_deg.tolist(),
            'r_au': ephem.r_au.tolist(),
            'delta_au': ephem.delta_au.tolist(),
            'V_mag': ephem.V_mag.tolist(),
            'alpha_deg': ephem.alpha_deg.tolist(),
            'RSS_3sigma_arcsec': ephem.RSS_3sigma_arcsec.tolist(),
            'SMAA_3sigma_arcsec': ephem.SMAA_3sigma_arcsec.tolist(),
            'SMIA_3sigma_arcsec': ephem.SMIA_3sigma_arcsec.tolist(),
            'Theta_3sigma_deg': ephem.Theta_3sigma_deg.tolist(),
        }

    return {
        'target': result.target,
        'start': serialize_time(result.start),
        'end': serialize_time(result.end),
        'ephemeris': serialize_ephemeris(result.ephemeris)
    }

class EphemerisTab:
    """GUI tab for managing ephemeris queries and displaying results.

    Args:
        controller (ObjectDetectionController): The main application controller.
    """
    def __init__(self, controller):
        self.controller = controller

        # Widgets
        self.ephemeris_source = pn.widgets.RadioButtonGroup(
            name='Ephemeris Source',
            options=['Use Existing Data', 'Upload ECSV'],
            value='Use Existing Data'
        )
        self.file_upload = pn.widgets.input.FileInput(accept='.ecsv', multiple=False)
        self.service = pn.widgets.Select(name="Service", options=["Horizons", "Miriade"], value="Horizons")
        self.target_name = pn.widgets.TextInput(name="Target Name")
        self.target_type = pn.widgets.Select(
            name="Target Type",
            options=["smallbody", "comet_name"],
            value="smallbody"
        )
        self.start_time = pn.widgets.DatetimePicker(
            name="Start Time",
            value=datetime.datetime.now(),
            enable_time=True
        )
        self.time_spec = pn.widgets.RadioButtonGroup(
            name='Time Specification',
            options=['End Time', 'Day Range'],
            value='End Time'
        )
        self.end_time = pn.widgets.DatetimePicker(
            name="End Time",
            value=datetime.datetime.now() + datetime.timedelta(days=1),
            enable_time=True
        )
        self.day_range = pn.widgets.IntInput(name="Day Range", value=1, start=1, width=120)
        self.step_value = pn.widgets.FloatInput(
            name="Step Value",
            value=1,
            start=1,
            step=1,
            width=120
        )
        self.step_unit = pn.widgets.Select(
            name="Step Unit",
            options=['d', 'h', 'm'],
            value='h',
            width=50
        )
        self.save_data = pn.widgets.Checkbox(name="Save Ephemeris")
        self.run_button = pn.widgets.Button(
            name=pn.bind(lambda s: "Update" if s == 'Upload ECSV' else "Run Query",
                         self.ephemeris_source.param.value
                        ),
            button_type='primary'
        )
        self.result_pane = pn.pane.JSON(object={}, name="Results", depth=3, height=300)

        # Set up visibility bindings
        self.end_time.visible = pn.bind(lambda ts: ts == 'End Time', self.time_spec.param.value)
        self.day_range.visible = pn.bind(lambda ts: ts == 'Day Range', self.time_spec.param.value)
        self.start_time.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)
        self.end_time.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)
        self.day_range.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)
        self.step_value.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)
        self.step_unit.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)
        self.save_data.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)

        # Link button click
        self.run_button.on_click(self.run_query)

        # Visualization components with all EphemerisData fields
        self.table_view = pn.widgets.Tabulator(
            sizing_mode='stretch_width',
            height=450,
            page_size=10,
            configuration={
                'columns': [
                    {'title': 'Time', 'field': 'datetime'},
                    {'title': 'RA (deg)', 'field': 'RA_deg'},
                    {'title': 'Dec (deg)', 'field': 'DEC_deg'},
                    {'title': 'RA Rate ("/h)', 'field': 'RA_rate_arcsec_per_h'},
                    {'title': 'Dec Rate ("/h)', 'field': 'DEC_rate_arcsec_per_h'},
                    {'title': 'AZ (deg)', 'field': 'AZ_deg'},
                    {'title': 'EL (deg)', 'field': 'EL_deg'},
                    {'title': 'r (au)', 'field': 'r_au'},
                    {'title': 'Delta (au)', 'field': 'delta_au'},
                    {'title': 'V mag', 'field': 'V_mag'},
                    {'title': 'Alpha (deg)', 'field': 'alpha_deg'},
                    {'title': 'RSS 3σ (")', 'field': 'RSS_3sigma_arcsec'},
                    {'title': 'SMAA 3σ (")', 'field': 'SMAA_3sigma_arcsec'},
                    {'title': 'SMIA 3σ (")', 'field': 'SMIA_3sigma_arcsec'},
                    {'title': 'Theta 3σ (deg)', 'field': 'Theta_3sigma_deg'},
                ]
            }
        )

        def conditional_upload(source):
            """Condition check for ephemeris data source"""
            return self.file_upload if source == 'Upload ECSV' else pn.pane.Str("Using existing data.")

        self.layout = pn.Row(
            pn.Column(
                "### Ephemeris Query Parameters",
                self.ephemeris_source,
                pn.bind(conditional_upload, self.ephemeris_source.param.value),
                self.service,
                self.target_name,
                self.target_type,
                self.start_time,
                self.time_spec,
                pn.Row(self.end_time, self.day_range),
                pn.Row(
                    pn.Column("### Step", self.step_value),
                    pn.Column("### Unit", self.step_unit),
                    sizing_mode='stretch_width',
                ),
                self.save_data,
                self.run_button,
                sizing_mode='stretch_width'
            ), self.table_view
        )

    def get_step_string(self):
        """Combine step value and unit into the required format"""
        return f"{self.step_value.value}{self.step_unit.value}"

    def run_query(self, event):
        """Handle query execution and display results"""
        if self.ephemeris_source.value == 'Upload ECSV':
            root_logger.info("Processing uploaded ECSV file.")
            if not self.file_upload.value:
                self.result_pane.object = {"error": "Please upload an ECSV file."}
                self.table_view.value = pd.DataFrame()
                return
            try:
                # Read the uploaded ECSV
                content = io.BytesIO(self.file_upload.value)
                table = Table.read(content, format='ascii.ecsv')
                df = table.to_pandas()

                # Update the table view
                self.table_view.value = df

                input_data = {
                    "ephemeris": {
                        "service": self.service.value,
                        "target": self.target_name.value,
                        "target_type": self.target_type.value,
                        "ecsv_file": self.file_upload.filename
                    }
                }
                self.controller.ephemeris_results = self.controller.api_connection(input_data)
                root_logger.info("ECSV file processed successfully.")
            except Exception as e:
                self.result_pane.object = {"error": str(e)}
                self.table_view.value = pd.DataFrame()
                root_logger.error(f"Error processing ECSV: {str(e)}")

        else:
            root_logger.info("Running Ephemeris service.")
            input_data = {
                "ephemeris": {
                    "service": self.service.value,
                    "target": self.target_name.value,
                    "target_type": self.target_type.value,
                    "start": self.start_time.value.strftime('%Y-%m-%d %H:%M:%S'),
                    "save_data": self.save_data.value,
                    "observer_location": "X05",
                    "step": self.get_step_string()
                }
            }

            # Handle time specification
            if self.time_spec.value == 'End Time':
                input_data["ephemeris"]["end"] = self.end_time.value.strftime('%Y-%m-%d %H:%M:%S')
            else:
                try:
                    start_time = Time(self.start_time.value)
                    end_time = start_time + self.day_range.value * u.day
                    input_data["ephemeris"]["end"] = end_time.iso
                except Exception as e:
                    self.result_pane.object = {"error": f"Invalid start time: {str(e)}"}
                    self.table_view.value = pd.DataFrame()
                    return
            try:
                self.controller.ephemeris_results = self.controller.api_connection(input_data)

                if 'ephemeris' in self.controller.ephemeris_results and isinstance(
                    self.controller.ephemeris_results['ephemeris'],
                    QueryResult
                ):
                    serialized_query = serialize_query_result(self.controller.ephemeris_results['ephemeris'])
                    ephemeris_data = serialized_query['ephemeris']
                    df = pd.DataFrame(ephemeris_data)
                    self.table_view.value = df
                else:
                    self.result_pane.object = self.controller.ephemeris_results
                    self.table_view.value = pd.DataFrame()

            except Exception as e:
                self.result_pane.object = {"error": str(e)}
                self.table_view.value = pd.DataFrame()


class ImageTab:
    """GUI tab for image search operations using ephemeris data.

    Args:
        controller (ObjectDetectionController): The main application controller.
    """
    def __init__(self, controller):
        self.controller = controller

        # Widgets
        self.filters = pn.widgets.ToggleGroup(
            name='Filters',
            options=['u', 'g', 'r', 'i', 'z', 'y'],
            value=['r'],
            behavior="check",
            button_type='success',
            width=60,
            height=60,
            margin=5,
            align=('center', 'center'),
            orientation='horizontal'
        )
        self.run_button = pn.widgets.Button(name='Run Image Query', button_type='primary')
        self.results_pane = pn.pane.JSON(object={}, height=300)
        self.table_view = pn.widgets.Tabulator(
            sizing_mode='stretch_width',
            height=450,
            page_size=10,
            configuration={
                'columns': [
                    {'title': 'Visit ID', 'field': 'visit_id'},
                    {'title': 'Detector ID', 'field': 'detector_id'},
                    {'title': 'Filter', 'field': 'band'},
                    {'title': 'T min', 'field': 't_min'},
                    {'title': 'T max', 'field': 't_max'},
                ]
            }
        )

        # Layout
        self.layout = pn.Row(
            pn.Column(
                "### Image Query Parameters",
                pn.Column(
                    pn.Row("### Filters:", margin=(0, 10)),
                    pn.Row(self.filters, margin=(0, 10)),
                ),
                self.run_button,
                sizing_mode='stretch_width'
            ), self.table_view
        )

        self.run_button.on_click(self.run_query)


    def run_query(self, event):
        """Image query main logic"""
        # logger.info("Running Image search.")
        root_logger.info("Starting the image query...")
        input_data = {
            "image": {
                "filters": self.filters.value,
                "ephemeris_data": self.controller.ephemeris_results.get("ephemeris"),
            }
        }

        try:
            result = self.controller.api_connection(input_data)
            if 'image' in result:
                self.results_pane.object = result['image']
                df = pd.DataFrame(result['image'])
                df['t_min'] = df['t_min'].apply(lambda x: Time(x, format='mjd').iso)
                df['t_max'] = df['t_max'].apply(lambda x: Time(x, format='mjd').iso)
                df2 = df.filter(items=["visit_id", "detector_id", "band", "t_min", "t_max"])
                self.table_view.value = df2
            else:
                self.results_pane.object = result
                self.table_view.value = pd.DataFrame()
        except Exception as e:
            self.results_pane.object = {"error": str(e)}
            self.table_view.value = pd.DataFrame()

class PhotometryTab:
    """GUI tab for configuring and executing photometry analysis.

    Args:
        controller (ObjectDetectionController): The main application controller.
    """
    def __init__(self, controller):
        self.controller = controller

        # Widgets for photometry parameters
        self.image_type = pn.widgets.Select(name="Image type",
                                            options=["calexp", "goodSeeingDiff_differenceExp"],
                                            value="calexp"
                                           )
        self.detection_threshold = pn.widgets.FloatInput(name="Detection Threshold",
                                                         value=5.0,
                                                         start=0,
                                                         width=100
                                                        )
        self.cutout_size = pn.widgets.IntInput(name="Cutout Size (pixels)", value=800, start=100, width=100)
        self.save_cutouts = pn.widgets.Checkbox(name="Save Cutouts", value=False)
        self.display = pn.widgets.Checkbox(name="Display Results", value=False)
        self.save_json = pn.widgets.Checkbox(name="Save Result to JSON", value=False)
        self.run_button = pn.widgets.Button(name='Run Photometry', button_type='primary')

        # Results pane and table
        self.results_pane = pn.pane.JSON(object={}, height=300)
        self.table_view = pn.widgets.Tabulator(
            sizing_mode='stretch_width',
            height=450,
            page_size=10,
            configuration={
                'columns': [
                    {'title': 'Visit ID', 'field': 'visit_id'},
                    {'title': 'Detector ID', 'field': 'detector_id'},
                    {'title': 'Band', 'field': 'band'},
                    {'title': 'Flux (nJy)', 'field': 'flux'},
                    {'title': 'Flux Error', 'field': 'flux_err'},
                    {'title': 'Magnitude', 'field': 'mag'},
                    {'title': 'Mag Error', 'field': 'mag_err'},
                    {'title': 'SNR', 'field': 'snr'},
                    {'title': 'Nearby Sources', 'field': 'num_sources'},
                ]
            }
        )

        # Layout
        self.layout = pn.Row(
            pn.Column(
                "### Photometry Parameters",
                self.image_type,
                self.detection_threshold,
                self.cutout_size,
                self.save_cutouts,
                self.display,
                self.save_json,
                self.run_button,
                sizing_mode='stretch_width'
            ),
            self.table_view
        )

        self.run_button.on_click(self.run_photometry)

    def run_photometry(self, event):
        """
        Main logic for photometry
        """
        root_logger.info("Starting photometry process...")
        input_data = {
            "photometry": {
                "image_type": self.image_type.value,
                "threshold": self.detection_threshold.value,
                "min_cutout_size": self.cutout_size.value,
                "save_cutouts": self.save_cutouts.value,
                "display": self.display.value,
                "save_json": self.save_json.value
            }
        }
        try:
            result = self.controller.api_connection(input_data)
            if 'photometry' in result and result['photometry']:
                # Flatten the photometry results for the table
                photometry_data = result['photometry']
                rows = []
                for presult in photometry_data:
                    if presult.get('forced_phot_on_target'):
                        row = {
                            'visit_id': presult['visit_id'],
                            'detector_id': presult['detector_id'],
                            'band': presult['band'],
                            'flux': presult['forced_phot_on_target']['flux'],
                            'flux_err': presult['forced_phot_on_target']['flux_err'],
                            'mag': presult['forced_phot_on_target']['mag'],
                            'mag_err': presult['forced_phot_on_target']['mag_err'],
                            'snr': presult['forced_phot_on_target']['snr'],
                            'num_sources': len(presult.get('phot_within_error_ellipse', []))
                        }
                        rows.append(row)
                df = pd.DataFrame(rows)
                self.table_view.value = df
                self.results_pane.object = result
            else:
                self.table_view.value = pd.DataFrame()
                self.results_pane.object = {"error": "No photometry results found."}
        except Exception as e:
            self.results_pane.object = {"error": str(e)}
            self.table_view.value = pd.DataFrame()


class CompleteRunTab:
    """GUI tab for handling a complete run from ephemeris to photometry.

    Args:
        controller (ObjectDetectionController): The main application controller.
    """
    def __init__(self, controller):
        self.controller = controller

        # Ephemeris Section Widgets
        self.ephemeris_source = pn.widgets.RadioButtonGroup(
            name='Ephemeris Source',
            options=['Use Generated Data', 'Upload ECSV'],
            value='Use Generated Data'
        )
        self.file_upload = pn.widgets.input.FileInput(accept='.ecsv', multiple=False)
        self.service = pn.widgets.Select(name="Service", options=["Horizons", "Miriade"], value="Horizons")
        self.target_name = pn.widgets.TextInput(name="Target Name")
        self.target_type = pn.widgets.Select(name="Target Type",
                                             options=["smallbody", "comet_name"],
                                             value="smallbody",
                                            )
        self.start_time = pn.widgets.DatetimePicker(
            name="Start Time",
            value=datetime.datetime.now(),
            enable_time=True
        )
        self.time_spec = pn.widgets.RadioButtonGroup(
            name='Time Specification',
            options=['End Time', 'Day Range'],
            value='End Time'
        )
        self.end_time = pn.widgets.DatetimePicker(
            name="End Time",
            value=datetime.datetime.now() + datetime.timedelta(days=1),
            enable_time=True
        )
        self.day_range = pn.widgets.IntInput(name="Day Range", value=1, start=1, width=120)
        self.step_value = pn.widgets.FloatInput(
            name="Step Value",
            value=1,
            start=1,
            step=1,
            width=120
        )
        self.step_unit = pn.widgets.Select(
            name="Step Unit",
            options=['d', 'h', 'm'],
            value='h',
            width=50
        )
        self.save_data = pn.widgets.Checkbox(name="Save Ephemeris")

        # Image Section Widgets
        self.filters = pn.widgets.ToggleGroup(
            name='Filters',
            options=['u', 'g', 'r', 'i', 'z', 'y'],
            value=['r'],
            behavior="check",
            button_type='success'
        )

        # Photometry Section Widgets
        self.image_type = pn.widgets.Select(
            name="Image type",
            options=["calexp", "goodSeeingDiff_differenceExp"],
            value="calexp"
        )
        self.detection_threshold = pn.widgets.FloatInput(
            name="Detection Threshold",
            value=5.0,
            start=0,
            width=100
        )
        self.cutout_size = pn.widgets.IntInput(
            name="Cutout Size (pixels)",
            value=800,
            start=100,
            width=100
        )
        self.save_cutouts = pn.widgets.Checkbox(name="Save Cutouts", value=False)
        self.display = pn.widgets.Checkbox(name="Display Results", value=False)
        self.save_json = pn.widgets.Checkbox(name="Save Result to JSON", value=False)

        # Set up visibility bindings
        self.end_time.visible = pn.bind(lambda ts: ts == 'End Time', self.time_spec.param.value)
        self.day_range.visible = pn.bind(lambda ts: ts == 'Day Range', self.time_spec.param.value)
        self.start_time.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)
        self.end_time.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)
        self.day_range.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)
        self.step_value.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)
        self.step_unit.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)
        self.save_data.disabled = pn.bind(lambda s: s == 'Upload ECSV', self.ephemeris_source.param.value)

        # Results Table
        self.table_view = pn.widgets.Tabulator(
            sizing_mode='stretch_width',
            height=450,
            page_size=10,
            configuration={
                'columns': [
                    {'title': 'Visit ID', 'field': 'visit_id'},
                    {'title': 'Detector ID', 'field': 'detector_id'},
                    {'title': 'Band', 'field': 'band'},
                    {'title': 'Flux (nJy)', 'field': 'flux'},
                    {'title': 'Flux Error', 'field': 'flux_err'},
                    {'title': 'Magnitude', 'field': 'mag'},
                    {'title': 'Mag Error', 'field': 'mag_err'},
                    {'title': 'SNR', 'field': 'snr'},
                    {'title': 'Nearby Sources', 'field': 'num_sources'},
                ]
            }
        )

        # Run All Button
        self.run_all_button = pn.widgets.Button(name='Run All', button_type='primary')
        self.run_all_button.on_click(self.run_all)

        # Layout
        self.layout = pn.Row(
            pn.Column(
                pn.Card(
                    pn.Column(
                        "### Ephemeris Parameters",
                        self.ephemeris_source,
                        pn.bind(self.conditional_upload, self.ephemeris_source.param.value),
                        self.service,
                        self.target_name,
                        self.target_type,
                        self.start_time,
                        self.time_spec,
                        pn.Row(self.end_time, self.day_range),
                        pn.Row(self.step_value, self.step_unit),
                        self.save_data,
                    ),
                    title="Ephemeris Settings",
                    collapsed=False
                ),
                pn.Card(
                    pn.Column(
                        "### Image Search Parameters",
                        self.filters,
                    ),
                    title="Image Search Settings",
                    collapsed=False
                )
            ),
            pn.Column(
                pn.Card(
                    pn.Column(
                        "### Photometry Parameters",
                        self.image_type,
                        self.detection_threshold,
                        self.cutout_size,
                        self.save_cutouts,
                        self.display,
                        self.save_json,
                    ),
                    title="Photometry Settings",
                    collapsed=False
                ),
                self.run_all_button,
                self.table_view,
                sizing_mode='stretch_width'
            )
        )

    def conditional_upload(self, source):
        """Condition check for ephemeris data source"""
        return self.file_upload if source == 'Upload ECSV' else pn.pane.Str("Using generated ephemeris data.")

    def get_step_string(self):
        """Get the step value from GUI input field"""
        return f"{self.step_value.value}{self.step_unit.value}"

    def run_all(self, event):
        """End-to-end process"""
        root_logger.info("Starting complete run...")

        # Ephemeris Step
        if self.ephemeris_source.value == 'Upload ECSV':
            root_logger.info("Processing uploaded ECSV file.")
            if not self.file_upload.value:
                root_logger.error("Please upload an ECSV file.")
                self.table_view.value = pd.DataFrame()
                return
            try:
                # Read the uploaded ECSV
                content = io.BytesIO(self.file_upload.value)
                table = Table.read(content, format='ascii.ecsv')
                df = table.to_pandas()

                # Prepare input data for API
                input_data_ephemeris = {
                    "ephemeris": {
                        "service": self.service.value,
                        "target": self.target_name.value,
                        "target_type": self.target_type.value,
                        "ecsv_file": self.file_upload.filename
                    }
                }

                self.controller.ephemeris_results = self.controller.api_connection(input_data_ephemeris)
                root_logger.info("ECSV file processed successfully.")
            except Exception as e:
                root_logger.error(f"Error processing ECSV: {str(e)}")
                self.table_view.value = pd.DataFrame()
                return
        else:
            # Existing ephemeris generation logic
            input_data_ephemeris = {
                "ephemeris": {
                    "service": self.service.value,
                    "target": self.target_name.value,
                    "target_type": self.target_type.value,
                    "start": self.start_time.value.strftime('%Y-%m-%d %H:%M:%S'),
                    "save_data": self.save_data.value,
                    "observer_location": "X05",
                    "step": self.get_step_string()
                }
            }

            # Handle time specification
            if self.time_spec.value == 'End Time':
                input_data_ephemeris["ephemeris"]["end"] = self.end_time.value.strftime('%Y-%m-%d %H:%M:%S')
            else:
                try:
                    start_time = Time(self.start_time.value)
                    end_time = start_time + self.day_range.value * u.day
                    input_data_ephemeris["ephemeris"]["end"] = end_time.iso
                except Exception as e:
                    root_logger.error(f"Ephemeris time error: {str(e)}")
                    self.table_view.value = pd.DataFrame()
                    return

            try:
                root_logger.info("Running ephemeris query...")
                self.controller.ephemeris_results = self.controller.api_connection(input_data_ephemeris)
            except Exception as e:
                root_logger.error(f"Ephemeris query failed: {str(e)}")
                self.table_view.value = pd.DataFrame()
                return

        # Image Step
        input_data_image = {
            "image": {
                "filters": self.filters.value,
                "ephemeris_data": self.controller.ephemeris_results.get("ephemeris"),
            }
        }

        try:
            root_logger.info("Running image search...")
            self.controller.image_result = self.controller.api_connection(input_data_image)
        except Exception as e:
            root_logger.error(f"Image search failed: {str(e)}")
            self.table_view.value = pd.DataFrame()
            return

        # Photometry Step
        input_data_photometry = {
            "photometry": {
                "image_type": self.image_type.value,
                "threshold": self.detection_threshold.value,
                "min_cutout_size": self.cutout_size.value,
                "save_cutouts": self.save_cutouts.value,
                "display": self.display.value,
                "save_json": self.save_json.value
            }
        }

        try:
            root_logger.info("Running photometry...")
            photometry_result = self.controller.api_connection(input_data_photometry)
            if 'photometry' in photometry_result and photometry_result['photometry']:
                # Process results
                rows = []
                for presult in photometry_result['photometry']:
                    if presult.get('forced_phot_on_target'):
                        row = {
                            'visit_id': presult['visit_id'],
                            'detector_id': presult['detector_id'],
                            'band': presult['band'],
                            'flux': presult['forced_phot_on_target']['flux'],
                            'flux_err': presult['forced_phot_on_target']['flux_err'],
                            'mag': presult['forced_phot_on_target']['mag'],
                            'mag_err': presult['forced_phot_on_target']['mag_err'],
                            'snr': presult['forced_phot_on_target']['snr'],
                            'num_sources': len(presult.get('phot_within_error_ellipse', []))
                        }
                        rows.append(row)
                df = pd.DataFrame(rows)
                self.table_view.value = df
            else:
                self.table_view.value = pd.DataFrame()
                root_logger.warning("No photometry results found.")
        except Exception as e:
            root_logger.error(f"Photometry failed: {str(e)}")
            self.table_view.value = pd.DataFrame()

# Documentation tab
try:
    with open("control_panel_documentation.md", "r") as md_file:
        documentation_content = md_file.read()
except FileNotFoundError:
    documentation_content = "# Documentation\n\nError: File 'documentation.md' not found."
documentation_tab = pn.pane.Markdown(
    documentation_content,
    sizing_mode="stretch_width",
    styles={"background": "black", 'height': '600px', 'overflow-x': 'hidden',
            'overflow-y': 'auto', 'margin': '0', "padding": "20px"}
)


# Create the application
controller = ObjectDetectionController()
ephemeris_tab = EphemerisTab(controller).layout
image_tab = ImageTab(controller).layout
photometry_tab = PhotometryTab(controller).layout
complete_run_tab = CompleteRunTab(controller).layout

# Terminal test and clear
def test_logging():
    """Test the terminal widget"""
    logging.debug("Debug message")
    logging.info("Info message")
    logging.warning("Warning message")
    logging.error("Error message")
    print("Standard output message")
    print("Error message", file=sys.stderr)

test_log_button = pn.widgets.Button(name="Test Logging", button_type="success")
test_log_button.on_click(lambda e: test_logging())
test_button = pn.widgets.Button(name="Test Terminal", button_type="success")
test_button.on_click(lambda e: print("Terminal test successful!", flush=True))
clear_button = pn.widgets.Button(name="Clear Terminal", button_type="warning")
clear_button.on_click(lambda x: terminal.clear())

# Create the main layout for the app
template.main.append(
    pn.Column(
        pn.Tabs(
            ("Ephemeris", ephemeris_tab),
            ("Image", image_tab),
            ("Photometry", photometry_tab),
            ("Complete Run", complete_run_tab),
            ("Documentation", documentation_tab),
            sizing_mode='stretch_width'
        ),
        pn.Row(test_button, test_log_button, clear_button),
        terminal,
        sizing_mode='stretch_both'
    )
)

# Initialize the app
template.servable()
