import datetime
import io
import logging
import os
import sys

import astropy.units as u
import pandas as pd
import panel as pn
from astropy.table import Table
from astropy.time import Time
from ephemeris.data_loader import DataLoader
from ephemeris.data_model import QueryResult
from odc import ObjectDetectionController
from tornado import gen

# Set up logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Panel extensions and template
pn.extension("tabulator", "terminal", design="material")
pn.config.theme = "dark"

template = pn.template.MaterialTemplate(
    title="Faint Solar System Object Detection Service", logo="rubin_logo.svg"
)

# Initialize terminal and redirect
terminal = pn.widgets.Terminal(
    "Application Output:\n",
    options={
        "cursorBlink": True,
        "scrollback": 1000,
        "encoding": "utf-8",
        "fontSize": 11,
        "scrollOnOutput": True,
        "theme": {"background": "#000000"},
        "convertEol": True,
        "fitAddon": True,
    },
    height=3500,
    max_height=4000,
    sizing_mode="stretch_width",
    styles={
        "overflow-y": "auto",
        "height": "auto",
        "min-height": "800px",
    },
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
        self._setup_periodic_flush()

    def _setup_periodic_flush(self):
        """Set up a periodic callback to flush the terminal widget."""
        pn.state.add_periodic_callback(self._flush_terminal, period=100)

    def _flush_terminal(self):
        """Flush the terminal widget to ensure updates are displayed."""
        if hasattr(self.terminal_widget, "_comm") and self.terminal_widget._comm:
            self.terminal_widget._comm.send(self.terminal_widget._model_json())

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
            # Force update to browser
            pn.io.push_notebook(self.terminal_widget)
        except Exception:
            self.handleError(record)

    def flush(self):
        """Flush function for terminal widget"""
        if hasattr(self.terminal_widget, "_comm") and self.terminal_widget._comm:
            self.terminal_widget._comm.send(self.terminal_widget._model_json())


terminal_handler = TerminalHandler(terminal)
terminal_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

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
        self.logger = logging.getLogger("stdout" if not is_error else "stderr")
        self.logger.setLevel(logging.INFO)

    def write(self, buf):
        """Write function for terminal widget"""
        if buf.rstrip():
            sys.__stdout__.write(buf)
            if self.is_error:
                self.logger.error(buf.rstrip())
            else:
                self.logger.info(buf.rstrip())
            # Force update to browser
            pn.io.push_notebook(self.terminal)
        self.flush()

    def flush(self):
        """Flush function for terminal widget"""
        for handler in self.logger.handlers:
            handler.flush()


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
            "datetime": [serialize_time(t) for t in ephem.datetime],
            "RA_deg": ephem.RA_deg.tolist(),
            "DEC_deg": ephem.DEC_deg.tolist(),
            "RA_rate_arcsec_per_h": ephem.RA_rate_arcsec_per_h.tolist(),
            "DEC_rate_arcsec_per_h": ephem.DEC_rate_arcsec_per_h.tolist(),
            "AZ_deg": ephem.AZ_deg.tolist(),
            "EL_deg": ephem.EL_deg.tolist(),
            "r_au": ephem.r_au.tolist(),
            "delta_au": ephem.delta_au.tolist(),
            "V_mag": ephem.V_mag.tolist(),
            "alpha_deg": ephem.alpha_deg.tolist(),
            "RSS_3sigma_arcsec": ephem.RSS_3sigma_arcsec.tolist(),
            "SMAA_3sigma_arcsec": ephem.SMAA_3sigma_arcsec.tolist(),
            "SMIA_3sigma_arcsec": ephem.SMIA_3sigma_arcsec.tolist(),
            "Theta_3sigma_deg": ephem.Theta_3sigma_deg.tolist(),
        }

    return {
        "target": result.target,
        "start": serialize_time(result.start),
        "end": serialize_time(result.end),
        "ephemeris": serialize_ephemeris(result.ephemeris),
    }


class EphemerisTab:
    """
    GUI tab for managing ephemeris queries and displaying the results.

    This tab allows users to either upload an ECSV file with pre-computed ephemeris data
    or perform a live query using selected services (Horizons, Miriade) for a specific target
    within a defined time range and step. Query results are displayed in a table and JSON pane.

    Parameters
    ----------
    controller : ObjectDetectionController
        The main application controller instance, used to interact with the backend
        ephemeris services and store results.
    """

    def __init__(self, controller):
        self.controller = controller
        root_logger.warning(
            """Note: The image and photometry service may take a while.
            This terminal widget will not refresh in real-time until each full step completes.
            Please be patient after initiating a run."""
        )

        # Widgets
        self.ephemeris_source = pn.widgets.RadioButtonGroup(
            name="Ephemeris Source", options=["Use Existing Data", "Upload ECSV"], value="Use Existing Data"
        )
        self.file_upload = pn.widgets.input.FileInput(accept=".ecsv", multiple=False)
        self.service = pn.widgets.Select(name="Service", options=["Horizons", "Miriade"], value="Horizons")
        self.target_name = pn.widgets.TextInput(name="Target Name")
        self.target_type = pn.widgets.Select(
            name="Target Type",
            options=["smallbody", "asteroid_name", "comet_name", "designation"],
            value="smallbody",
        )
        self.start_time = pn.widgets.DatetimePicker(
            name="Start Time", value=datetime.datetime.now(), enable_time=True
        )
        self.time_spec = pn.widgets.RadioButtonGroup(
            name="Time Specification", options=["End Time", "Day Range"], value="End Time"
        )
        self.end_time = pn.widgets.DatetimePicker(
            name="End Time", value=datetime.datetime.now() + datetime.timedelta(days=1), enable_time=True
        )
        self.day_range = pn.widgets.IntInput(name="Day Range", value=1, start=1, width=120)
        self.step_value = pn.widgets.FloatInput(name="Step Value", value=1, start=1, step=1, width=120)
        self.step_unit = pn.widgets.Select(name="Step Unit", options=["d", "h", "m"], value="h", width=50)
        self.save_ephem_data = pn.widgets.Checkbox(name="Save Ephemeris")
        self.output_folder = pn.widgets.TextInput(name="Output folder", value="./output")
        self.run_button = pn.widgets.Button(
            name=pn.bind(
                lambda s: "Update" if s == "Upload ECSV" else "Run Query", self.ephemeris_source.param.value
            ),
            button_type="primary",
        )
        self.result_pane = pn.pane.JSON(object={}, name="Results", depth=3, height=300)

        # Set up visibility bindings
        self.end_time.visible = pn.bind(lambda ts: ts == "End Time", self.time_spec.param.value)
        self.day_range.visible = pn.bind(lambda ts: ts == "Day Range", self.time_spec.param.value)
        self.start_time.disabled = pn.bind(lambda s: s == "Upload ECSV", self.ephemeris_source.param.value)
        self.end_time.disabled = pn.bind(lambda s: s == "Upload ECSV", self.ephemeris_source.param.value)
        self.day_range.disabled = pn.bind(lambda s: s == "Upload ECSV", self.ephemeris_source.param.value)
        self.step_value.disabled = pn.bind(lambda s: s == "Upload ECSV", self.ephemeris_source.param.value)
        self.step_unit.disabled = pn.bind(lambda s: s == "Upload ECSV", self.ephemeris_source.param.value)
        self.save_ephem_data.disabled = pn.bind(
            lambda s: s == "Upload ECSV", self.ephemeris_source.param.value
        )
        self.output_folder.visible = pn.bind(
            lambda save_checked: save_checked, self.save_ephem_data.param.value
        )

        # Link button click
        self.run_button.on_click(self.run_query)

        # Visualization components with all EphemerisData fields
        self.table_view = pn.widgets.Tabulator(
            sizing_mode="stretch_width",
            height=450,
            page_size=10,
            configuration={
                "columns": [
                    {"title": "Time", "field": "datetime"},
                    {"title": "RA (deg)", "field": "RA_deg"},
                    {"title": "Dec (deg)", "field": "DEC_deg"},
                    {"title": "RA Rate", "field": "RA_rate_arcsec_per_h"},
                    {"title": "Dec Rate", "field": "DEC_rate_arcsec_per_h"},
                    {"title": "AZ (deg)", "field": "AZ_deg"},
                    {"title": "EL (deg)", "field": "EL_deg"},
                    {"title": "r (au)", "field": "r_au"},
                    {"title": "Delta (au)", "field": "delta_au"},
                    {"title": "V mag", "field": "V_mag"},
                    {"title": "Alpha (deg)", "field": "alpha_deg"},
                    {"title": "RSS 3σ", "field": "RSS_3sigma_arcsec"},
                    {"title": "SMAA 3σ", "field": "SMAA_3sigma_arcsec"},
                    {"title": "SMIA 3σ", "field": "SMIA_3sigma_arcsec"},
                    {"title": "Theta 3σ (deg)", "field": "Theta_3sigma_deg"},
                ]
            },
        )

        def conditional_upload(source):
            """Condition check for ephemeris data source"""
            return self.file_upload if source == "Upload ECSV" else pn.pane.Str("Using Ephemeris service.")

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
                    sizing_mode="stretch_width",
                ),
                self.save_ephem_data,
                self.output_folder,
                self.run_button,
                sizing_mode="stretch_width",
            ),
            self.table_view,
        )

    def get_step_string(self):
        """Combine step value and unit into the required format"""
        return f"{self.step_value.value}{self.step_unit.value}"

    async def run_query(self, event):
        """
        Handles the execution of the ephemeris query when the "Run Query" or "Update Table"
        button is clicked.

        This asynchronous method manages two main workflows:
        1.  **Uploading ECSV**: Reads an ECSV file, converts it to a Pandas DataFrame,
            and displays it in the table. It also attempts to pass the file to the controller.
        2.  **Running Live Query**: Constructs the input data for a live ephemeris query
            based on user selections (target, time range, step, service) and calls the
            'ObjectDetectionController''s API connection. Displays results in the table.

        Parameters
        ----------
        event : pn.viewable.singles.Button
            The button click event (unused, but required by Panel's on_click signature).
        """
        await gen.sleep(0.01)
        if self.ephemeris_source.value == "Upload ECSV":
            root_logger.info("Processing uploaded ECSV file.")
            if not self.file_upload.value:
                self.result_pane.object = {"error": "Please upload an ECSV file."}
                self.table_view.value = pd.DataFrame()
                return
            try:
                # Read the uploaded ECSV
                content = io.BytesIO(self.file_upload.value)
                table = Table.read(content, format="ascii.ecsv")
                df = table.to_pandas()

                # Update the table view
                self.table_view.value = df

                input_data = {
                    "ephemeris": {
                        "ephemeris_service": self.service.value.lower(),
                        "target": self.target_name.value,
                        "target_type": self.target_type.value,
                        "ecsv_file": self.file_upload.filename,
                    }
                }
                # TODO Fix this!! Right now, only works (control_panel) if the file
                # is already uploaded next to the control_panel.py
                self.controller.ephemeris_results = self.controller.api_connection(input_data)
                root_logger.info("ECSV file processed successfully.")
            except Exception as e:
                self.result_pane.object = {"error": str(e)}
                self.table_view.value = pd.DataFrame()
                root_logger.error(f"Error processing ECSV: {str(e)}")

        else:
            await gen.sleep(0.01)
            root_logger.info("Running Ephemeris service.")
            input_data = {
                "ephemeris": {
                    "ephemeris_service": self.service.value.lower(),
                    "target": self.target_name.value,
                    "target_type": self.target_type.value,
                    "start": self.start_time.value.strftime("%Y-%m-%d %H:%M:%S"),
                    "step": self.get_step_string(),
                    "observer_location": "X05",
                    "save_ephem_data": self.save_ephem_data.value,
                    "output_folder": self.output_folder.value,
                }
            }

            # Handle time specification
            if self.time_spec.value == "End Time":
                input_data["ephemeris"]["end"] = self.end_time.value.strftime("%Y-%m-%d %H:%M:%S")
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

                if "ephemeris" in self.controller.ephemeris_results and isinstance(
                    self.controller.ephemeris_results["ephemeris"], QueryResult
                ):
                    serialized_query = serialize_query_result(self.controller.ephemeris_results["ephemeris"])
                    ephemeris_data = serialized_query["ephemeris"]
                    df = pd.DataFrame(ephemeris_data)
                    self.table_view.value = df
                else:
                    self.result_pane.object = self.controller.ephemeris_results
                    self.table_view.value = pd.DataFrame()

            except Exception as e:
                self.result_pane.object = {"error": str(e)}
                self.table_view.value = pd.DataFrame()


class ImageTab:
    """
    GUI tab for configuring and executing image search operations using ephemeris data.

    This tab allows users to select image filters, define the search method (point or polygon),
    and configure polygon-specific parameters like widening and time interval. The results
    (image metadata) are displayed in a table.

    Parameters
    ----------
    controller : ObjectDetectionController
        The main application controller instance, which holds the ephemeris results
        and provides the image search functionality.
    """

    def __init__(self, controller):
        self.controller = controller

        # Widgets
        self.search_method = pn.widgets.RadioButtonGroup(
            name="Image Search Method",
            options=["Point", "Polygon"],
            value="Point",
        )

        self.filters = pn.widgets.ToggleGroup(
            name="Filters",
            options=["u", "g", "r", "i", "z", "y"],
            value=["r"],
            behavior="check",
            button_type="success",
            width=70,
            height=60,
            margin=5,
            align=("center", "center"),
            orientation="horizontal",
        )

        # Polygon-specific options
        self.widening = pn.widgets.FloatInput(
            name="Widening (arcsec)",
            value=1.0,
            start=0,
            step=0.5,
            width=120,
        )

        self.time_interval = pn.widgets.FloatInput(
            name="Time Interval (days)",
            value=5.0,
            start=0.1,
            step=0.5,
            width=120,
        )

        # Set up visibility bindings for polygon options
        self.widening.visible = pn.bind(lambda method: method == "Polygon", self.search_method.param.value)
        self.time_interval.visible = pn.bind(
            lambda method: method == "Polygon", self.search_method.param.value
        )

        self.run_button = pn.widgets.Button(name="Run Image Query", button_type="primary")
        self.results_pane = pn.pane.JSON(object={}, height=300)
        self.table_view = pn.widgets.Tabulator(
            sizing_mode="stretch_width",
            height=450,
            page_size=10,
            configuration={
                "columns": [
                    {"title": "Visit ID", "field": "visit_id"},
                    {"title": "Detector ID", "field": "detector_id"},
                    {"title": "Filter", "field": "band"},
                    {"title": "T min", "field": "t_min"},
                    {"title": "T max", "field": "t_max"},
                ]
            },
        )

        # Layout
        self.layout = pn.Row(
            pn.Column(
                "### Image Query Parameters",
                self.search_method,
                pn.Column(
                    pn.Row("### Filters:", margin=(0, 10)),
                    pn.Row(self.filters, margin=(0, 10)),
                ),
                # Polygon-specific options (conditionally visible)
                pn.Column(
                    "### Polygon Options",
                    self.widening,
                    self.time_interval,
                    visible=pn.bind(lambda method: method == "Polygon", self.search_method.param.value),
                ),
                self.run_button,
                sizing_mode="stretch_width",
            ),
            self.table_view,
        )

        self.run_button.on_click(self.run_query)

    async def run_query(self, event):
        """
        Handles the execution of the image search query when the "Run Image Search" button is clicked.

        This asynchronous method constructs the input data for the image search based on
        user selections (filters, search method, polygon parameters) and calls the
        'ObjectDetectionController''s API connection. It then displays the returned
        image metadata in a Pandas DataFrame within a Tabulator widget.

        Parameters
        ----------
        event : pn.viewable.singles.Button
            The button click event (unused, but required by Panel's on_click signature).
        """
        # logger.info("Running Image search.")
        await gen.sleep(0.01)
        # root_logger.info("Starting the image query...")

        input_data = {
            "image": {
                "filters": self.filters.value,
                "ephemeris_data": self.controller.ephemeris_results,
                "image_search_method": self.search_method.value.lower(),
            }
        }

        # Add polygon-specific parameters if polygon method is selected
        if self.search_method.value == "Polygon":
            input_data["image"]["widening"] = self.widening.value
            input_data["image"]["time_interval"] = self.time_interval.value

        try:
            result = self.controller.api_connection(input_data)
            if "image" in result:
                await gen.sleep(0.01)
                self.results_pane.object = result["image"]
                df = pd.DataFrame(result["image"])
                df["t_min"] = df["t_min"].apply(lambda x: Time(x, format="mjd").iso)
                df["t_max"] = df["t_max"].apply(lambda x: Time(x, format="mjd").iso)
                df2 = df.filter(items=["visit_id", "detector_id", "band", "t_min", "t_max"])
                self.table_view.value = df2
            else:
                await gen.sleep(0.01)
                self.results_pane.object = result
                self.table_view.value = pd.DataFrame()
        except Exception as e:
            self.results_pane.object = {"error": str(e)}
            self.table_view.value = pd.DataFrame()


class PhotometryTab:
    """
    GUI tab for configuring and executing photometry analysis on detected images.

    This tab allows users to specify photometry parameters such as image type,
    detection threshold, cutout size, and error ellipse override. It also provides
    options to save diagnostic plots, FITS files, and the final results in JSON/CSV formats.
    Results are displayed in a table summarizing photometry measurements.

    Parameters
    ----------
    controller : ObjectDetectionController
        The main application controller instance, which holds the image metadata results
        and provides the photometry processing functionality.
    """

    def __init__(self, controller):
        self.controller = controller

        # Widgets for photometry parameters
        self.image_type = pn.widgets.Select(
            name="Image type", options=["visit_image", "difference_image"], value="visit_image"
        )
        self.detection_threshold = pn.widgets.FloatInput(
            name="Detection Threshold", value=5.0, start=0, width=150
        )
        self.cutout_size = pn.widgets.IntInput(name="Cutout Size (pixels)", value=800, start=0, width=150)
        self.override_error = pn.widgets.FloatInput(
            name="Override error (arcsec)",
            value=0.0,
            start=0,
            step=0.1,
            width=150,
        )
        self.refine_ephemeris = pn.widgets.Checkbox(
            name="Refine Ephemeris at Observation Times",
            value=False,
        )
        self.save_diag_plots = pn.widgets.Checkbox(name="Save Diagnostic Plots", value=False)
        self.save_fits = pn.widgets.Checkbox(name="Save Fits", value=False)
        self.display = pn.widgets.Checkbox(name="Display Results", value=False)
        self.save_json = pn.widgets.Checkbox(name="Save Result to JSON", value=False)
        self.save_csv = pn.widgets.Checkbox(name="Save Result to csv", value=False)
        self.error_ellipse_sources = pn.widgets.Checkbox(
            name="Save all the sources within the error ellipse", value=False
        )
        self.output_folder = pn.widgets.TextInput(name="Output folder", value="./output")
        self.run_button = pn.widgets.Button(name="Run Photometry", button_type="primary")

        # Set up visibility bindings
        self.output_folder.visible = pn.bind(
            lambda diag, fits, json, csv: diag or fits or json or csv,
            self.save_diag_plots.param.value,
            self.save_fits.param.value,
            self.save_json.param.value,
            self.save_csv.param.value,
        )
        self.error_ellipse_sources.visible = pn.bind(
            lambda save_checked: save_checked, self.save_csv.param.value
        )

        # Results pane and table
        self.results_pane = pn.pane.JSON(object={}, height=300)
        self.table_view = pn.widgets.Tabulator(
            sizing_mode="stretch_width",
            height=450,
            page_size=10,
            configuration={
                "columns": [
                    {"title": "Visit ID", "field": "visit_id"},
                    {"title": "Detector ID", "field": "detector_id"},
                    {"title": "Band", "field": "band"},
                    {"title": "Flux (nJy)", "field": "flux"},
                    {"title": "Flux Error", "field": "flux_err"},
                    {"title": "Magnitude", "field": "mag"},
                    {"title": "Mag Error", "field": "mag_err"},
                    {"title": "SNR", "field": "snr"},
                    {"title": "Nearby Sources", "field": "num_sources"},
                ]
            },
        )

        # Layout
        self.layout = pn.Row(
            pn.Column(
                "### Photometry Parameters",
                self.image_type,
                self.detection_threshold,
                self.cutout_size,
                self.override_error,
                self.refine_ephemeris,
                self.save_diag_plots,
                self.save_fits,
                self.display,
                self.save_json,
                self.save_csv,
                self.output_folder,
                self.error_ellipse_sources,
                self.run_button,
                sizing_mode="stretch_width",
            ),
            self.table_view,
        )

        self.run_button.on_click(self.run_photometry)

    async def run_photometry(self, event):
        """
        Handles the execution of the photometry analysis when the "Run Photometry Analysis" button is clicked.

        This asynchronous method constructs the input data for the photometry process based on
        user selections (image type, detection threshold, saving options, etc.) and calls the
        'ObjectDetectionController''s API connection. It then processes and displays the
        photometry results in a Pandas DataFrame within a Tabulator widget.

        Parameters
        ----------
        event : pn.viewable.singles.Button
            The button click event (unused, but required by Panel's on_click signature).
        """

        await gen.sleep(0.01)
        root_logger.info("Starting photometry process...")
        input_data = {
            "photometry": {
                "image_type": self.image_type.value,
                "threshold": self.detection_threshold.value,
                "min_cutout_size": self.cutout_size.value,
                "override_error": self.override_error.value,
                "refine_ephemeris": self.refine_ephemeris.value,
                "save_diag_plots": self.save_diag_plots.value,
                "save_fits": self.save_fits.value,
                "display": self.display.value,
                "save_json": self.save_json.value,
                "save_csv": self.save_csv.value,
                "save_error_sources": self.error_ellipse_sources.value,
                "output_folder": self.output_folder.value,
            }
        }
        try:
            await gen.sleep(0.01)
            result = self.controller.api_connection(input_data)
            if "photometry" in result and result["photometry"]:
                # Flatten the photometry results for the table
                photometry_data = result["photometry"]
                rows = []
                for presult in photometry_data:
                    if presult.get("forced_phot_on_target"):
                        row = {
                            "visit_id": presult["visit_id"],
                            "detector_id": presult["detector_id"],
                            "band": presult["band"],
                            "flux": presult["forced_phot_on_target"]["flux"],
                            "flux_err": presult["forced_phot_on_target"]["flux_err"],
                            "mag": presult["forced_phot_on_target"]["mag"],
                            "mag_err": presult["forced_phot_on_target"]["mag_err"],
                            "snr": presult["forced_phot_on_target"]["snr"],
                            "num_sources": len(presult.get("phot_within_error_ellipse", [])),
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
    """
    GUI tab for handling a complete end-to-end run of the object detection pipeline,
    from ephemeris query to image search and finally photometry.

    This tab consolidates all parameters from the individual tabs into one interface,
    allowing users to configure and execute the entire workflow sequentially.
    Results of the final photometry step are displayed in a table.

    Parameters
    ----------
    controller : ObjectDetectionController
        The main application controller instance that orchestrates all steps of the pipeline.
    """

    def __init__(self, controller):
        self.controller = controller

        # Ephemeris Section Widgets
        self.ephemeris_source = pn.widgets.RadioButtonGroup(
            name="Ephemeris Source", options=["Use Generated Data", "Upload ECSV"], value="Use Generated Data"
        )
        self.file_upload = pn.widgets.input.FileInput(accept=".ecsv", multiple=False)
        self.service = pn.widgets.Select(name="Service", options=["Horizons", "Miriade"], value="Horizons")
        self.target_name = pn.widgets.TextInput(name="Target Name")
        self.target_type = pn.widgets.Select(
            name="Target Type", options=["smallbody", "comet_name", "designation"], value="smallbody"
        )
        self.start_time = pn.widgets.DatetimePicker(
            name="Start Time", value=datetime.datetime.now(), enable_time=True
        )
        self.time_spec = pn.widgets.RadioButtonGroup(
            name="Time Specification", options=["End Time", "Day Range"], value="End Time"
        )
        self.end_time = pn.widgets.DatetimePicker(
            name="End Time", value=datetime.datetime.now() + datetime.timedelta(days=1), enable_time=True
        )
        self.day_range = pn.widgets.IntInput(name="Day Range", value=1, start=1, width=120)
        self.step_value = pn.widgets.FloatInput(name="Step Value", value=12, start=1, step=1, width=120)
        self.step_unit = pn.widgets.Select(name="Step Unit", options=["d", "h", "m"], value="h", width=50)
        self.save_ephem_data = pn.widgets.Checkbox(name="Save Ephemeris")
        self.output_folder = pn.widgets.TextInput(name="Output folder", value="./output")

        # Image Section Widgets
        self.search_method = pn.widgets.RadioButtonGroup(
            name="Image Search Method",
            options=["Point", "Polygon"],
            value="Point",
        )

        self.filters = pn.widgets.ToggleGroup(
            name="Filters",
            options=["u", "g", "r", "i", "z", "y"],
            value=["r"],
            behavior="check",
            button_type="success",
        )

        # Polygon-specific options
        self.widening = pn.widgets.FloatInput(
            name="Widening (arcsec)",
            value=1.0,
            start=0,
            step=0.5,
            width=120,
        )

        self.time_interval = pn.widgets.FloatInput(
            name="Time Interval (days)",
            value=5.0,
            start=0.1,
            step=0.5,
            width=120,
        )

        # Photometry Section Widgets
        self.image_type = pn.widgets.Select(
            name="Image type", options=["visit_image", "difference_image"], value="visit_image"
        )
        self.detection_threshold = pn.widgets.FloatInput(
            name="Detection Threshold", value=5.0, start=0, width=150
        )
        self.cutout_size = pn.widgets.IntInput(name="Cutout Size (pixels)", value=800, start=0, width=150)
        self.override_error = pn.widgets.FloatInput(
            name="Override error (arcsec)",
            value=0.0,
            start=0,
            step=0.1,
            width=150,
        )
        self.refine_ephemeris = pn.widgets.Checkbox(
            name="Refine Ephemeris at Observation Times",
            value=False,
        )
        self.save_diag_plots = pn.widgets.Checkbox(name="Save Diagnostic Plots", value=False)
        self.save_fits = pn.widgets.Checkbox(name="Save Fits", value=False)
        self.display = pn.widgets.Checkbox(name="Display Results", value=False)
        self.save_json = pn.widgets.Checkbox(name="Save Result to JSON", value=False)
        self.save_csv = pn.widgets.Checkbox(name="Save Result to csv", value=False)
        self.error_ellipse_sources = pn.widgets.Checkbox(
            name="Save all the sources within the error ellipse", value=False
        )
        self.output_folder = pn.widgets.TextInput(name="Output folder", value="./output")

        # Set up visibility bindings
        self.end_time.visible = pn.bind(lambda ts: ts == "End Time", self.time_spec.param.value)
        self.day_range.visible = pn.bind(lambda ts: ts == "Day Range", self.time_spec.param.value)
        self.start_time.disabled = pn.bind(lambda s: s == "Upload ECSV", self.ephemeris_source.param.value)
        self.end_time.disabled = pn.bind(lambda s: s == "Upload ECSV", self.ephemeris_source.param.value)
        self.day_range.disabled = pn.bind(lambda s: s == "Upload ECSV", self.ephemeris_source.param.value)
        self.step_value.disabled = pn.bind(lambda s: s == "Upload ECSV", self.ephemeris_source.param.value)
        self.step_unit.disabled = pn.bind(lambda s: s == "Upload ECSV", self.ephemeris_source.param.value)
        self.save_ephem_data.disabled = pn.bind(
            lambda s: s == "Upload ECSV", self.ephemeris_source.param.value
        )

        # Visibility for polygon options in Image section
        self.widening.visible = pn.bind(lambda method: method == "Polygon", self.search_method.param.value)
        self.time_interval.visible = pn.bind(
            lambda method: method == "Polygon", self.search_method.param.value
        )

        # Combined visibility for output_folder
        self.output_folder.visible = pn.bind(
            lambda save_eph, save_diag, save_fits, save_js, save_csv: (
                save_eph or save_diag or save_fits or save_js or save_csv
            ),
            self.save_ephem_data.param.value,
            self.save_diag_plots.param.value,
            self.save_fits.param.value,
            self.save_json.param.value,
            self.save_csv.param.value,
        )

        # Visibility for error ellipse sources in Photometry section
        self.error_ellipse_sources.visible = pn.bind(lambda save_csv: save_csv, self.save_csv.param.value)

        # Results Table
        self.table_view = pn.widgets.Tabulator(
            sizing_mode="stretch_width",
            height=450,
            page_size=10,
            configuration={
                "columns": [
                    {"title": "Visit ID", "field": "visit_id"},
                    {"title": "Detector ID", "field": "detector_id"},
                    {"title": "Band", "field": "band"},
                    {"title": "Flux (nJy)", "field": "flux"},
                    {"title": "Flux Error", "field": "flux_err"},
                    {"title": "Magnitude", "field": "mag"},
                    {"title": "Mag Error", "field": "mag_err"},
                    {"title": "SNR", "field": "snr"},
                    {"title": "Nearby Sources", "field": "num_sources"},
                ]
            },
        )

        # Run All Button
        self.run_all_button = pn.widgets.Button(name="Run All", button_type="primary")
        self.run_all_button.on_click(self.run_all)

        # --- Layout ---
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
                        self.save_ephem_data,
                    ),
                    title="Ephemeris Settings",
                    collapsed=False,
                ),
                pn.Card(
                    pn.Column(
                        "### Image Search Parameters",
                        self.search_method,
                        pn.Row("### Filters:", margin=(0, 10)),
                        pn.Row(self.filters, margin=(0, 10)),
                        pn.Column(
                            "### Polygon Options",
                            self.widening,
                            self.time_interval,
                            visible=pn.bind(
                                lambda method: method == "Polygon", self.search_method.param.value
                            ),
                        ),
                    ),
                    title="Image Search Settings",
                    collapsed=False,
                ),
            ),
            pn.Column(
                pn.Card(
                    pn.Column(
                        "### Photometry Parameters",
                        self.image_type,
                        self.detection_threshold,
                        self.cutout_size,
                        self.override_error,
                        self.refine_ephemeris,
                        self.save_diag_plots,
                        self.save_fits,
                        self.display,
                        self.save_json,
                        self.save_csv,
                        self.error_ellipse_sources,
                    ),
                    title="Photometry Settings",
                    collapsed=False,
                ),
                pn.Card(
                    pn.Column(
                        "### Output Settings",
                        self.output_folder,
                    ),
                    title="Output",
                    collapsed=False,
                ),
                self.run_all_button,
                self.table_view,
                sizing_mode="stretch_width",
            ),
        )

    def conditional_upload(self, source):
        """Condition check for ephemeris data source"""
        return self.file_upload if source == "Upload ECSV" else pn.pane.Str("Using generated ephemeris data.")

    def get_step_string(self):
        """Get the step value from GUI input field"""
        return f"{self.step_value.value}{self.step_unit.value}"

    async def run_all(self, event):
        """
        Handles the execution of the entire end-to-end object detection pipeline
        when the "Run All Services" button is clicked.

        This asynchronous method orchestrates the sequential execution of:
        1.  **Ephemeris Query**: Based on user selection (live query or ECSV upload).
        2.  **Image Search**: Uses the results from the ephemeris step.
        3.  **Photometry Analysis**: Uses results from the image search step.

        It logs progress and updates the final photometry results table.

        Parameters
        ----------
        event : pn.viewable.singles.Button
            The button click event (unused, but required by Panel's on_click signature).
        """
        await gen.sleep(0.01)
        root_logger.info("Starting complete run...")

        # Ephemeris Step
        if self.ephemeris_source.value == "Upload ECSV":
            root_logger.info("Processing uploaded ECSV file.")
            if not self.file_upload.value:
                root_logger.error("Please upload an ECSV file.")
                self.table_view.value = pd.DataFrame()
                return
            try:
                await gen.sleep(0.01)
                # Read the uploaded ECSV
                # content = io.BytesIO(self.file_upload.value)
                # table = Table.read(content, format="ascii.ecsv")
                # df = table.to_pandas()

                # Prepare input data for API
                input_data_ephemeris = {
                    "ephemeris": {
                        "ephemeris_service": self.service.value,
                        "target": self.target_name.value,
                        "target_type": self.target_type.value,
                        "ecsv_file": self.file_upload.filename,
                    }
                }

                self.controller.ephemeris_results = self.controller.api_connection(input_data_ephemeris)
                root_logger.info("ECSV file processed successfully.")
                await gen.sleep(0.01)
            except Exception as e:
                root_logger.error(f"Error processing ECSV: {str(e)}")
                self.table_view.value = pd.DataFrame()
                return
        else:
            input_data_ephemeris = {
                "ephemeris": {
                    "ephemeris_service": self.service.value,
                    "target": self.target_name.value,
                    "target_type": self.target_type.value,
                    "start": self.start_time.value.strftime("%Y-%m-%d %H:%M:%S"),
                    "save_ephem_data": self.save_ephem_data.value,
                    "output_folder": self.output_folder.value,
                    "observer_location": "X05",
                    "step": self.get_step_string(),
                }
            }

            # Handle time specification
            if self.time_spec.value == "End Time":
                input_data_ephemeris["ephemeris"]["end"] = self.end_time.value.strftime("%Y-%m-%d %H:%M:%S")
            else:
                try:
                    start_time = Time(self.start_time.value)
                    end_time = start_time + self.day_range.value * u.day
                    input_data_ephemeris["ephemeris"]["end"] = end_time.iso
                    await gen.sleep(0.01)
                except Exception as e:
                    root_logger.error(f"Ephemeris time error: {str(e)}")
                    self.table_view.value = pd.DataFrame()
                    return

            try:
                root_logger.info("Running ephemeris query...")
                self.controller.ephemeris_results = self.controller.api_connection(input_data_ephemeris)
                await gen.sleep(0.01)
            except Exception as e:
                root_logger.error(f"Ephemeris query failed: {str(e)}")
                self.table_view.value = pd.DataFrame()
                return

        # Image Step
        await gen.sleep(0.01)
        input_data_image = {
            "image": {
                "filters": self.filters.value,
                "ephemeris_data": self.controller.ephemeris_results,
                "image_search_method": self.search_method.value.lower(),
            }
        }

        if self.search_method.value == "Polygon":
            input_data_image["image"]["widening"] = self.widening.value
            input_data_image["image"]["time_interval"] = self.time_interval.value

        try:
            root_logger.info("Running image search...")
            self.controller.api_connection(input_data_image)
            await gen.sleep(0.01)
        except Exception as e:
            root_logger.error(f"Image search failed: {str(e)}")
            self.table_view.value = pd.DataFrame()
            return

        # Photometry Step
        await gen.sleep(0.01)
        input_data_photometry = {
            "photometry": {
                "image_type": self.image_type.value,
                "threshold": self.detection_threshold.value,
                "min_cutout_size": self.cutout_size.value,
                "override_error": self.override_error.value,
                "refine_ephemeris": self.refine_ephemeris.value,
                "save_diag_plots": self.save_diag_plots.value,
                "save_fits": self.save_fits.value,
                "display": self.display.value,
                "save_json": self.save_json.value,
                "save_csv": self.save_csv.value,
                "save_error_sources": self.error_ellipse_sources.value,
                "output_folder": self.output_folder.value,
            }
        }

        try:
            root_logger.info("Running photometry...")
            await gen.sleep(0.01)
            photometry_result = self.controller.api_connection(input_data_photometry)
            if "photometry" in photometry_result and photometry_result["photometry"]:
                # Process results
                rows = []
                for presult in photometry_result["photometry"]:
                    if presult.get("forced_phot_on_target"):
                        row = {
                            "visit_id": presult["visit_id"],
                            "detector_id": presult["detector_id"],
                            "band": presult["band"],
                            "flux": presult["forced_phot_on_target"]["flux"],
                            "flux_err": presult["forced_phot_on_target"]["flux_err"],
                            "mag": presult["forced_phot_on_target"]["mag"],
                            "mag_err": presult["forced_phot_on_target"]["mag_err"],
                            "snr": presult["forced_phot_on_target"]["snr"],
                            "num_sources": len(presult.get("phot_within_error_ellipse", [])),
                        }
                        rows.append(row)
                df = pd.DataFrame(rows)
                self.table_view.value = df
            else:
                await gen.sleep(0.01)
                self.table_view.value = pd.DataFrame()
                root_logger.warning("No photometry results found.")
        except Exception as e:
            root_logger.error(f"Photometry failed: {str(e)}")
            self.table_view.value = pd.DataFrame()


# Documentation tab
class StandalonePhotometryTab:
    """
    GUI tab for standalone photometry without ephemeris dependency.

    This tab allows users to perform forced photometry at arbitrary coordinates
    without requiring ephemeris queries. Supports three input modes:
    1. Single coordinate measurement
    2. Batch CSV processing
    3. Multiple coordinates in same image

    Parameters
    ----------
    controller : ObjectDetectionController
        The main application controller instance.
    """

    def __init__(self, controller):
        self.controller = controller

        # Input mode selector
        self.input_mode = pn.widgets.Select(
            name="Input Mode",
            options=["Single Coordinate", "Batch CSV", "Multiple in Image"],
            value="Single Coordinate",
        )

        # Image specification widgets
        self.visit_id = pn.widgets.IntInput(name="Visit ID", value=512055, step=1, width=150)
        self.detector = pn.widgets.IntInput(name="Detector ID", value=75, step=1, start=0, width=150)
        self.band = pn.widgets.Select(name="Band", options=["u", "g", "r", "i", "z", "y"], value="g")

        # Single coordinate widgets
        self.ra = pn.widgets.FloatInput(name="RA (degrees)", value=53.076, step=0.001, width=200)
        self.dec = pn.widgets.FloatInput(name="Dec (degrees)", value=-28.110, step=0.001, width=200)

        # CSV upload widget
        self.csv_upload = pn.widgets.FileInput(
            name="Upload CSV", accept=".csv", sizing_mode="stretch_width"
        )

        # Multiple coordinates text area
        self.coords_text = pn.widgets.TextAreaInput(
            name="Coordinates (RA, Dec - one per line)",
            placeholder="53.076, -28.110\n53.080, -28.115",
            height=150,
            sizing_mode="stretch_width",
        )

        # Common parameters
        self.error_radius = pn.widgets.FloatInput(
            name="Error Radius (arcsec)", value=3.0, step=0.5, start=0, width=150
        )
        self.detection_threshold = pn.widgets.FloatInput(
            name="Detection Threshold (SNR)", value=5.0, step=0.5, start=1.0, width=150
        )
        self.image_type = pn.widgets.Select(
            name="Image Type", options=["visit_image", "difference_image"], value="visit_image"
        )

        # Save options
        self.save_diag_plots = pn.widgets.Checkbox(name="Save Diagnostic Plots", value=False)
        self.save_fits = pn.widgets.Checkbox(name="Save FITS Cutouts", value=False)
        self.save_csv = pn.widgets.Checkbox(name="Save Results CSV", value=False)
        self.save_json = pn.widgets.Checkbox(name="Save Results JSON", value=False)
        self.all_ellipse_sources = pn.widgets.Checkbox(
            name="Save all sources within error ellipse", value=False
        )
        self.output_folder = pn.widgets.TextInput(name="Output Folder", value="./output")

        # Run button
        self.run_button = pn.widgets.Button(
            name="Run Standalone Photometry", button_type="primary", sizing_mode="stretch_width"
        )

        # Results table
        self.table_view = pn.widgets.Tabulator(
            sizing_mode="stretch_width",
            height=450,
            page_size=20,
            pagination="remote",
        )

        # Download button (initially disabled)
        self.download_button = pn.widgets.FileDownload(
            callback=self._download_csv,
            filename="standalone_results.csv",
            button_type="success",
            label="Download Results CSV",
            sizing_mode="stretch_width",
        )

        # Create conditional layout based on input mode
        self.input_section = pn.Column(sizing_mode="stretch_width")
        self._update_input_section()

        # Make all_ellipse_sources visible only when save_csv is checked
        # Note: This option is only relevant for CSV output, not JSON
        self.all_ellipse_sources.visible = pn.bind(
            lambda save_checked: save_checked, self.save_csv.param.value
        )

        # Layout
        self.layout = pn.Row(
            pn.Column(
                "### Standalone Photometry",
                pn.pane.Markdown(
                    "*Perform photometry at arbitrary coordinates without ephemeris*",
                    styles={"font-style": "italic", "color": "#888"},
                ),
                "---",
                self.input_mode,
                self.input_section,
                "---",
                "### Common Parameters",
                self.error_radius,
                self.detection_threshold,
                self.image_type,
                "---",
                "### Output Options",
                self.save_diag_plots,
                self.save_fits,
                self.save_csv,
                self.save_json,
                self.all_ellipse_sources,
                self.output_folder,
                "---",
                self.run_button,
                min_width=400,
                max_width=500,
            ),
            pn.Column(
                "### Results",
                self.table_view,
                self.download_button,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_both",
        )

        # Set up event handlers
        self.input_mode.param.watch(self._on_input_mode_change, "value")
        self.run_button.on_click(self.run_standalone_photometry)

    def _update_input_section(self):
        """Update the input section based on selected mode."""
        mode = self.input_mode.value

        if mode == "Single Coordinate":
            self.input_section.objects = [
                self.visit_id,
                self.detector,
                self.band,
                self.ra,
                self.dec,
            ]
        elif mode == "Batch CSV":
            self.input_section.objects = [
                self.csv_upload,
                pn.pane.Markdown(
                    "**CSV Format**: visit_id, detector, band, ra, dec, [error_radius], [target_name]",
                    styles={"font-size": "0.9em", "color": "#aaa"},
                ),
            ]
        elif mode == "Multiple in Image":
            self.input_section.objects = [
                self.visit_id,
                self.detector,
                self.band,
                self.coords_text,
            ]

    def _on_input_mode_change(self, event):
        """Handle input mode changes."""
        self._update_input_section()

    def _download_csv(self):
        """Callback for CSV download."""
        if hasattr(self, "results_df") and self.results_df is not None:
            return io.StringIO(self.results_df.to_csv(index=False))
        return io.StringIO("No results available")

    async def run_standalone_photometry(self, event):
        """
        Execute standalone photometry based on selected input mode.

        Parameters
        ----------
        event : pn.viewable.singles.Button
            The button click event.
        """
        from photometry_api import PhotometryRequest, StandalonePhotometryService

        await gen.sleep(0.01)
        root_logger.info("Starting standalone photometry...")

        try:
            service = StandalonePhotometryService(
                output_folder=self.output_folder.value,
                detection_threshold=self.detection_threshold.value,
            )

            mode = self.input_mode.value

            if mode == "Single Coordinate":
                # Single coordinate mode
                request = PhotometryRequest(
                    visit_id=self.visit_id.value,
                    detector=self.detector.value,
                    band=self.band.value,
                    ra=self.ra.value,
                    dec=self.dec.value,
                    error_radius=self.error_radius.value,
                    detection_threshold=self.detection_threshold.value,
                    image_type=self.image_type.value,
                )

                result = service.measure_single(
                    request=request,
                    save_diag_plots=self.save_diag_plots.value,
                    save_fits=self.save_fits.value,
                    output_folder=self.output_folder.value,
                )

                # Convert to DataFrame
                results_df = service._results_to_dataframe(
                    [result], [request], include_all_ellipse_sources=self.all_ellipse_sources.value
                )
                self.results_df = results_df
                self.table_view.value = results_df

                # Save CSV if requested
                if self.save_csv.value:
                    csv_path = f"{self.output_folder.value}/standalone_results.csv"
                    os.makedirs(self.output_folder.value, exist_ok=True)
                    results_df.to_csv(csv_path, index=False)
                    root_logger.info(f"Results saved to: {csv_path}")

                # Save JSON if requested
                if self.save_json.value:
                    json_path = f"{self.output_folder.value}/standalone_results.json"
                    os.makedirs(self.output_folder.value, exist_ok=True)
                    service._results_to_json([result], json_path)
                    root_logger.info(f"Results saved to: {json_path}")

                root_logger.info("Single measurement complete")

            elif mode == "Batch CSV":
                # CSV batch mode
                if self.csv_upload.value is None:
                    root_logger.error("No CSV file uploaded")
                    return

                # Save uploaded CSV temporarily
                import tempfile

                with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".csv") as f:
                    f.write(self.csv_upload.value)
                    temp_csv = f.name

                try:
                    results_df = service.measure_from_csv(
                        csv_path=temp_csv,
                        save_diag_plots=self.save_diag_plots.value,
                        save_fits=self.save_fits.value,
                        output_folder=self.output_folder.value,
                        output_csv=(
                            f"{self.output_folder.value}/standalone_results.csv"
                            if self.save_csv.value
                            else None
                        ),
                        output_json=(
                            f"{self.output_folder.value}/standalone_results.json"
                            if self.save_json.value
                            else None
                        ),
                        all_ellipse_sources=self.all_ellipse_sources.value,
                        default_error_radius=self.error_radius.value,
                        default_detection_threshold=self.detection_threshold.value,
                        default_image_type=self.image_type.value,
                    )

                    self.results_df = results_df
                    self.table_view.value = results_df

                    root_logger.info(
                        f"Batch processing complete: {len(results_df)} measurements, "
                        f"{results_df['success'].sum()} successful"
                    )
                finally:
                    os.unlink(temp_csv)

            elif mode == "Multiple in Image":
                # Multiple coordinates in same image
                coords_lines = self.coords_text.value.strip().split("\n")
                coordinates = []
                for line in coords_lines:
                    if line.strip() and not line.startswith("#"):
                        try:
                            ra, dec = map(float, line.strip().split(","))
                            coordinates.append((ra, dec))
                        except ValueError:
                            root_logger.warning(f"Skipping invalid line: {line}")

                if not coordinates:
                    root_logger.error("No valid coordinates provided")
                    return

                results_dict = service.measure_multi_targets_in_image(
                    visit_id=self.visit_id.value,
                    detector=self.detector.value,
                    band=self.band.value,
                    coordinates=coordinates,
                    error_radius=self.error_radius.value,
                    image_type=self.image_type.value,
                    save_diag_plots=self.save_diag_plots.value,
                    save_fits=self.save_fits.value,
                    output_folder=self.output_folder.value,
                )

                # Convert to DataFrame
                results_list = list(results_dict.values())
                requests_list = [
                    PhotometryRequest(
                        visit_id=self.visit_id.value,
                        detector=self.detector.value,
                        band=self.band.value,
                        ra=ra,
                        dec=dec,
                        target_name=name,
                    )
                    for (ra, dec), name in zip(coordinates, results_dict.keys())
                ]
                results_df = service._results_to_dataframe(
                    results_list, requests_list, include_all_ellipse_sources=self.all_ellipse_sources.value
                )
                self.results_df = results_df
                self.table_view.value = results_df

                # Save CSV if requested
                if self.save_csv.value:
                    csv_path = f"{self.output_folder.value}/standalone_results.csv"
                    os.makedirs(self.output_folder.value, exist_ok=True)
                    results_df.to_csv(csv_path, index=False)
                    root_logger.info(f"Results saved to: {csv_path}")

                # Save JSON if requested
                if self.save_json.value:
                    json_path = f"{self.output_folder.value}/standalone_results.json"
                    os.makedirs(self.output_folder.value, exist_ok=True)
                    service._results_to_json(results_list, json_path)
                    root_logger.info(f"Results saved to: {json_path}")

                root_logger.info(f"Multi-target processing complete: {len(results_df)} measurements")

        except Exception as e:
            root_logger.error(f"Standalone photometry failed: {str(e)}")
            import traceback

            root_logger.error(traceback.format_exc())


try:
    with open("control_panel_documentation.md", "r") as md_file:
        documentation_content = md_file.read()
except FileNotFoundError:
    documentation_content = "# Documentation\n\nError: File documentation.md not found."
documentation_tab = pn.pane.Markdown(
    documentation_content,
    sizing_mode="stretch_width",
    styles={
        "background": "black",
        "height": "800px",
        "overflow-x": "hidden",
        "overflow-y": "auto",
        "margin": "0",
        "padding": "20px",
    },
)


# Create the application
controller = ObjectDetectionController()
data_loader = DataLoader()
ephemeris_tab = EphemerisTab(controller).layout
image_tab = ImageTab(controller).layout
photometry_tab = PhotometryTab(controller).layout
standalone_tab = StandalonePhotometryTab(controller).layout
complete_run_tab = CompleteRunTab(controller).layout


# Terminal test and clear
def test_logging():
    """
    Tests the logging system by emitting messages at different log levels
    and writing to stdout/stderr. These messages should appear in the
    Terminal widget.
    """
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
            ("Standalone Photometry", standalone_tab),
            ("Complete Run", complete_run_tab),
            ("Documentation", documentation_tab),
            sizing_mode="stretch_width",
        ),
        pn.Row(test_button, test_log_button, clear_button),
        terminal,
        sizing_mode="stretch_both",
    )
)

# Initialize the app
template.servable()
