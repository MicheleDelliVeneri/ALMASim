from typing import TYPE_CHECKING, Any, overload
import PyQt6
from PyQt6.sip import wrappertype

if TYPE_CHECKING:
    from PyQt6.QtCore import QEvent, QPoint, QObject, Qt, QRunnable
    from PyQt6.QtGui import QFont, QPalette, QWindow
    from PyQt6.QtWidgets import (
        QMainWindow,
        QWidget,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QDialog,
        QStyle,
    )
from _typeshed import Incomplete
from astropy.time import Time as Time
from datetime import date as date
from distributed import WorkerPlugin

class _PyQtWrapperType(type):
    pass

class MemoryLimitPlugin(WorkerPlugin):
    memory_limit: Incomplete
    def __init__(self, memory_limit) -> None: ...
    def setup(self, worker) -> None: ...
    def teardown(self, worker) -> None: ...
    def transition(self, key, start, finish, *args, **kwargs): ...

class TerminalLogger(QObject, metaclass=_PyQtWrapperType):
    log_signal: Incomplete
    terminal: Incomplete
    def __init__(self, terminal) -> None: ...
    def add_log(self, message) -> None: ...
    def update_progress(self, value) -> None: ...

class PlotWindow(QWidget, metaclass=_PyQtWrapperType):
    layout: Incomplete
    initial_width: int
    initial_height: int
    scroll_area: Incomplete
    scroll_widget: Incomplete
    scroll_layout: Incomplete
    def __init__(self, parent: Incomplete | None = None) -> None: ...
    def resizeEvent(self, event) -> None: ...
    def update_plot_sizes(self) -> None: ...
    def create_science_keyword_plots(self) -> None: ...

class SignalEmitter(QObject, metaclass=_PyQtWrapperType):
    simulationFinished: Incomplete
    progress: Incomplete

class SimulatorRunnable(QRunnable, QObject, metaclass=_PyQtWrapperType):
    alma_simulator: Incomplete
    args: Incomplete
    kwargs: Incomplete
    signals: Incomplete
    def __init__(self, alma_simulator_instance, *args, **kwargs) -> None: ...
    def run(self) -> None: ...

class ParallelSimulatorRunnable(QRunnable, metaclass=_PyQtWrapperType):
    alma_simulator: Incomplete
    def __init__(self, alma_simulator_instance) -> None: ...
    def run(self) -> None: ...

class ParallelSimulatorRunnableRemote(QRunnable, metaclass=_PyQtWrapperType):
    alma_simulator_instance: Incomplete
    input_params: Incomplete
    def __init__(self, alma_simulator_instance, input_params) -> None: ...
    def run(self) -> None: ...

class SlurmSimulatorRunnableRemote(QRunnable, metaclass=_PyQtWrapperType):
    alma_simulator_instance: Incomplete
    input_params: Incomplete
    def __init__(self, alma_simulator_instance, input_params) -> None: ...
    def run(self) -> None: ...

class SimulatorWorker(QRunnable, QObject, metaclass=_PyQtWrapperType):
    alma_simulator: Incomplete
    df: Incomplete
    signals: Incomplete
    def __init__(self, alma_simulator_instance, df, *args, **kwargs) -> None: ...
    def run(self) -> None: ...

class DownloadGalaxyZooRunnable(QRunnable, metaclass=_PyQtWrapperType):
    """Runnable for downloading Galaxy Zoo data in a separate thread."""

    alma_simulator: Incomplete  # Note: This should be "ALMASimulator"
    def __init__(self, alma_simulator_instance) -> None: ...  # Corrected annotation
    def run(self) -> None: ...

# ... other imports and class definitions ...

# ... other imports
# ... _PyQtWrapperType definition ...

# ... ALMASimulator definition ...
class QApplication(PyQt6.QtWidgets.QApplication, metaclass=_PyQtWrapperType):
    @classmethod
    def aboutQt(cls) -> None: ...
    @classmethod
    def activeModalWidget(cls) -> QWidget | None: ...
    @classmethod
    def activePopupWidget(cls) -> QWidget | None: ...
    @classmethod
    def activeWindow(cls) -> QWidget | None: ...
    @classmethod
    def alert(cls, widget: QWidget | None, msecs: int = 0) -> None: ...
    @classmethod
    def allWidgets(cls) -> list[QWidget]: ...
    @classmethod
    def autoSipEnabled(cls) -> bool: ...
    @classmethod
    def beep(cls) -> None: ...
    @classmethod
    def closeAllWindows(cls) -> None: ...
    @classmethod
    def cursorFlashTime(cls) -> int: ...
    @classmethod
    def doubleClickInterval(cls) -> int: ...
    @classmethod
    def event(cls, event: QEvent | None) -> bool: ...
    @classmethod
    def exec(cls) -> int: ...
    @classmethod
    def focusWidget(cls) -> QWidget | None: ...
    @overload  # Add overloads
    @classmethod
    def font(cls) -> QFont: ...
    @overload
    @classmethod
    def font(cls, arg__1: QWidget | None) -> QFont: ...
    @overload
    @classmethod
    def font(cls, className: str | None) -> QFont: ...
    @classmethod
    def isEffectEnabled(cls, type: Qt.UIEffect) -> bool: ...
    @classmethod
    def keyboardInputInterval(cls) -> int: ...
    @classmethod
    def notify(
        cls, receiver: QObject | None, event: QEvent | None
    ) -> bool: ...  # Modified
    @overload  # Add overloads
    @classmethod
    def palette(cls) -> QPalette: ...
    @overload
    @classmethod
    def palette(cls, arg__1: QWidget | None) -> QPalette: ...
    @overload
    @classmethod
    def palette(cls, className: str | None) -> QPalette: ...
    @classmethod
    def setActiveWindow(cls, act: QWidget | None) -> None: ...
    @classmethod
    def setAutoSipEnabled(cls, enabled: bool) -> None: ...
    @classmethod
    def setCursorFlashTime(cls, ms: int) -> None: ...
    @classmethod
    def setDoubleClickInterval(cls, ms: int) -> None: ...
    @classmethod
    def setEffectEnabled(cls, type: Qt.UIEffect, enable: bool = True) -> None: ...
    @classmethod
    def setFont(cls, font: QFont, className: str | None = None) -> None: ...
    @classmethod
    def setKeyboardInputInterval(cls, ms: int) -> None: ...
    @classmethod
    def setPalette(cls, pal: QPalette, className: str | None = None) -> None: ...
    @classmethod
    def setStartDragDistance(cls, l: int) -> None: ...
    @classmethod
    def setStartDragTime(cls, ms: int) -> None: ...
    @overload
    @classmethod
    def setStyle(cls, style: QStyle | None) -> None: ...
    @overload
    @classmethod
    def setStyle(cls, style: str | None) -> QStyle | None: ...
    @classmethod
    def setStyleSheet(cls, sheet: str | None) -> None: ...
    @classmethod
    def setWheelScrollLines(cls, lines: int) -> None: ...
    @classmethod
    def startDragDistance(cls) -> int: ...
    @classmethod
    def startDragTime(cls) -> int: ...
    @classmethod
    def style(cls) -> QStyle | None: ...
    @classmethod
    def styleSheet(cls) -> str: ...
    @overload
    @classmethod
    def topLevelAt(cls, p: QPoint) -> QWidget | None: ...  # Overload for QApplication
    @overload
    @classmethod
    def topLevelAt(
        cls, x: int, y: int
    ) -> QWidget | None: ...  # Overload for QApplication
    @overload  # Added to address the error
    @classmethod
    def topLevelAt(
        cls, pos: QPoint
    ) -> QWindow | None: ...  # Overload for QGuiApplication
    @classmethod  # Make it a class method
    def topLevelWidgets(cls) -> list[QWidget]: ...
    @classmethod
    def wheelScrollLines(cls) -> int: ...
    @overload
    @classmethod
    def widgetAt(cls, p: QPoint) -> QWidget | None: ...  # Modified
    @overload
    @classmethod
    def widgetAt(cls, x: int, y: int) -> QWidget | None: ...

class ALMASimulator(QMainWindow, metaclass=_PyQtWrapperType):
    # ... (attributes)

    def __init__(self) -> None: ...
    @classmethod  # Make it a class method
    def event(cls, event: QEvent | None) -> bool: ...
    @classmethod
    def setStyle(
        cls, style: QStyle | None
    ) -> None: ...  # Corrected signature for ALMASimulator
    @classmethod
    def style(cls) -> QStyle | None: ...
    settings_file: Incomplete
    ncpu_entry: Incomplete
    terminal: Incomplete
    update_progress: Incomplete
    settings: Incomplete
    on_remote: bool
    settings_path: Incomplete
    metadata_path_label: Incomplete
    metadata_path_entry: Incomplete
    metadata_path_button: Incomplete
    metadata_path_row: Incomplete
    start_button: Incomplete
    reset_button: Incomplete
    term: Incomplete
    progress_bar_entry: Incomplete
    progress_bar: Incomplete
    left_layout: Incomplete
    line_displayed: bool
    def initialize_ui(self) -> None: ...
    def set_window_size(self) -> None: ...
    def has_widget(self, layout, widget_type): ...
    output_label: Incomplete
    output_entry: Incomplete
    output_button: Incomplete
    tng_label: Incomplete
    tng_entry: Incomplete
    tng_button: Incomplete
    galaxy_zoo_label: Incomplete
    galaxy_zoo_entry: Incomplete
    galaxy_zoo_button: Incomplete
    project_name_label: Incomplete
    project_name_entry: Incomplete
    n_sims_label: Incomplete
    n_sims_entry: Incomplete
    ncpu_label: Incomplete
    save_format_label: Incomplete
    save_format_combo: Incomplete
    comp_mode_label: Incomplete
    comp_mode_combo: Incomplete
    local_mode_label: Incomplete
    local_mode_combo: Incomplete
    remote_mode_label: Incomplete
    remote_mode_combo: Incomplete
    remote_folder_checkbox: Incomplete
    remote_dir_line: Incomplete
    remote_address_label: Incomplete
    remote_address_entry: Incomplete
    remote_config_label: Incomplete
    remote_config_entry: Incomplete
    remote_config_button: Incomplete
    remote_user_label: Incomplete
    remote_user_entry: Incomplete
    remote_key_label: Incomplete
    remote_key_entry: Incomplete
    key_button: Incomplete
    remote_key_pass_label: Incomplete
    remote_key_pass_entry: Incomplete
    remote_address_row: Incomplete
    remote_info_row: Incomplete
    def add_folder_widgets(self) -> None: ...
    def toggle_config_label(self) -> None: ...
    def toggle_remote_row(self) -> None: ...
    def toggle_remote_dir_line(self) -> None: ...
    def toggle_comp_mode(self) -> None: ...
    line_mode_checkbox: Incomplete
    line_index_label: Incomplete
    line_index_entry: Incomplete
    line_mode_row: Incomplete
    redshift_entry: Incomplete
    num_lines_entry: Incomplete
    non_line_mode_row1: Incomplete
    non_line_mode_row2: Incomplete
    def add_line_widgets(self) -> None: ...
    def toggle_line_mode_widgets(self) -> None: ...
    snr_checkbox: Incomplete
    snr_entry: Incomplete
    ir_luminosity_checkbox: Incomplete
    ir_luminosity_entry: Incomplete
    fix_spatial_checkbox: Incomplete
    n_pix_entry: Incomplete
    fix_spectral_checkbox: Incomplete
    n_channels_entry: Incomplete
    serendipitous_checkbox: Incomplete
    def add_dim_widgets(self): ...
    model_label: Incomplete
    model_combo: Incomplete
    model_row: Incomplete
    tng_api_key_label: Incomplete
    tng_api_key_entry: Incomplete
    tng_api_key_row: Incomplete
    def add_model_widgets(self) -> None: ...
    def toggle_tng_api_key_row(self) -> None: ...
    def toggle_dim_widgets_visibility(self, widget) -> None: ...
    metadata_mode_label: Incomplete
    metadata_mode_combo: Incomplete
    metadata_mode_row: Incomplete
    def add_meta_widgets(self) -> None: ...
    def add_metadata_widgets(self) -> None: ...
    query_type_label: Incomplete
    query_type_combo: Incomplete
    query_type_row: Incomplete
    query_save_label: Incomplete
    query_save_entry: Incomplete
    query_save_button: Incomplete
    query_execute_button: Incomplete
    query_save_row: Incomplete
    target_list_label: Incomplete
    target_list_entry: Incomplete
    target_list_button: Incomplete
    target_list_row: Incomplete
    def add_query_widgets(self) -> None: ...
    metadata_query_widgets_added: bool
    def remove_metadata_query_widgets(self) -> None: ...
    metadata: Incomplete
    def remove_metadata_browse(self) -> None: ...
    def toggle_metadata_browse(self, mode) -> None: ...
    def remove_query_widgets(self) -> None: ...
    def reset_fields(self) -> None: ...
    def load_settings(self) -> None: ...
    def load_settings_on_remote(self) -> None: ...
    @classmethod
    def populate_class_variables(cls, terminal, ncpu_entry) -> None: ...
    def closeEvent(self, event) -> None: ...
    def show_hide_widgets(self, layout, show: bool = True) -> None: ...
    science_keyword_entry: Incomplete
    scientific_category_entry: Incomplete
    band_entry: Incomplete
    fov_entry: Incomplete
    time_resolution_entry: Incomplete
    frequency_entry: Incomplete
    continue_query_button: Incomplete
    science_keyword_row: Incomplete
    scientific_category_row: Incomplete
    band_row: Incomplete
    fov_row: Incomplete
    time_resolution_row: Incomplete
    frequency_row: Incomplete
    continue_query_row: Incomplete
    def add_metadata_query_widgets(self) -> None: ...
    def browse_output_directory(self) -> None: ...
    def map_to_remote_directory(self, directory): ...
    def browse_tng_directory(self) -> None: ...
    def browse_galaxy_zoo_directory(self) -> None: ...
    def browse_metadata_path(self) -> None: ...
    def browse_ssh_key(self) -> None: ...
    def browse_slurm_config(self) -> None: ...
    def select_metadata_path(self) -> None: ...
    def metadata_path_set(self) -> None: ...
    def browse_target_list(self) -> None: ...
    def get_tap_service(self): ...
    def plot_science_keywords_distributions(master_path): ...
    def update_query_save_label(self, query_type) -> None: ...
    target_list: Incomplete
    def execute_query(self) -> None: ...
    def show_scientific_keywords(self) -> None: ...
    def query_for_metadata_by_science_type(self): ...
    def query_for_metadata_by_targets(self): ...
    def load_metadata(self, metadata_path) -> None: ...
    def line_display(self) -> None: ...
    def download_galaxy_zoo(self) -> None: ...
    def download_galaxy_zoo_on_remote(self) -> None: ...
    def check_tng_dirs(self) -> None: ...
    remote_main_dir: Incomplete
    remote_venv_dir: Incomplete
    def create_remote_environment(self) -> None: ...
    def create_remote_output_dir(self) -> None: ...
    def remote_check_tng_dirs(self) -> None: ...
    def copy_metadata_on_remote(self) -> None: ...
    def copy_settings_on_remote(self) -> None: ...
    @staticmethod
    def nan_to_none(value): ...
    def run_on_pbs_cluster(self) -> None: ...
    def run_on_mpi_machine(self) -> None: ...
    def run_on_slurm_cluster(self) -> None: ...
    source_type: str
    def transform_source_type_label(self) -> None: ...
    def sample_given_redshift(
        self, metadata, n, rest_frequency, extended, zmax: Incomplete | None = None
    ): ...
    def remove_non_numeric(self, text): ...
    @staticmethod
    def closest_power_of_2(x): ...
    def freq_supp_extractor(self, freq_sup, obs_freq): ...
    output_path: Incomplete
    main_path: Incomplete
    input_params: Incomplete
    def start_simulation(self) -> None: ...
    def run_simulator_sequentially(self) -> None: ...
    def run_simulator_parallel_remote(self, input_params) -> None: ...
    def run_simulator_slurm_remote(self, input_params) -> None: ...
    def run_simulator_parallel(self) -> None: ...
    def initiate_parallel_simulation(self) -> None: ...
    @classmethod
    def initiate_parallel_simulation_remote(cls, window_instance) -> None: ...
    @classmethod
    def initialize_slurm_simulation_remote(cls, window_istance) -> None: ...
    def cont_finder(self, cont_frequencies, line_frequency): ...
    def normalize_sed(
        self,
        sed,
        lum_infrared,
        solid_angle,
        cont_sens,
        freq_min,
        freq_max,
        remote: bool = False,
    ): ...
    def sed_reading(
        self,
        type_,
        path,
        cont_sens,
        freq_min,
        freq_max,
        remote,
        lum_infrared: Incomplete | None = None,
        redshift: Incomplete | None = None,
    ): ...
    def process_spectral_data(
        self,
        type_,
        master_path,
        redshift,
        central_frequency,
        delta_freq,
        source_frequency,
        n_channels,
        lum_infrared,
        cont_sens,
        line_names: Incomplete | None = None,
        n_lines: Incomplete | None = None,
        remote: bool = False,
    ): ...
    def print_variable_info(self, args) -> None: ...
    def simulator(self, *args, **kwargs): ...
    def handle_progress(self, value) -> None: ...
    def update_progress_bar(self, value) -> None: ...
    modelCube: Incomplete
    dirtyCube: Incomplete
    visCube: Incomplete
    dirtyvisCube: Incomplete
    Npix: Incomplete
    Np4: Incomplete
    Nchan: Incomplete
    gamma: Incomplete
    currcmap: Incomplete
    Xaxmax: Incomplete
    lfac: Incomplete
    u: Incomplete
    v: Incomplete
    UVpixsize: Incomplete
    w_min: Incomplete
    w_max: Incomplete
    plot_dir: Incomplete
    idx: Incomplete
    wavelength: Incomplete
    totsampling: Incomplete
    beam: Incomplete
    fmtB: Incomplete
    curzoom: Incomplete
    Nphf: Incomplete
    Xmax: Incomplete
    antPos: Incomplete
    Nant: Incomplete
    def plot_simulation_results(self, simulation_results) -> None: ...
