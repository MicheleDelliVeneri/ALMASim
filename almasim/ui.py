import numpy as np
import pandas as pd
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QGridLayout,
    QTextEdit,
    QSizePolicy,
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QComboBox,
    QProgressBar,
    QSlider,
    QSystemTrayIcon,
    QMenu,
)
from PyQt6.QtCore import (
    QSettings,
    pyqtSignal,
    Qt,
    QObject,
    QRunnable,
    QThreadPool,
    pyqtSlot,
)
from PyQt6.QtGui import QPixmap, QGuiApplication, QIcon
from kaggle import api
from os.path import isfile
import dask
import dask.config
import dask.dataframe as dd
from distributed import Client, LocalCluster, WorkerPlugin
from dask_jobqueue import SLURMCluster
from concurrent.futures import ThreadPoolExecutor
import json
import astropy.units as U
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM
import math
from math import pi, ceil
import time
import shutil
from time import strftime, gmtime
import paramiko
import pysftp
import plistlib
import psutil
import almasim.alma as ual
import almasim.astro as uas
import almasim.skymodels as usm
import almasim.interferometer as uin
import threading
import matplotlib
import matplotlib.pyplot as plt
import logging
import pyvo
import re
import seaborn as sns
import subprocess
from pathlib import Path
import inspect
import requests
import zipfile

# import yagmail

matplotlib.use("Agg")
os.environ["LC_ALL"] = "C"


class MemoryLimitPlugin(WorkerPlugin):
    def __init__(self, memory_limit):
        self.memory_limit = memory_limit

    def setup(self, worker):
        pass

    def teardown(self, worker):
        pass

    def transition(self, key, start, finish, *args, **kwargs):
        if finish == "memory" and psutil.virtual_memory().percent > self.memory_limit:
            # If memory usage exceeds the limit, skip the task
            return "erred"


class TerminalLogger(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self, terminal):
        super().__init__()
        self.terminal = terminal
        self.log_signal.connect(self.terminal.append)

    def add_log(self, message):
        message = message.replace("\r", "").replace("\n", "")
        self.log_signal.emit(message)

    @pyqtSlot(int)
    def update_progress(self, value):
        # Update terminal or GUI with progress value
        # print(f"Progress: {value}%")
        self.add_log(f"Progress: {value}%")


class PlotWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.initial_width = 2400  # 20 inches * 60 pixels per inch
        self.initial_height = 1200  # 20 inches * 60 pixels per inch
        self.scroll_area = QScrollArea()
        self.scroll_area.setFixedWidth(self.initial_width)
        self.scroll_area.setFixedHeight(self.initial_height)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)
        # Create a central widget for the scroll area
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)

        # Set size policies for the layout to expand
        self.scroll_layout.setRowStretch(0, 1)
        self.scroll_layout.setRowStretch(1, 1)
        self.scroll_layout.setColumnStretch(0, 1)
        self.scroll_layout.setColumnStretch(1, 1)

        self.scroll_area.setWidget(self.scroll_widget)
        self.resize(self.initial_width, self.initial_height)  # Set initial size
        self.create_science_keyword_plots()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_plot_sizes()

    def update_plot_sizes(self):
        available_width = self.scroll_widget.width()
        available_height = self.scroll_widget.height()
        plot_width = available_width // 2  # Keep integer division for layout
        plot_height = available_height // 3

        for i in range(self.scroll_layout.count()):
            item = self.scroll_layout.itemAt(i)
            if isinstance(item.widget(), QLabel):
                label = item.widget()
                pixmap = label.pixmap()
                # Calculate the aspect ratio of the original pixmap
                aspect_ratio = pixmap.width() / pixmap.height()
                # Determine if the available width or height is the limiting factor
                if plot_width / aspect_ratio <= plot_height:
                    new_height = int(plot_width / aspect_ratio)  # Convert to integer
                    new_width = plot_width
                else:
                    new_width = int(plot_height * aspect_ratio)  # Convert to integer
                    new_height = plot_height
                scaled_pixmap = pixmap.scaled(
                    new_width, new_height, Qt.AspectRatioMode.KeepAspectRatio
                )  # Scale with aspect ratio
                label.setPixmap(scaled_pixmap)

    def create_science_keyword_plots(self):
        """Creates and displays plots of science keyword distributions in the window."""
        try:
            plot_dir = os.path.join(os.getcwd(), "plots")  # Get the directory for plots
            expected_plots = [
                "science_vs_bands.png",
                "science_vs_int_time.png",
                "science_vs_FoV.png",
                "science_vs_source_freq.png",
            ]

            # Check if plots need to be generated
            if not all(
                os.path.exists(os.path.join(plot_dir, plot)) for plot in expected_plots
            ):
                self.plot_science_keywords_distributions(
                    os.getcwd()
                )  # Generate plots if not found

            row, col = 0, 0
            for plot_file in expected_plots:  # Iterate through the expected plot files
                plot_path = os.path.join(plot_dir, plot_file)

                pixmap = QPixmap()
                if not pixmap.load(plot_path):  # Load the image for the plot
                    self.terminal.add_log(f"Error loading plot: {plot_path}")
                    continue

                max_width = self.width() // 2
                max_height = self.height() // 2

                scaled_pixmap = pixmap.scaled(
                    max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio
                )  # Scale the image while maintaining aspect ratio

                label = QLabel()
                label.setPixmap(scaled_pixmap)
                label.setSizePolicy(
                    QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
                )  # Allow resizing
                self.scroll_layout.addWidget(label, row, col)
                self.scroll_layout.setRowStretch(row, 1)
                self.scroll_layout.setColumnStretch(col, 1)

                col += 1
                if col == 2:  # 2 columns per row
                    col = 0
                    row += 1

            self.adjustSize()

        except Exception as e:  # Catch any potential exceptions
            self.terminal.add_log(
                f"Error in create_science_keyword_plots: {e}"
            )  # Log the error


class SignalEmitter(QObject):
    simulationFinished = pyqtSignal(object)
    progress = pyqtSignal(int)


class SimulatorRunnable(QRunnable, QObject):
    def __init__(self, alma_simulator_instance, *args, **kwargs):
        super().__init__()
        QObject.__init__(self)  # Initialize QObject
        self.alma_simulator = alma_simulator_instance
        self.args = args
        self.kwargs = kwargs
        self.signals = SignalEmitter()  # Create an instance of the SignalEmitter

    @pyqtSlot()
    def run(self):
        try:
            self.finish_flag = False
            results = None
            while not self.alma_simulator.stop_simulation_flag:
                if not self.finish_flag:
                    results = self.alma_simulator.simulator(*self.args, **self.kwargs)
                    self.finish_flag = True
                    self.signals.simulationFinished.emit(
                        results
                    )  # Emit the results when done

        except Exception as e:
            logging.error(f"Error in SimulatorRunnable: {e}", exc_info=True)


class ParallelSimulatorRunnable(QRunnable):
    def __init__(self, alma_simulator_instance):
        super().__init__()
        self.alma_simulator = alma_simulator_instance

    @pyqtSlot()
    def run(self):
        self.alma_simulator.run_simulator_parallel()


class ParallelSimulatorRunnableRemote(QRunnable):
    def __init__(self, alma_simulator_instance, input_params):
        super().__init__()
        self.alma_simulator_instance = alma_simulator_instance
        self.input_params = input_params

    @pyqtSlot()
    def run(self):
        self.alma_simulator_instance.run_simulator_parallel_remote(self.input_params)


class SlurmSimulatorRunnableRemote(QRunnable):
    def __init__(self, alma_simulator_instance, input_params):
        super().__init__()
        self.alma_simulator_instance = alma_simulator_instance
        self.input_params = input_params

    @pyqtSlot()
    def run(self):
        self.alma_simulator_instance.run_simulator_slurm_remote(self.input_params)


class SimulatorWorker(QRunnable, QObject):
    def __init__(self, alma_simulator_instance, df, *args, **kwargs):
        super().__init__()
        QObject.__init__(self)
        self.alma_simulator = alma_simulator_instance
        self.df = df
        self.signals = SignalEmitter()

    @pyqtSlot()
    def run(self):
        for i, row in self.df.iterrows():
            row = row.where(~row.isna(), None)
            results = self.alma_simulator.simulator(*row)
            self.signals.simulationFinished.emit(results)


class DownloadGalaxyZooRunnable(QRunnable):
    """Runnable for downloading Galaxy Zoo data in a separate thread."""

    def __init__(self, alma_simulator_instance):
        super().__init__()
        self.alma_simulator = (
            alma_simulator_instance  # Store a reference to the main UI class
        )

    def run(self):
        """Downloads Galaxy Zoo data."""
        self.alma_simulator.download_galaxy_zoo()


class DownloadTNGStructureRunnable(QRunnable):
    """Runnable for downloading TNG Folders in a separate thread."""

    def __init__(self, alma_simulator_instance):
        super().__init__()
        self.alma_simulator = (
            alma_simulator_instance  # Store a reference to the main UI class
        )

    def run(self):
        self.alma_simulator.check_tng_dirs()


class DownloadHubbleRunnable(QRunnable):
    """Runnable for downloading Hubble 100 data in a separate thread."""

    def __init__(self, alma_simulator_instance):
        super().__init__()
        self.alma_simulator = (
            alma_simulator_instance  # Store a reference to the main UI class
        )

    def run(self):
        """Downloads Galaxy Zoo data."""
        self.alma_simulator.download_hubble()


class PlotResultsRunnable(QRunnable):
    def __init__(self, alma_simulator_instance, simulation_results):
        super().__init__()
        self.alma_simulator = (
            alma_simulator_instance  # Store a reference to the main UI class
        )
        self.simulation_results = simulation_results

    def run(self):
        """Downloads Galaxy Zoo data."""
        self.alma_simulator.plot_simulation_results(self.simulation_results)


class ALMASimulator(QMainWindow):
    settings_file = None
    ncpu_entry = None
    terminal = None
    thread_pool = None
    update_progress = pyqtSignal(int)
    nextSimulation = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.settings = QSettings("INFN Section of Naples", "ALMASim")
        self.tray_icon = None
        self.thread_pool = QThreadPool.globalInstance()
        self.main_path = Path(inspect.getfile(inspect.currentframe())).resolve().parent
        path = os.path.dirname(self.main_path)
        icon_path = os.path.join(path, "pictures", "almasim-icon.png")
        icon = QIcon(icon_path)
        self.setWindowIcon(icon)

        if ALMASimulator.settings_file is not None:
            with open(ALMASimulator.settings_file, "rb") as f:
                settings_data = plistlib.load(f)
                for key, value in settings_data.items():
                    self.settings.setValue(key, value)
            self.on_remote = True
        else:
            self.on_remote = False

        self.settings_path = self.settings.fileName()
        print(self.settings_path)
        self.initialize_ui()
        self.stop_simulation_flag = False
        self.remote_simulation_finished = True
        self.terminal.add_log("Setting file path is {}".format(self.settings_path))

    # -------- Widgets and UI -------------------------
    def initialize_ui(self):
        self.setWindowTitle("ALMASim: set up your simulation parameters")
        # --- Create Widgets ---
        # self.main_path = os.path.sep + os.path.join(
        #    *str(Path(inspect.getfile(inspect.currentframe())).resolve()).split(
        #        os.path.sep
        #    )[:-1]
        # )
        self.metadata_path_label = QLabel("Metadata Path:")
        self.metadata_path_entry = QLineEdit()
        self.metadata_path_button = QPushButton("Browse")
        self.metadata_path_button.clicked.connect(self.browse_metadata_path)
        self.metadata_path_row = QHBoxLayout()
        self.metadata_path_row.addWidget(self.metadata_path_label)
        self.metadata_path_row.addWidget(self.metadata_path_entry)
        self.metadata_path_row.addWidget(self.metadata_path_button)

        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_fields)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)

        self.term = QTextEdit(self)
        self.term.setReadOnly(True)
        self.terminal = TerminalLogger(self.term)

        self.progress_bar_entry = QLabel()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        # --- Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Button Row
        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        self.left_layout.addStretch(1)
        self.left_layout.addLayout(button_row)

        right_layout.addWidget(self.term)
        right_layout.addWidget(self.progress_bar_entry)
        right_layout.addWidget(self.progress_bar)

        main_layout.addLayout(self.left_layout)
        main_layout.addLayout(right_layout)
        main_layout.setStretch(0, 3)  # left_layout stretch factor
        main_layout.setStretch(1, 2)  # right_layout stretch factor

        self.line_displayed = False
        self.add_folder_widgets()
        self.add_line_widgets()
        self.add_width_slider()
        self.add_robust_slider()
        self.add_dim_widgets()
        self.add_model_widgets()
        self.add_meta_widgets()
        self.add_query_widgets()
        ALMASimulator.populate_class_variables(
            self.terminal, self.ncpu_entry, self.thread_pool
        )
        # Load saved settings
        if self.on_remote is True:
            self.load_settings_on_remote()
        else:
            self.load_settings()
        # self.terminal.start_log("")
        # Check metadata mode on initialization
        self.toggle_line_mode_widgets()
        self.metadata_mode_combo.currentTextChanged.connect(self.toggle_metadata_browse)
        if self.metadata_path_entry.text() != "" and isfile(
            self.metadata_path_entry.text()
        ):
            self.load_metadata(self.metadata_path_entry.text())
        current_mode = self.metadata_mode_combo.currentText()
        self.toggle_metadata_browse(current_mode)  # Call here
        self.set_window_size()
        ALMASimulator.populate_class_variables(
            self.terminal, self.ncpu_entry, self.thread_pool
        )

    def set_window_size(self):
        screen = QGuiApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()
        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.5)
        self.setGeometry(
            (screen_width - window_width) // 2,
            (screen_height - window_height) // 2,
            window_width,
            window_height,
        )

    def has_widget(self, layout, widget_type):
        """Check if the layout contains a widget of a specific type."""
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(item.widget(), widget_type):
                return True
        return False

    def add_folder_widgets(self):
        # 1
        self.output_label = QLabel("Output Directory:")
        self.output_entry = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.browse_output_directory)
        # 2
        self.tng_label = QLabel("TNG Directory:")
        self.tng_entry = QLineEdit()
        self.tng_button = QPushButton("Browse")
        self.tng_button.clicked.connect(self.browse_tng_directory)
        # 3
        self.galaxy_zoo_label = QLabel("Galaxy Zoo Directory:")
        self.galaxy_zoo_entry = QLineEdit()
        self.galaxy_zoo_button = QPushButton("Browse")
        self.galaxy_zoo_button.clicked.connect(self.browse_galaxy_zoo_directory)
        # 4
        self.hubble_label = QLabel("Hubble Top 100 Directory:")
        self.hubble_entry = QLineEdit()
        self.hubble_button = QPushButton("Browse")
        self.hubble_button.clicked.connect(self.browse_hubble_directory)
        # 5
        self.mail_label = QLabel("Email:")
        self.mail_entry = QLineEdit()
        # 6
        self.project_name_label = QLabel("Project Name:")
        self.project_name_entry = QLineEdit()
        # 7
        self.n_sims_label = QLabel("Number of Simulations:")
        self.n_sims_entry = QLineEdit()
        # 8
        self.ncpu_label = QLabel("N. CPUs / Processes:")
        self.ncpu_entry = QLineEdit()
        # 9
        self.save_format_label = QLabel("Save Format:")
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["npz", "fits", "h5"])
        # 10
        self.comp_mode_label = QLabel("Computation Mode:")
        self.comp_mode_combo = QComboBox()
        self.comp_mode_combo.addItems(["sequential", "parallel"])
        # 11
        self.local_mode_label = QLabel("Local or Remote:")
        self.local_mode_combo = QComboBox()
        self.local_mode_combo.addItems(["local", "remote"])
        self.local_mode_combo.currentTextChanged.connect(self.toggle_comp_mode)
        # 12
        self.remote_mode_label = QLabel("Mode:")
        self.remote_mode_combo = QComboBox()
        self.remote_mode_combo.addItems(["MPI", "SLURM", "PBS"])
        self.remote_folder_checkbox = QCheckBox("Set Work Directory:")
        self.remote_dir_line = QLineEdit()
        self.remote_folder_checkbox.stateChanged.connect(self.toggle_remote_dir_line)

        self.remote_address_label = QLabel("Remote Host:")
        self.remote_address_entry = QLineEdit()
        self.remote_config_label = QLabel("Slurm Config:")
        self.remote_config_entry = QLineEdit()
        self.remote_config_button = QPushButton("Browse", self)
        self.remote_config_button.clicked.connect(self.browse_slurm_config)
        self.remote_user_label = QLabel("Username")
        self.remote_user_entry = QLineEdit()
        self.remote_key_label = QLabel("SSH Key:")
        self.remote_key_entry = QLineEdit()
        self.key_button = QPushButton("Browse", self)
        self.key_button.clicked.connect(self.browse_ssh_key)
        self.remote_key_pass_label = QLabel("Key Passwd:")
        self.remote_key_pass_entry = QLineEdit()
        self.remote_key_pass_entry.setEchoMode(QLineEdit.EchoMode.Password)

        # 10
        # self.flux_mode_label = QLabel('Flux Simulation Mode:')
        # self.flux_mode_combo = QComboBox()
        # self.flux_mode_combo.addItems(["direct", 'line-ratios'])

        # Output Directory Row
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_label)
        output_row.addWidget(self.output_entry)
        output_row.addWidget(self.output_button)
        self.left_layout.insertLayout(1, output_row)

        # TNG Directory Row
        tng_row = QHBoxLayout()
        tng_row.addWidget(self.tng_label)
        tng_row.addWidget(self.tng_entry)
        tng_row.addWidget(self.tng_button)
        self.left_layout.insertLayout(2, tng_row)

        # Galaxy Zoo Directory Row
        galaxy_row = QHBoxLayout()
        self.galaxy_zoo_checkbox = QCheckBox("Get Galaxy Zoo Data")
        galaxy_row.addWidget(self.galaxy_zoo_label)
        galaxy_row.addWidget(self.galaxy_zoo_entry)
        galaxy_row.addWidget(self.galaxy_zoo_button)
        galaxy_row.addWidget(self.galaxy_zoo_checkbox)
        self.left_layout.insertLayout(3, galaxy_row)

        # Hubble Directory Row
        hubble_row = QHBoxLayout()
        self.hubble_checkbox = QCheckBox("Get Hubble 100 Data")
        hubble_row.addWidget(self.hubble_label)
        hubble_row.addWidget(self.hubble_entry)
        hubble_row.addWidget(self.hubble_button)
        hubble_row.addWidget(self.hubble_checkbox)
        self.left_layout.insertLayout(4, hubble_row)

        # User Email Row
        mail_row = QHBoxLayout()
        mail_row.addWidget(self.mail_label)
        mail_row.addWidget(self.mail_entry)
        self.left_layout.insertLayout(5, mail_row)

        # Project Name Row
        project_name_row = QHBoxLayout()
        project_name_row.addWidget(self.project_name_label)
        project_name_row.addWidget(self.project_name_entry)
        self.left_layout.insertLayout(6, project_name_row)

        # Number of Simulations Row
        n_sims_row = QHBoxLayout()
        n_sims_row.addWidget(self.n_sims_label)
        n_sims_row.addWidget(self.n_sims_entry)
        self.left_layout.insertLayout(7, n_sims_row)

        # Number of CPUs Row
        ncpu_row = QHBoxLayout()
        ncpu_row.addWidget(self.ncpu_label)
        ncpu_row.addWidget(self.ncpu_entry)
        self.left_layout.insertLayout(8, ncpu_row)

        # Save format Row
        save_format_row = QHBoxLayout()
        save_format_row.addWidget(self.save_format_label)
        save_format_row.addWidget(self.save_format_combo)
        self.left_layout.insertLayout(9, save_format_row)

        # Computation Mode Row
        comp_mode_row = QHBoxLayout()
        comp_mode_row.addWidget(self.comp_mode_label)
        comp_mode_row.addWidget(self.comp_mode_combo)
        self.left_layout.insertLayout(10, comp_mode_row)

        # Local Mode Row
        local_mode_row = QHBoxLayout()
        local_mode_row.addWidget(self.local_mode_label)
        local_mode_row.addWidget(self.local_mode_combo)
        local_mode_row.addWidget(self.remote_mode_label)
        local_mode_row.addWidget(self.remote_mode_combo)
        local_mode_row.addWidget(self.remote_folder_checkbox)
        local_mode_row.addWidget(self.remote_dir_line)
        self.left_layout.insertLayout(11, local_mode_row)
        self.remote_mode_label.hide()
        self.remote_mode_combo.hide()
        self.remote_folder_checkbox.hide()
        self.remote_dir_line.hide()
        self.remote_mode_combo.currentTextChanged.connect(self.toggle_remote_row)

        self.remote_address_row = QHBoxLayout()
        self.remote_address_row.addWidget(self.remote_address_label)
        self.remote_address_row.addWidget(self.remote_address_entry)
        self.remote_address_row.addWidget(self.remote_config_label)
        self.remote_address_row.addWidget(self.remote_config_entry)
        self.remote_address_row.addWidget(self.remote_config_button)
        self.left_layout.insertLayout(12, self.remote_address_row)
        self.show_hide_widgets(self.remote_address_row, show=False)

        self.remote_info_row = QHBoxLayout()
        self.remote_info_row.addWidget(self.remote_user_label)
        self.remote_info_row.addWidget(self.remote_user_entry)
        self.remote_info_row.addWidget(self.remote_key_label)
        self.remote_info_row.addWidget(self.remote_key_entry)
        self.remote_info_row.addWidget(self.key_button)
        self.remote_info_row.addWidget(self.remote_key_pass_label)
        self.remote_info_row.addWidget(self.remote_key_pass_entry)
        self.left_layout.insertLayout(13, self.remote_info_row)
        self.show_hide_widgets(self.remote_info_row, show=False)
        self.local_mode_combo.currentTextChanged.connect(self.toggle_remote_row)

    def toggle_config_label(self):
        if self.remote_mode_combo.currentText() == "SLURM":
            self.remote_config_label.setText("Slurm Config:")
        elif self.remote_mode_combo.currentText() == "PBS":
            self.remote_config_label.setText("PBS Config:")
        else:
            self.remote_config_label.setText("MPI Config")

    def toggle_remote_row(self):
        if self.local_mode_combo.currentText() == "remote":
            self.show_hide_widgets(self.remote_address_row, show=True)
            self.show_hide_widgets(self.remote_info_row, show=True)
            self.toggle_config_label()
            self.remote_mode_label.show()
            self.remote_mode_combo.show()
            self.remote_folder_checkbox.show()
            if self.output_entry.text() != "" and self.remote_user_entry.text() != "":
                self.output_entry.setText(
                    self.map_to_remote_directory(self.output_entry.text())
                )
            if self.tng_entry.text() != "" and self.remote_user_entry.text() != "":
                self.tng_entry.setText(
                    self.map_to_remote_directory(self.tng_entry.text())
                )
            if (
                self.galaxy_zoo_entry.text() != ""
                and self.remote_user_entry.text() != ""
            ):
                self.galaxy_zoo_entry.setText(
                    self.map_to_remote_directory(self.galaxy_zoo_entry.text())
                )
        else:
            self.show_hide_widgets(self.remote_address_row, show=False)
            self.show_hide_widgets(self.remote_info_row, show=False)
            self.remote_mode_label.hide()
            self.remote_mode_combo.hide()
            self.remote_folder_checkbox.hide()
            self.remote_dir_line.hide()
            if self.output_entry.text() != "":
                folder = self.output_entry.text().split(os.path.sep)[-1]
                self.output_entry.setText(os.path.join(os.path.expanduser("~"), folder))
            if self.tng_entry.text() != "":
                folder = self.tng_entry.text().split(os.path.sep)[-1]
                self.tng_entry.setText(os.path.join(os.path.expanduser("~"), folder))
            if self.galaxy_zoo_entry.text() != "":
                folder = self.galaxy_zoo_entry.text().split(os.path.sep)[-1]
                self.galaxy_zoo_entry.setText(
                    os.path.join(os.path.expanduser("~"), folder)
                )

    def toggle_remote_dir_line(self):
        if self.remote_folder_checkbox.isChecked():
            self.remote_dir_line.show()
        else:
            self.remote_dir_line.hide()

    def toggle_comp_mode(self):
        if self.local_mode_combo.currentText() == "remote":
            self.comp_mode_combo.clear()
            self.comp_mode_combo.addItems(["parallel"])
        else:
            self.comp_mode_combo.clear()
            self.comp_mode_combo.addItems(["sequential", "parallel"])

    def add_line_widgets(self):
        self.line_mode_checkbox = QCheckBox("Line Mode")
        self.line_mode_checkbox.stateChanged.connect(self.toggle_line_mode_widgets)
        # self.left_layout.insertWidget(8, self.line_mode_checkbox)
        # Widgets for Line Mode
        self.line_index_label = QLabel("Select Line Indices (space-separated):")
        self.line_index_entry = QLineEdit()
        self.line_mode_row = QHBoxLayout()
        self.line_mode_row.addWidget(self.line_mode_checkbox)
        self.left_layout.insertLayout(14, self.line_mode_row)  # Insert at the end
        # Widgets for Non-Line Mode
        redshift_label = QLabel("Redshifts (space-separated):")
        self.redshift_entry = QLineEdit()
        num_lines_label = QLabel("Number of Lines to Simulate:")
        self.num_lines_entry = QLineEdit()
        self.non_line_mode_row1 = QHBoxLayout()
        self.non_line_mode_row1.addWidget(redshift_label)
        self.non_line_mode_row1.addWidget(self.redshift_entry)
        self.non_line_mode_row2 = QHBoxLayout()
        self.non_line_mode_row2.addWidget(num_lines_label)
        self.non_line_mode_row2.addWidget(self.num_lines_entry)
        self.left_layout.insertLayout(15, self.non_line_mode_row1)  # Insert at the end
        self.left_layout.insertLayout(16, self.non_line_mode_row2)  # Insert at the end
        self.show_hide_widgets(self.non_line_mode_row1, show=False)
        self.show_hide_widgets(self.non_line_mode_row2, show=False)

    def toggle_line_mode_widgets(self):
        """Shows/hides the appropriate input rows based on line mode checkbox state."""
        if self.line_mode_checkbox.isChecked():
            if not self.has_widget(self.line_mode_row, QLabel):
                self.line_mode_row.addWidget(self.line_index_label)
                self.line_mode_row.addWidget(self.line_index_entry)
            # Show the widgets in line_mode_row
            self.show_hide_widgets(self.line_mode_row, show=True)
            # Hide the widgets in non_line_mode_row1 and non_line_mode_row2
            self.show_hide_widgets(self.non_line_mode_row1, show=False)
            self.show_hide_widgets(self.non_line_mode_row2, show=False)
            if self.line_displayed is False:
                self.line_display()
        else:
            # Hide the widgets in line_mode_row
            self.show_hide_widgets(self.line_mode_row, show=False)
            if self.has_widget(self.line_mode_row, QLabel):
                self.line_mode_row.removeWidget(self.line_index_label)
                self.line_mode_row.removeWidget(self.line_index_entry)
            self.show_hide_widgets(self.line_mode_row, show=True)
            # Show the widgets in non_line_mode_row1 and non_line_mode_row2
            self.show_hide_widgets(self.non_line_mode_row1, show=True)
            self.show_hide_widgets(self.non_line_mode_row2, show=True)

    def add_width_slider(self):
        self.line_width_label = QLabel("Min/Max Line Widths in Km/s:")
        self.max_line_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_line_width_value_label = QLabel(
            f"{self.max_line_width_slider.value()} km/s"
        )
        self.max_line_width_slider.setRange(50, 1700)
        self.max_line_width_slider.setTickInterval(25)
        self.max_line_width_slider.setSingleStep(5)
        self.max_line_width_slider.setValue(400)
        self.max_line_width_slider.valueChanged.connect(
            self.update_max_line_width_label
        )
        self.min_line_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_line_width_value_label = QLabel(
            f"{self.min_line_width_slider.value()} km/s"
        )
        self.min_line_width_slider.setRange(50, 1700)
        self.min_line_width_slider.setTickInterval(25)
        self.min_line_width_slider.setSingleStep(5)
        self.min_line_width_slider.setValue(200)
        self.min_line_width_slider.valueChanged.connect(
            self.update_min_line_width_label
        )
        self.min_line_width_slider.valueChanged.connect(self.sync_line_width_sliders)
        self.max_line_width_slider.valueChanged.connect(self.sync_line_width_sliders)
        self.line_width_row = QHBoxLayout()
        self.line_width_row.addWidget(self.line_width_label)
        self.line_width_row.addWidget(self.min_line_width_slider)
        self.line_width_row.addWidget(self.min_line_width_value_label)
        self.line_width_row.addWidget(self.max_line_width_slider)
        self.line_width_row.addWidget(self.max_line_width_value_label)
        self.left_layout.insertLayout(17, self.line_width_row)

    def update_max_line_width_label(self, value):
        self.max_line_width_value_label.setText(f"{value} km/s")

    def update_min_line_width_label(self, value):
        self.min_line_width_value_label.setText(f"{value} km/s")

    def sync_line_width_sliders(self):
        min_value = self.min_line_width_slider.value()
        max_value = self.max_line_width_slider.value()
        if min_value > max_value:
            self.min_line_width_slider.setValue(max_value)
            self.min_line_width_value_label.setText(f"{max_value} km/s")
        elif max_value < min_value:
            self.max_line_width_slider.setValue(min_value)
            self.max_line_width_value_label.setText(f"{min_value} km/s")

    def add_robust_slider(self):
        self.robust_label = QLabel("Briggs Robustness:")
        self.robust_slider = QSlider(Qt.Orientation.Horizontal)
        self.robust_value_label = QLabel(f"{self.robust_slider.value()}")
        self.robust_slider.setRange(-20, 20)
        self.robust_slider.setTickInterval(1)
        self.robust_slider.setSingleStep(1)
        self.robust_slider.setValue(0)
        self.robust_slider.valueChanged.connect(self.update_robust_label)
        self.robust_row = QHBoxLayout()
        self.robust_row.addWidget(self.robust_label)
        self.robust_row.addWidget(self.robust_slider)
        self.robust_row.addWidget(self.robust_value_label)
        self.left_layout.insertLayout(18, self.robust_row)

    def update_robust_label(self, value):
        self.robust_value_label.setText(f"{value / 10}")

    def add_dim_widgets(self):
        # --- Set SNR ---
        self.snr_checkbox = QCheckBox("Set SNR")
        self.snr_entry = QLineEdit()
        self.snr_entry.setVisible(False)
        self.snr_checkbox.stateChanged.connect(
            lambda: self.toggle_dim_widgets_visibility(self.snr_entry)
        )

        # --- Set Infrared Luminosity ---
        self.ir_luminosity_checkbox = QCheckBox("Set IR Luminosity")
        self.ir_luminosity_entry = QLineEdit()
        self.ir_luminosity_entry.setVisible(False)
        self.ir_luminosity_checkbox.stateChanged.connect(
            lambda: self.toggle_dim_widgets_visibility(self.ir_luminosity_entry)
        )

        # --- Fix Spatial Dimension Checkbox and Field ---
        self.fix_spatial_checkbox = QCheckBox("Fix Spatial Dim")
        self.n_pix_entry = QLineEdit()
        self.n_pix_entry.setVisible(False)
        self.fix_spatial_checkbox.stateChanged.connect(
            lambda: self.toggle_dim_widgets_visibility(self.n_pix_entry)
        )

        # --- Fix Spectral Dimension Checkbox and Field ---
        self.fix_spectral_checkbox = QCheckBox("Fix Spectral Dim")
        self.n_channels_entry = QLineEdit()
        self.n_channels_entry.setVisible(False)
        self.fix_spectral_checkbox.stateChanged.connect(
            lambda: self.toggle_dim_widgets_visibility(self.n_channels_entry)
        )

        # --- Inject Serendipitous sources ----
        self.serendipitous_checkbox = QCheckBox("Inject Serendipitous")

        # --- Layout for Checkboxes and Fields ---
        checkbox_row = QHBoxLayout()
        checkbox_row.addWidget(self.snr_checkbox)
        checkbox_row.addWidget(self.snr_entry)
        checkbox_row.addWidget(self.ir_luminosity_checkbox)
        checkbox_row.addWidget(self.ir_luminosity_entry)
        checkbox_row.addWidget(self.fix_spatial_checkbox)
        checkbox_row.addWidget(self.n_pix_entry)
        checkbox_row.addWidget(self.fix_spectral_checkbox)
        checkbox_row.addWidget(self.n_channels_entry)
        checkbox_row.addWidget(self.serendipitous_checkbox)
        self.left_layout.insertLayout(19, checkbox_row)

    def add_model_widgets(self):
        self.model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            [
                "Point",
                "Gaussian",
                "Extended",
                "Diffuse",
                "Galaxy Zoo",
                "Hubble 100",
                "Molecular",
            ]
        )
        self.model_row = QHBoxLayout()
        self.model_row.addWidget(self.model_label)
        self.model_row.addWidget(self.model_combo)
        self.left_layout.insertLayout(20, self.model_row)
        self.tng_api_key_label = QLabel("TNG API Key:")
        self.tng_api_key_entry = QLineEdit()
        self.tng_api_key_row = QHBoxLayout()
        self.tng_api_key_row.addWidget(self.tng_api_key_label)
        self.tng_api_key_row.addWidget(self.tng_api_key_entry)

        # Initially hide the TNG API key row
        self.show_hide_widgets(self.tng_api_key_row, show=False)

        self.left_layout.insertLayout(
            21, self.tng_api_key_row
        )  # Insert after model_row
        # Connect the model_combo's signal to update visibility
        self.model_combo.currentTextChanged.connect(self.toggle_tng_api_key_row)

    def toggle_tng_api_key_row(self):
        """Shows/hides the TNG API key row based on the selected model."""
        if self.model_combo.currentText() == "Extended":
            self.show_hide_widgets(self.tng_api_key_row, show=True)
        else:
            self.show_hide_widgets(self.tng_api_key_row, show=False)

    def toggle_dim_widgets_visibility(self, widget):
        widget.setVisible(self.sender().isChecked())

    def add_meta_widgets(self):
        self.metadata_mode_label = QLabel("Metadata Retrieval Mode:")
        self.metadata_mode_combo = QComboBox()
        self.metadata_mode_combo.addItems(["query", "get"])
        self.metadata_mode_combo.currentTextChanged.connect(self.toggle_metadata_browse)
        # Metadata Retrieval Mode Row
        self.metadata_mode_row = QHBoxLayout()
        self.metadata_mode_row.addWidget(self.metadata_mode_label)
        self.metadata_mode_row.addWidget(self.metadata_mode_combo)
        self.left_layout.insertLayout(22, self.metadata_mode_row)

    def add_metadata_widgets(self):
        self.metadata_path_label = QLabel("Metadata Path:")
        self.metadata_path_entry = QLineEdit()
        self.metadata_path_button = QPushButton("Browse")
        self.metadata_path_button.clicked.connect(self.browse_metadata_path)
        self.metadata_path_row = QHBoxLayout()
        self.metadata_path_row.addWidget(self.metadata_path_label)
        self.metadata_path_row.addWidget(self.metadata_path_entry)
        self.metadata_path_row.addWidget(self.metadata_path_button)
        self.left_layout.insertLayout(23, self.metadata_path_row)
        self.left_layout.update()

    def add_query_widgets(self):
        # Create widgets for querying
        self.query_type_label = QLabel("Query Type:")
        self.query_type_combo = QComboBox()
        self.query_type_combo.addItems(["science", "target"])
        self.query_type_row = QHBoxLayout()
        self.query_type_row.addWidget(self.query_type_label)
        self.query_type_row.addWidget(self.query_type_combo)
        self.query_save_label = QLabel("Save Metadata to:")
        # Set the initial label text
        self.query_save_entry = QLineEdit()
        self.query_save_button = QPushButton("Browse")
        # Connect browse button to appropriate method (you'll need to implement this)
        self.query_save_button.clicked.connect(self.select_metadata_path)
        self.query_execute_button = QPushButton("Execute Query")
        self.query_execute_button.clicked.connect(self.execute_query)
        self.query_save_row = QHBoxLayout()
        self.query_save_row.addWidget(self.query_save_label)
        self.query_save_row.addWidget(self.query_save_entry)
        self.query_save_row.addWidget(self.query_save_button)
        self.target_list_label = QLabel("Load Target List:")
        self.target_list_entry = QLineEdit()
        self.target_list_button = QPushButton("Browse")
        self.target_list_button.clicked.connect(
            self.browse_target_list
        )  # Add function for browsing
        self.target_list_row = QHBoxLayout()
        self.target_list_row.addWidget(self.target_list_label)
        self.target_list_row.addWidget(self.target_list_entry)
        self.target_list_row.addWidget(self.target_list_button)
        # self.target_list_row.hide()  # Initially hide the row
        self.show_hide_widgets(self.target_list_row, show=False)

        # Insert layouts at the correct positions
        self.left_layout.insertLayout(23, self.query_type_row)
        self.left_layout.insertLayout(
            24, self.target_list_row
        )  # Insert target list row
        self.left_layout.insertLayout(25, self.query_save_row)
        self.left_layout.insertWidget(26, self.query_execute_button)
        self.query_type_combo.currentTextChanged.connect(self.update_query_save_label)

    def remove_metadata_query_widgets(self):
        # Similar to remove_query_widgets from the previous response, but remove
        # all the rows and widgets added in add_metadata_query_widgets.
        widgets_to_remove = [
            self.science_keyword_row,
            self.scientific_category_row,
            self.band_row,
            self.fov_row,
            self.time_resolution_row,
            self.frequency_row,
        ]

        for widget in widgets_to_remove:
            if widget.parent() is not None:
                layout = widget.parent()
                layout.removeItem(widget)
                widget.setParent(None)
                for i in reversed(range(widget.count())):
                    item = widget.takeAt(i)
                    if item.widget():
                        item.widget().deleteLater()

        self.metadata_query_widgets_added = False
        self.query_execute_button.hide()
        self.continue_query_button.hide()

    def remove_metadata_browse(self):
        if self.metadata_path_row.parent() is not None:
            layout = self.metadata_path_row.parent()  # Get the parent layout

            # Remove all items from the layout
            for i in reversed(range(self.metadata_path_row.count())):
                item = self.metadata_path_row.takeAt(i)
                if item.widget():
                    item.widget().deleteLater()

            layout.removeItem(
                self.metadata_path_row
            )  # Remove the row layout from its parent
            self.metadata_path_row.setParent(None)  # Set the parent to None
            self.metadata = None  # Clear any loaded metadata

    def toggle_metadata_browse(self, mode):
        if mode == "get":
            if self.metadata_path_row.parent() is None:  # Check if already added
                # self.left_layout.insertLayout(8, self.metadata_path_row)
                # # Re-insert at correct position
                # self.left_layout.update()  # Force layout update to show the row
                self.remove_query_widgets()  # Remove query widgets if present
                self.add_metadata_widgets()
                if self.metadata_query_widgets_added:
                    self.remove_metadata_query_widgets()
        elif mode == "query":
            if self.query_type_row.parent() is None:
                self.remove_metadata_browse()  # Remove browse widgets if present
                self.add_query_widgets()
        else:
            self.remove_metadata_browse()
            self.remove_query_widgets()

    def remove_query_widgets(self):
        """Removes the query type and save location rows from the layout."""

        # Remove query type row
        if self.query_type_row.parent() is not None:  # Check if row is in layout
            layout = self.query_type_row.parent()
            layout.removeItem(self.query_type_row)  # Remove the row
            self.query_type_row.setParent(None)  # Disassociate the row from parent

            # Delete widgets in row
            for i in reversed(range(self.query_type_row.count())):
                item = self.query_type_row.takeAt(i)
                widget = item.widget()
                if widget is not None:  # Check if it's a widget before deleting
                    widget.deleteLater()

        # Remove query save row (same logic as above)
        if self.query_save_row.parent() is not None:
            layout = self.query_save_row.parent()
            layout.removeItem(self.query_save_row)
            self.query_save_row.setParent(None)

            for i in reversed(range(self.query_save_row.count())):
                item = self.query_save_row.takeAt(i)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        # Remove execute query button
        if self.query_execute_button.parent() is not None:
            layout = self.left_layout  # Directly access the main vertical layout
            # Remove the widget from the layout
            index = layout.indexOf(self.query_execute_button)
            if index >= 0:
                item = layout.takeAt(index)
                if item.widget() is not None:
                    item.widget().deleteLater()
        if self.target_list_row.parent() is not None:
            layout = self.target_list_row.parent()
            layout.removeItem(self.target_list_row)
            self.target_list_row.setParent(None)
            for i in reversed(range(self.target_list_row.count())):
                item = self.target_list_row.takeAt(i)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        self.metadata_query_widgets_added = False

    def reset_fields(self):
        self.output_entry.clear()
        self.tng_entry.clear()
        self.galaxy_zoo_entry.clear()
        self.galaxy_zoo_checkbox.setChecked(False)
        self.hubble_entry.clear()
        self.hubble_checkbox.setChecked(False)
        self.ncpu_entry.clear()
        self.mail_entry.clear()
        self.n_sims_entry.clear()
        self.metadata_path_entry.clear()
        self.comp_mode_combo.setCurrentText("sequential")
        if self.local_mode_combo.currentText() == "remote":
            self.remote_address_entry.clear()
            self.remote_user_entry.clear()
            self.remote_key_entry.clear()
            self.remote_key_pass_entry.clear()
            self.remote_config_entry.clear()
            self.remote_mode_combo.setCurrentText("MPI")
            self.remote_dir_line.clear()
            self.remote_folder_checkbox.setChecked(False)
        self.local_mode_combo.setCurrentText("local")
        if self.metadata_mode_combo.currentText() == "query":
            self.query_save_entry.clear()
        self.metadata_mode_combo.setCurrentText("get")
        self.project_name_entry.clear()
        self.save_format_combo.setCurrentText("npz")
        self.redshift_entry.clear()
        self.num_lines_entry.clear()
        self.snr_checkbox.setChecked(False)
        self.min_line_width_slider.setValue(200)
        self.max_line_width_slider.setValue(400)
        self.robust_slider.setValue(0)
        self.robust_value_label.setText("0")
        self.snr_entry.clear()
        self.fix_spatial_checkbox.setChecked(False)
        self.n_pix_entry.clear()
        self.fix_spectral_checkbox.setChecked(False)
        self.n_channels_entry.clear()
        self.ir_luminosity_checkbox.setChecked(False)
        self.ir_luminosity_entry.clear()
        self.model_combo.setCurrentText("Point")  # Reset to default model
        self.tng_api_key_entry.clear()
        self.line_mode_checkbox.setChecked(False)
        self.serendipitous_checkbox.setChecked(False)

    def load_settings(self):
        self.output_entry.setText(self.settings.value("output_directory", ""))
        self.tng_entry.setText(self.settings.value("tng_directory", ""))
        self.galaxy_zoo_entry.setText(self.settings.value("galaxy_zoo_directory", ""))
        self.galaxy_zoo_checkbox.setChecked(
            self.settings.value("get_galaxy_zoo", False, type=bool)
        )
        self.hubble_entry.setText(self.settings.value("hubble_directory", ""))
        self.hubble_checkbox.setChecked(
            self.settings.value("get_hubble_100", False, type=bool)
        )
        self.mail_entry.setText(self.settings.value("email", ""))
        self.n_sims_entry.setText(self.settings.value("n_sims", ""))
        self.ncpu_entry.setText(self.settings.value("ncpu", ""))
        self.metadata_mode_combo.setCurrentText(
            self.settings.value("metadata_mode", "")
        )
        self.comp_mode_combo.setCurrentText(self.settings.value("comp_mode", ""))
        self.local_mode_combo.setCurrentText(self.settings.value("local_mode", ""))
        if self.local_mode_combo.currentText() == "remote":
            self.remote_address_entry.setText(self.settings.value("remote_address", ""))
            self.remote_user_entry.setText(self.settings.value("remote_user", ""))
            self.remote_key_entry.setText(self.settings.value("remote_key", ""))
            self.remote_key_pass_entry.setText(
                self.settings.value("remote_key_pass", "")
            )
            self.remote_config_entry.setText(self.settings.value("remote_config", ""))
            self.remote_mode_combo.setCurrentText(
                self.settings.value("remote_mode", "")
            )
            remote_folder = self.settings.value("remote_folder", False, type=bool)
            self.remote_folder_checkbox.setChecked(remote_folder)
            if remote_folder:
                self.remote_dir_line.setText(self.settings.value("remote_dir", ""))
        self.metadata_path_entry.setText(self.settings.value("metadata_path", ""))
        self.project_name_entry.setText(self.settings.value("project_name", ""))
        self.save_format_combo.setCurrentText(self.settings.value("save_format", ""))
        if (
            self.metadata_mode_combo.currentText() == "get"
            and self.metadata_path_entry.text() != ""
        ):
            self.load_metadata(self.metadata_path_entry.text())
        elif self.metadata_mode_combo.currentText() == "query":
            self.query_save_entry.setText(self.settings.value("query_save_entry", ""))
        if self.galaxy_zoo_entry.text() != "":
            if self.local_mode_combo.currentText() == "local":
                kaggle_path = os.path.join(os.path.expanduser("~"), ".kaggle")
                if not os.path.exists(kaggle_path):
                    os.mkdir(kaggle_path)
                kaggle_file = os.path.join(kaggle_path, "kaggle.json")
                if not os.path.exists(kaggle_file):
                    shutil.copyfile(
                        os.path.join(os.getcwd(), "kaggle.json"), kaggle_file
                    )
                try:
                    if os.path.exists(self.galaxy_zoo_entry.text()):
                        if self.galaxy_zoo_checkbox.isChecked():
                            if not os.path.exists(
                                os.path.join(self.galaxy_zoo_entry.text(), "images_gz2")
                            ):
                                self.terminal.add_log("Downloading Galaxy Zoo")
                                runnable = DownloadGalaxyZooRunnable(self)
                                self.thread_pool.start(runnable)
                except Exception as e:
                    self.terminal.add_log(f"Cannot dowload Galaxy Zoo: {e}")

            else:
                if (
                    self.remote_address_entry.text() != ""
                    and self.remote_user_entry.text() != ""
                    and self.remote_key_entry.text() != ""
                ):
                    try:
                        self.download_galaxy_zoo_on_remote()
                    except (
                        Exception
                    ) as e:  # Catch any exception that occurs during download
                        error_message = (
                            f"Error downloading Galaxy Zoo data on remote machine: {e}"
                        )
                        print(error_message)  # Print the error to the console
                        self.terminal.add_log(
                            error_message
                        )  # Add the error to your ALMASimulator terminal
        if self.hubble_entry.text() != "":
            if self.local_mode_combo.currentText() == "local":
                try:
                    if not os.path.exists(self.hubble_entry.text()):
                        os.mkdir(self.hubble_entry.text())
                    if self.hubble_checkbox.isChecked():
                        if not os.path.exists(
                            os.path.join(self.hubble_entry.text(), "top100")
                        ):
                            # pool = QThreadPool.globalInstance()
                            runnable = DownloadHubbleRunnable(self)
                            self.thread_pool.start(runnable)
                except Exception as e:
                    self.terminal.add_log(f"Cannot dowload Hubble 100: {e}")

            else:
                if (
                    self.remote_address_entry.text() != ""
                    and self.remote_user_entry.text() != ""
                    and self.remote_key_entry.text() != ""
                ):
                    try:
                        self.download_hubble_on_remote()
                    except (
                        Exception
                    ) as e:  # Catch any exception that occurs during download
                        error_message = (
                            f"Error downloading Galaxy Zoo data on remote machine: {e}"
                        )
                        print(error_message)  # Print the error to the console
                        self.terminal.add_log(
                            error_message
                        )  # Add the error to your ALMASimulator terminal
        line_mode = self.settings.value("line_mode", False, type=bool)
        self.tng_api_key_entry.setText(self.settings.value("tng_api_key", ""))
        self.line_mode_checkbox.setChecked(line_mode)
        if line_mode:
            self.line_index_entry.setText(self.settings.value("line_indices", ""))
        else:
            # Load non-line mode values
            self.redshift_entry.setText(self.settings.value("redshifts", ""))
            self.num_lines_entry.setText(self.settings.value("num_lines", ""))
        self.min_line_width_slider.setValue(
            int(self.settings.value("min_line_width", 200))
        )
        self.max_line_width_slider.setValue(
            int(self.settings.value("max_line_width", 400))
        )
        self.robust_slider.setValue(int(self.settings.value("robust", 0)))
        self.snr_entry.setText(self.settings.value("snr", ""))
        self.snr_checkbox.setChecked(self.settings.value("set_snr", False, type=bool))
        self.fix_spatial_checkbox.setChecked(
            self.settings.value("fix_spatial", False, type=bool)
        )
        self.n_pix_entry.setText(self.settings.value("n_pix", ""))
        self.fix_spectral_checkbox.setChecked(
            self.settings.value("fix_spectral", False, type=bool)
        )
        self.n_channels_entry.setText(self.settings.value("n_channels", ""))
        self.serendipitous_checkbox.setChecked(
            self.settings.value("inject_serendipitous", False, type=bool)
        )
        self.model_combo.setCurrentText(self.settings.value("model", ""))
        self.tng_api_key_entry.setText(self.settings.value("tng_api_key", ""))
        self.toggle_tng_api_key_row()
        self.ir_luminosity_checkbox.setChecked(
            self.settings.value("set_ir_luminosity", False, type=bool)
        )
        self.ir_luminosity_entry.setText(self.settings.value("ir_luminosity", ""))

    def load_settings_on_remote(self):
        self.output_entry.setText(self.settings.value("output_directory", ""))
        self.tng_entry.setText(self.settings.value("tng_directory", ""))
        self.galaxy_zoo_entry.setText(self.settings.value("galaxy_zoo_directory", ""))
        self.hubble_entry.setText(self.settings.value("hubble_directory", ""))
        self.n_sims_entry.setText(self.settings.value("n_sims", ""))
        self.ncpu_entry.setText(self.settings.value("ncpu", ""))
        self.mail_entry.setText(self.settings.value("email", ""))
        self.metadata_path_entry.setText("")

    @classmethod
    def populate_class_variables(cls, terminal, ncpu_entry, thread_pool):
        cls.terminal = terminal
        cls.ncpu_entry = ncpu_entry
        cls.thread_pool = thread_pool

    def closeEvent(self, event):
        if self.local_mode_combo.currentText() == "local":
            if self.thread_pool.activeThreadCount() > 0:
                event.ignore()
                self.hide()
                self.show_background_notification()
            else:
                self.save_settings()
                self.stop_simulation_flag = True
                self.thread_pool.waitForDone()
                super().closeEvent(event)
        else:
            if self.remote_simulation_finished is False:
                event.ignore()
                self.hide()
                self.show_background_notification()
            else:
                self.save_settings()
                self.stop_simulation_flag = True
                self.thread_pool.waitForDone()
                super().closeEvent(event)

    def show_background_notification(self):
        if self.tray_icon is None:
            path = os.path.dirname(self.main_path)
            icon_path = os.path.join(path, "pictures", "almasim-icon.png")
            icon = QIcon(icon_path)
            self.tray_icon = QSystemTrayIcon(icon, self)
            menu = QMenu()
            restore_action = menu.addAction("Restore")
            restore_action.triggered.connect(self.showNormal)  # Restore the window
            exit_action = menu.addAction("Exit")
            exit_action.triggered.connect(QApplication.instance().quit())
            self.tray_icon.setContextMenu(menu)
            self.tray_icon.setIcon(icon)
        self.tray_icon.showMessage(
            "ALMA Simulator",
            "Simulations running in the background.",
            QSystemTrayIcon.MessageIcon.Information,
            5000,
        )
        self.tray_icon.show()

    def save_settings(self):
        self.settings.setValue("output_directory", self.output_entry.text())
        self.settings.setValue("tng_directory", self.tng_entry.text())
        self.settings.setValue("galaxy_zoo_directory", self.galaxy_zoo_entry.text())
        self.settings.setValue("hubble_directory", self.hubble_entry.text())
        self.settings.setValue("n_sims", self.n_sims_entry.text())
        self.settings.setValue("ncpu", self.ncpu_entry.text())
        self.settings.setValue("email", self.mail_entry.text())
        self.settings.setValue("project_name", self.project_name_entry.text())
        if self.metadata_mode_combo.currentText() == "get":
            self.settings.setValue("metadata_path", self.metadata_path_entry.text())
        elif self.metadata_mode_combo.currentText() == "query":
            self.settings.setValue("query_save_entry", self.query_save_entry.text())
        self.settings.setValue("metadata_mode", self.metadata_mode_combo.currentText())
        self.settings.setValue("comp_mode", self.comp_mode_combo.currentText())
        self.settings.setValue("local_mode", self.local_mode_combo.currentText())
        if self.local_mode_combo.currentText() == "remote":
            self.settings.setValue("remote_address", self.remote_address_entry.text())
            self.settings.setValue("remote_user", self.remote_user_entry.text())
            self.settings.setValue("remote_key", self.remote_key_entry.text())
            self.settings.setValue("remote_key_pass", self.remote_key_pass_entry.text())
            self.settings.setValue("remote_config", self.remote_config_entry.text())
            self.settings.setValue("remote_mode", self.remote_mode_combo.currentText())
            self.settings.setValue("remote_folder", self.remote_dir_line.text())
        self.settings.setValue("save_format", self.save_format_combo.currentText())
        self.settings.setValue("line_mode", self.line_mode_checkbox.isChecked())
        if self.line_mode_checkbox.isChecked():
            self.settings.setValue("line_indices", self.line_index_entry.text())
        else:
            # Save non-line mode values
            self.settings.setValue("redshifts", self.redshift_entry.text())
            self.settings.setValue("num_lines", self.num_lines_entry.text())
        self.settings.setValue("min_line_width", self.min_line_width_slider.value())
        self.settings.setValue("max_line_width", self.max_line_width_slider.value())
        self.settings.setValue("robust", self.robust_slider.value())
        self.settings.setValue("set_snr", self.snr_checkbox.isChecked())
        self.settings.setValue("snr", self.snr_entry.text())
        self.settings.setValue("fix_spatial", self.fix_spatial_checkbox.isChecked())
        self.settings.setValue("n_pix", self.n_pix_entry.text())
        self.settings.setValue("fix_spectral", self.fix_spectral_checkbox.isChecked())
        self.settings.setValue("n_channels", self.n_channels_entry.text())
        self.settings.setValue(
            "inject_serendipitous", self.serendipitous_checkbox.isChecked()
        )
        self.settings.setValue("model", self.model_combo.currentText())
        self.settings.setValue("tng_api_key", self.tng_api_key_entry.text())
        self.settings.setValue(
            "set_ir_luminosity", self.ir_luminosity_checkbox.isChecked()
        )
        self.settings.setValue("ir_luminosity", self.ir_luminosity_entry.text())

    def show_hide_widgets(self, layout, show=True):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                if show:
                    item.widget().show()
                else:
                    item.widget().hide()

    # -------- Browse Functions ------------------
    def add_metadata_query_widgets(self):
        # Create widgets for querying parameters
        science_keyword_label = QLabel(
            "Select Science Keyword by number (space-separated):"
        )
        self.science_keyword_entry = QLineEdit()  # Use QLineEdit instead of input

        scientific_category_label = QLabel(
            "Select Scientific Category by number (space-separated):"
        )
        self.scientific_category_entry = QLineEdit()

        band_label = QLabel("Select observing bands (space-separated):")
        self.band_entry = QLineEdit()

        fov_label = QLabel("Select FOV range (min max) or max only (space-separated):")
        self.fov_entry = QLineEdit()

        time_resolution_label = QLabel(
            "Select integration time  range (min max) or max only (space-separated):"
        )
        self.time_resolution_entry = QLineEdit()

        frequency_label = QLabel(
            "Select source frequency range (min max) or max only (space-separated):"
        )
        self.frequency_entry = QLineEdit()

        self.continue_query_button = QPushButton("Continue Query")
        self.continue_query_button.clicked.connect(self.execute_query)

        # Create layouts and add widgets
        self.science_keyword_row = QHBoxLayout()
        self.science_keyword_row.addWidget(science_keyword_label)
        self.science_keyword_row.addWidget(self.science_keyword_entry)

        self.scientific_category_row = QHBoxLayout()
        self.scientific_category_row.addWidget(scientific_category_label)
        self.scientific_category_row.addWidget(self.scientific_category_entry)

        self.band_row = QHBoxLayout()
        self.band_row.addWidget(band_label)
        self.band_row.addWidget(self.band_entry)

        self.fov_row = QHBoxLayout()
        self.fov_row.addWidget(fov_label)
        self.fov_row.addWidget(self.fov_entry)

        self.time_resolution_row = QHBoxLayout()
        self.time_resolution_row.addWidget(time_resolution_label)
        self.time_resolution_row.addWidget(self.time_resolution_entry)

        self.frequency_row = QHBoxLayout()
        self.frequency_row.addWidget(frequency_label)
        self.frequency_row.addWidget(self.frequency_entry)

        self.continue_query_row = QHBoxLayout()
        self.continue_query_row.addWidget(self.continue_query_button)

        # Insert rows into left_layout (adjust index if needed)
        self.left_layout.insertLayout(23, self.science_keyword_row)
        self.left_layout.insertLayout(24, self.scientific_category_row)
        self.left_layout.insertLayout(25, self.band_row)
        self.left_layout.insertLayout(26, self.fov_row)
        self.left_layout.insertLayout(27, self.time_resolution_row)
        self.left_layout.insertLayout(28, self.frequency_row)
        self.left_layout.insertWidget(29, self.continue_query_button)
        self.terminal.add_log(
            "\n\nFill out the fields and click 'Continue Query' to proceed."
        )
        self.query_execute_button.hide()  # Hide the execute query button

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if (
            self.local_mode_combo.currentText() == "remote"
            and self.remote_address_entry.text() != ""
            and self.remote_key_entry.text() != ""
            and self.remote_user_entry != ""
        ):
            if directory:
                remote_dir = self.map_to_remote_directory(directory)
                if self.remote_key_pass_entry.text() != "":
                    with pysftp.Connection(
                        self.remote_address_entry.text(),
                        username=self.remote_user_entry.text(),
                        private_key=self.remote_key_entry.text(),
                        private_key_pass=self.remote_key_pass_entry.text(),
                    ) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                else:
                    with pysftp.Connection(
                        self.remote_address_entry.text(),
                        username=self.remote_user_entry.text(),
                        private_key=self.remote_key_entry.text(),
                    ) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                self.output_entry.setText(remote_dir)
        else:
            if directory:
                self.output_entry.setText(directory)

    def map_to_remote_directory(self, directory):
        directory_name = directory.split(os.path.sep)[-1]
        if self.remote_dir_line.text() != "":
            if not self.remote_dir_line.text().startswith("/"):
                self.remote_dir_line.setText("/" + self.remote_dir_line.text())
            directory_path = os.path.join(self.remote_dir_line.text(), directory_name)
        else:
            directory_path = os.path.join(
                "/home", self.remote_user_entry.text(), directory_name
            )
        return directory_path

    def browse_tng_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select TNG Directory")
        if (
            self.local_mode_combo.currentText() == "remote"
            and self.remote_address_entry.text() != ""
            and self.remote_key_entry.text() != ""
            and self.remote_user_entry != ""
        ):
            if directory:
                remote_dir = self.map_to_remote_directory(directory)
                if self.remote_key_pass_entry.text() != "":
                    with pysftp.Connection(
                        self.remote_address_entry.text(),
                        username=self.remote_user_entry.text(),
                        private_key=self.remote_key_entry.text(),
                        private_key_pass=self.remote_key_pass_entry.text(),
                    ) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                else:
                    with pysftp.Connection(
                        self.remote_address_entry.text(),
                        username=self.remote_user_entry.text(),
                        private_key=self.remote_key_entry.text(),
                    ) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                self.tng_entry.setText(remote_dir)
        else:
            if directory:
                self.tng_entry.setText(directory)

    def browse_galaxy_zoo_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Galaxy Zoo Directory"
        )
        if (
            self.local_mode_combo.currentText() == "remote"
            and self.remote_address_entry.text() != ""
            and self.remote_key_entry.text() != ""
            and self.remote_user_entry != ""
        ):
            if directory:
                remote_dir = self.map_to_remote_directory(directory)
                if self.remote_key_pass_entry.text() != "":
                    with pysftp.Connection(
                        self.remote_address_entry.text(),
                        username=self.remote_user_entry.text(),
                        private_key=self.remote_key_entry.text(),
                        private_key_pass=self.remote_key_pass_entry.text(),
                    ) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                else:
                    with pysftp.Connection(
                        self.remote_address_entry.text(),
                        username=self.remote_user_entry.text(),
                        private_key=self.remote_key_entry.text(),
                    ) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                self.galaxy_zoo_entry.setText(remote_dir)
        else:
            if directory:
                self.galaxy_zoo_entry.setText(directory)

    def browse_hubble_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Hubble Top 100 Directory"
        )
        if (
            self.local_mode_combo.currentText() == "remote"
            and self.remote_address_entry.text() != ""
            and self.remote_key_entry.text() != ""
            and self.remote_user_entry != ""
        ):
            if directory:
                remote_dir = self.map_to_remote_directory(directory)
                if self.remote_key_pass_entry.text() != "":
                    with pysftp.Connection(
                        self.remote_address_entry.text(),
                        username=self.remote_user_entry.text(),
                        private_key=self.remote_key_entry.text(),
                        private_key_pass=self.remote_key_pass_entry.text(),
                    ) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                else:
                    with pysftp.Connection(
                        self.remote_address_entry.text(),
                        username=self.remote_user_entry.text(),
                        private_key=self.remote_key_entry.text(),
                    ) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                self.hubble_entry.setText(remote_dir)
        else:
            if directory:
                self.hubble_entry.setText(directory)

    def browse_metadata_path(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Metadata File",
            os.path.join(self.main_path, "metadata"),
            "CSV Files (*.csv)",
        )
        if file:
            self.metadata_path_entry.setText(file)
            self.metadata_path_set()

    def browse_ssh_key(self):
        file_dialog = QFileDialog()
        ssh_key_file, _ = file_dialog.getOpenFileName(
            self,
            "Select SSH Key File",
            os.path.join(os.path.expanduser("~"), ".ssh"),
            "SSH Key Files (*.pem *.ppk *.key  *rsa)",
        )
        if ssh_key_file:
            self.remote_key_entry.setText(ssh_key_file)

    def browse_slurm_config(self):
        file_dialog = QFileDialog()
        slurm_config_file, _ = file_dialog.getOpenFileName(
            self,
            "Select Slurm Config File",
            self.main_path,
            "Slurm Config Files (*.json)",
        )
        if slurm_config_file:
            self.remote_config_entry.setText(slurm_config_file)

    def select_metadata_path(self):
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Select Metadata File",
            os.path.join(self.main_path, "metadata"),
            "CSV Files (*.csv)",
        )
        if file:
            self.query_save_entry.setText(file)
            # self.metadata_path_set()

    def metadata_path_set(self):
        metadata_path = self.metadata_path_entry.text()
        self.load_metadata(metadata_path)  # Pass only the metadata_path

    def browse_target_list(self):
        """Opens a file dialog to select the target list file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Target List",
            os.path.join(self.main_path, "metadata"),
            "CSV Files (*.csv)",
        )
        if file_path:
            self.target_list_entry.setText(file_path)

    # -------- Query ALMA Database Functions -------
    def get_tap_service(self):
        urls = [
            "https://almascience.eso.org/tap",
            "https://almascience.nao.ac.jp/tap",
            "https://almascience.nrao.edu/tap",
        ]
        while True:  # Infinite loop to keep trying until successful
            for url in urls:
                try:
                    service = pyvo.dal.TAPService(url)
                    # Test the connection with a simple query to ensure the service is
                    # working
                    service.search("SELECT TOP 1 * FROM ivoa.obscore")
                    self.terminal.add_log(f"Connected successfully to {url}")
                    return service
                except Exception as e:
                    self.terminal.add_log(f"Failed to connect to {url}: {e}")
                    self.terminal.add_log("Retrying other servers...")
            self.terminal.add_log("All URLs attempted and failed, retrying...")

    def plot_science_keywords_distributions(self, master_path):
        service = self.get_tap_service()
        plot_dir = os.path.join(master_path, "plots")

        # Check if plot directory exists
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            existing_plots = (
                []
            )  # Initialize as empty list if plot directory doesn't exist
        else:
            # Check if plot files already exist
            existing_plots = [f for f in os.listdir(plot_dir) if f.endswith(".png")]

        expected_plots = [
            "science_vs_bands.png",
            "science_vs_int_time.png",
            "science_vs_source_freq.png",
            "science_vs_FoV.png",
        ]

        if all(plot_file in existing_plots for plot_file in expected_plots):
            return
        else:
            self.terminal.add_log(
                f"Generating helping plots to guide you in the scientific query, \
                    check them in {plot_dir}."
            )
            # Identify missing plots
        missing_plots = [plot for plot in expected_plots if plot not in existing_plots]

        # Query only for variables associated with missing plots
        query_variables = set()
        for missing_plot in missing_plots:
            if missing_plot == "science_vs_bands.png":
                query_variables.update(["science_keyword", "band_list"])
            elif missing_plot == "science_vs_int_time.png":
                query_variables.update(["science_keyword", "t_resolution"])
            elif missing_plot == "science_vs_source_freq.png":
                query_variables.update(["science_keyword", "frequency"])
            elif missing_plot == "science_vs_FoV.png":
                query_variables.update(["science_keyword", "band_list"])
        query = f"""
                SELECT {', '.join(query_variables)}, member_ous_uid
                FROM ivoa.obscore
                WHERE science_observation = 'T'
                AND is_mosaic = 'F'
                """

        custom_palette = sns.color_palette("tab20")
        sns.set_palette(custom_palette)
        db = service.search(query).to_table().to_pandas()
        db = db.drop_duplicates(subset="member_ous_uid")

        # Splitting the science keywords at commas
        db["science_keyword"] = db["science_keyword"].str.split(",")
        db["science_keyword"] = db["science_keyword"].apply(
            lambda x: [y.strip() for y in x]
        )
        db = db.explode("science_keyword")
        db = db.drop(db[db["science_keyword"] == ""].index)
        db = db.drop(db[db["science_keyword"] == "Exoplanets"].index)
        db = db.drop(db[db["science_keyword"] == "Galaxy structure &evolution"].index)
        db = db.drop(
            db[
                db["science_keyword"] == "Evolved stars: Shaping/physical structure"
            ].index
        )
        old_keys = [
            "Solar system - Trans-Neptunian Objects (TNOs)",
            "Photon-Dominated Regions (PDR)/X-Ray Dominated Regions (XDR)",
            "Luminous and Ultra-Luminous Infra-Red Galaxies (LIRG & ULIRG)",
            "Cosmic Microwave Background (CMB)/Sunyaev-Zel'dovich Effect (SZE)",
            "Active Galactic Nuclei (AGN)/Quasars (QSO)",
            "Inter-Stellar Medium (ISM)/Molecular clouds",
        ]
        short_keyword = {
            old_keys[0]: "Solar System - TNOs",
            old_keys[1]: "Photon/X-Ray Domanited Regions",
            old_keys[2]: "LIRG & ULIRG",
            old_keys[3]: "CMB/Sunyaev-Zel'dovich Effect",
            old_keys[4]: "AGN/QSO",
            old_keys[5]: "ISM & Molecular Clouds",
        }

        db["science_keyword"] = db["science_keyword"].replace(short_keyword)

        for missing_plot in missing_plots:
            if missing_plot == "science_vs_bands.png":
                db["band_list"] = db["band_list"].str.split(" ")
                db["band_list"] = db["band_list"].apply(
                    lambda x: [y.strip() for y in x]
                )
                db = db.explode("band_list")

                db_sk_b = (
                    db.groupby(["science_keyword", "band_list"])
                    .size()
                    .unstack(fill_value=0)
                )

                plt.rcParams["figure.figsize"] = (28, 20)
                db_sk_b.plot(kind="barh", stacked=True, color=custom_palette)
                plt.title("Science Keywords vs. ALMA Bands")
                plt.xlabel("Counts")
                plt.ylabel("Science Keywords")
                plt.legend(
                    bbox_to_anchor=(1.01, 1), loc="upper left", title="ALMA Bands"
                )
                plt.savefig(os.path.join(plot_dir, "science_vs_bands.png"))
                plt.close()

            elif missing_plot == "science_vs_int_time.png":
                db = db[db["t_resolution"] <= 3e4]
                time_bins = np.arange(
                    db["t_resolution"].min(), db["t_resolution"].max(), 1000
                )  # 1000 second bins
                db["time_bin"] = pd.cut(db["t_resolution"], bins=time_bins)

                db_sk_t = (
                    db.groupby(["science_keyword", "time_bin"])
                    .size()
                    .unstack(fill_value=0)
                )

                plt.rcParams["figure.figsize"] = (28, 20)
                db_sk_t.plot(kind="barh", stacked=True)
                plt.title("Science Keywords vs. Integration Time")
                plt.xlabel("Counts")
                plt.ylabel("Science Keywords")
                plt.legend(
                    title="Integration Time", loc="upper left", bbox_to_anchor=(1.01, 1)
                )
                plt.savefig(os.path.join(plot_dir, "science_vs_int_time.png"))
                plt.close()

            elif missing_plot == "science_vs_source_freq.png":
                frequency_bins = np.arange(
                    db["frequency"].min(), db["frequency"].max(), 50
                )  # 50 GHz bins
                db["frequency_bin"] = pd.cut(db["frequency"], bins=frequency_bins)

                db_sk_f = (
                    db.groupby(["science_keyword", "frequency_bin"])
                    .size()
                    .unstack(fill_value=0)
                )

                plt.rcParams["figure.figsize"] = (28, 20)
                db_sk_f.plot(kind="barh", stacked=True, color=custom_palette)
                plt.title("Science Keywords vs. Source Frequency")
                plt.xlabel("Counts")
                plt.ylabel("Science Keywords")
                plt.legend(
                    bbox_to_anchor=(1.01, 1), loc="upper left", title="Frequency"
                )
                plt.savefig(os.path.join(plot_dir, "science_vs_source_freq.png"))
                plt.close()

            elif missing_plot == "science_vs_FoV.png":
                db["band_list"] = db["band_list"].str.split(" ")
                db["band_list"] = db["band_list"].apply(
                    lambda x: [y.strip() for y in x]
                )
                db = db.explode("band_list")
                db["fov"] = db["band_list"].apply(
                    lambda x: ual.get_fov_from_band(int(x))
                )
                fov_bins = np.arange(
                    db["fov"].min(), db["fov"].max(), 10
                )  # 10 arcsec bins
                db["fov_bins"] = pd.cut(db["fov"], bins=fov_bins)

                db_sk_fov = (
                    db.groupby(["science_keyword", "fov_bins"])
                    .size()
                    .unstack(fill_value=0)
                )

                plt.rcParams["figure.figsize"] = (28, 20)
                db_sk_fov.plot(kind="barh", stacked=True, color=custom_palette)
                plt.title("Science Keywords vs. FoV")
                plt.xlabel("Counts")
                plt.ylabel("Science Keywords")
                plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="FoV")
                plt.savefig(os.path.join(plot_dir, "science_vs_FoV.png"))
                plt.close()

    def update_query_save_label(self, query_type):
        """Shows/hides the target list row and query save row based on query type."""
        if query_type == "science":
            self.show_hide_widgets(
                self.target_list_row, show=False
            )  # Hide target list row
            self.show_hide_widgets(
                self.query_save_row, show=True
            )  # Show query save row
            self.query_save_label.setText("Save Metadata to:")
        else:  # query_type == "target"
            self.show_hide_widgets(
                self.target_list_row, show=True
            )  # Show target list row
            self.show_hide_widgets(
                self.query_save_row, show=True
            )  # Hide query save row

    def execute_query(self):
        self.terminal.add_log("Executing query...")
        if self.metadata_mode_combo.currentText() == "query":
            query_type = self.query_type_combo.currentText()
            if query_type == "science":
                if (
                    not hasattr(self, "metadata_query_widgets_added")
                    or not self.metadata_query_widgets_added
                ):
                    self.show_scientific_keywords()
                    self.add_metadata_query_widgets()
                    self.metadata_query_widgets_added = True
                    self.query_execute_button.hide()
                else:
                    self.metadata = self.query_for_metadata_by_science_type()
                    self.remove_metadata_query_widgets()
                    self.query_execute_button.show()
            elif query_type == "target":
                if self.target_list_entry.text():
                    self.terminal.add_log(
                        f"Loading target list {self.target_list_entry.text()}"
                    )
                    target_list = pd.read_csv(self.target_list_entry.text())
                    self.target_list = target_list.values.tolist()
                    self.metadata = self.query_for_metadata_by_targets()
            else:
                # Handle invalid query type (optional)
                pass

    def show_scientific_keywords(self):
        # Implement the logic to query metadata based on science type
        self.terminal.add_log("Querying metadata by science type...")
        # self.plot_window = PlotWindow()
        # self.plot_window.show()
        self.science_keywords, self.scientific_categories = ual.get_science_types()
        self.terminal.add_log("Available science keywords:")
        for i, keyword in enumerate(self.science_keywords):
            self.terminal.add_log(f"{i}: {keyword}")
        self.terminal.add_log("\nAvailable scientific categories:")
        for i, category in enumerate(self.scientific_categories):
            self.terminal.add_log(f"{i}: {category}")

    def query_for_metadata_by_science_type(self):
        self.terminal.add_log("Querying by Science Keyword")
        science_keyword_number = self.science_keyword_entry.text()
        scientific_category_number = self.scientific_category_entry.text()
        band = self.band_entry.text()
        fov_input = self.fov_entry.text()
        time_resolution_input = self.time_resolution_entry.text()
        frequency_input = self.frequency_entry.text()
        save_to_input = self.query_save_entry.text()

        # Get selected science keywords and categories
        science_keyword = (
            [self.science_keywords[int(i)] for i in science_keyword_number.split()]
            if science_keyword_number
            else None
        )
        scientific_category = (
            [
                self.scientific_categories[int(i)]
                for i in scientific_category_number.split()
            ]
            if scientific_category_number
            else None
        )
        bands = [int(x) for x in band.split()] if band else None

        def to_range(text):
            values = [float(x) for x in text.split()] if text else None
            return (
                tuple(values)
                if values and len(values) > 1
                else (0, values[0]) if values else None
            )

        fov_range = to_range(fov_input)
        time_resolution_range = to_range(time_resolution_input)
        frequency_range = to_range(frequency_input)
        df = ual.query_by_science_type(
            science_keyword,
            scientific_category,
            bands,
            fov_range,
            time_resolution_range,
            frequency_range,
        )
        df = df.drop_duplicates(subset="member_ous_uid").drop(
            df[df["science_keyword"] == ""].index
        )
        # Rename columns and select relevant data
        rename_columns = {
            "target_name": "ALMA_source_name",
            "pwv": "PWV",
            "schedblock_name": "SB_name",
            "velocity_resolution": "Vel.res.",
            "spatial_resolution": "Ang.res.",
            "s_ra": "RA",
            "s_dec": "Dec",
            "s_fov": "FOV",
            "t_resolution": "Int.Time",
            "cont_sensitivity_bandwidth": "Cont_sens_mJybeam",
            "sensitivity_10kms": "Line_sens_10kms_mJybeam",
            "obs_release_date": "Obs.date",
            "band_list": "Band",
            "bandwidth": "Bandwidth",
            "frequency": "Freq",
            "frequency_support": "Freq.sup.",
        }
        df.rename(columns=rename_columns, inplace=True)
        database = df[
            [
                "ALMA_source_name",
                "Band",
                "PWV",
                "SB_name",
                "Vel.res.",
                "Ang.res.",
                "RA",
                "Dec",
                "FOV",
                "Int.Time",
                "Cont_sens_mJybeam",
                "Line_sens_10kms_mJybeam",
                "Obs.date",
                "Bandwidth",
                "Freq",
                "Freq.sup.",
                "antenna_arrays",
                "proposal_id",
                "member_ous_uid",
                "group_ous_uid",
            ]
        ]
        database.loc[:, "Obs.date"] = database["Obs.date"].apply(
            lambda x: x.split("T")[0]
        )
        database.to_csv(save_to_input, index=False)
        self.metadata = database
        self.terminal.add_log(f"Metadata saved to {save_to_input}")
        del database

    def query_for_metadata_by_targets(self):
        """Query for metadata for all predefined targets and compile the results
            into a single DataFrame.

        Parameters:
        service (pyvo.dal.TAPService): A TAPService instance for querying the database.
        targets (list of tuples): A list where each tuple contains
                (target_name, member_ous_uid).
        path (str): The path to save the results to.

        Returns:
        pandas.DataFrame: A DataFrame containing the results for all queried targets.
        """
        # Query all targets and compile the results
        self.terminal.add_log("Querying metadata from target list...")
        df = ual.query_all_targets(self.target_list)
        df = df.drop_duplicates(subset="member_ous_uid")
        save_to_input = self.query_save_entry.text()
        # Define a dictionary to map existing column names to new names with unit initials
        rename_columns = {
            "target_name": "ALMA_source_name",
            "pwv": "PWV",
            "schedblock_name": "SB_name",
            "velocity_resolution": "Vel.res.",
            "spatial_resolution": "Ang.res.",
            "s_ra": "RA",
            "s_dec": "Dec",
            "s_fov": "FOV",
            "t_resolution": "Int.Time",
            "cont_sensitivity_bandwidth": "Cont_sens_mJybeam",
            "sensitivity_10kms": "Line_sens_10kms_mJybeam",
            "obs_release_date": "Obs.date",
            "band_list": "Band",
            "bandwidth": "Bandwidth",
            "frequency": "Freq",
            "frequency_support": "Freq.sup.",
        }
        df.rename(columns=rename_columns, inplace=True)
        database = df[
            [
                "ALMA_source_name",
                "Band",
                "PWV",
                "SB_name",
                "Vel.res.",
                "Ang.res.",
                "RA",
                "Dec",
                "FOV",
                "Int.Time",
                "Cont_sens_mJybeam",
                "Line_sens_10kms_mJybeam",
                "Obs.date",
                "Bandwidth",
                "Freq",
                "Freq.sup.",
                "antenna_arrays",
                "proposal_id",
                "member_ous_uid",
                "group_ous_uid",
            ]
        ]
        database.loc[:, "Obs.date"] = database["Obs.date"].apply(
            lambda x: x.split("T")[0]
        )
        database.to_csv(save_to_input, index=False)
        self.metadata = database
        self.terminal.add_log(f"Metadata saved to {save_to_input}")

    # ----- Auxiliary Functions ----------------

    def load_metadata(self, metadata_path):
        try:
            self.terminal.add_log(f"Loading metadata from {metadata_path}")
            self.metadata = pd.read_csv(metadata_path)
            self.terminal.add_log(
                "Metadata contains {} samples".format(len(self.metadata))
            )

            # ... rest of your metadata loading logic ...
        except Exception as e:
            self.terminal.add_log(f"Error loading metadata: {e}")
            import traceback

            traceback.print_exc()

    def line_display(self):
        """
        Display the line emission's rest frequency.

        Parameter:
        main_path (str): Path to the directory where the file.csv is stored.

        Return:
        pd.DataFrame : Dataframe with line names and rest frequencies.
        """

        path_line_emission_csv = os.path.join(
            self.main_path, "brightnes", "calibrated_lines.csv"
        )
        db_line = uas.read_line_emission_csv(
            path_line_emission_csv, sep=","
        ).sort_values(by="Line")
        line_names = db_line["Line"].values
        rest_frequencies = db_line["freq(GHz)"].values
        self.terminal.add_log("Please choose the lines from the following list\n")
        for i in range(len(line_names)):
            self.terminal.add_log(
                f"{i}: {line_names[i]} - {rest_frequencies[i]:.2e} GHz\n"
            )
        self.line_displayed = True

    def download_galaxy_zoo(self):
        """
        Downloads a Kaggle dataset to the specified path.
        """
        self.terminal.add_log(
            "\nGalaxy Zoo data not found on disk, downloading from Kaggle..."
        )
        api.authenticate()  # Authenticate with your Kaggle credentials
        dataset_name = "jaimetrickz/galaxy-zoo-2-images"
        # Download the dataset as a zip file
        api.dataset_download_files(
            dataset_name, path=self.galaxy_zoo_entry.text(), unzip=True
        )
        self.terminal.add_log(
            f"\nDataset {dataset_name} downloaded to {self.galaxy_zoo_entry.text()}"
        )
        # os.remove(os.path.join(self.galaxy_zoo_entry.text(), "galaxy-zoo-2-images.zip"))

    def download_galaxy_zoo_on_remote(self):
        """
        Downloads a Kaggle dataset to the specified path.
        """
        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
                private_key_pass=self.remote_key_pass_entry.text(),
            )

        else:
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
            )
        if sftp.exists(self.galaxy_zoo_entry.text()):
            if not sftp.listdir(self.galaxy_zoo_entry.text()):
                self.terminal.add_log(
                    "\nGalaxy Zoo data not found on disk, downloading from Kaggle..."
                )
                if not sftp.exists(
                    "/home/{}/.kaggle".format(self.remote_user_entry.text())
                ):
                    sftp.mkdir("/home/{}/.kaggle".format(self.remote_user_entry.text()))
                if not sftp.exists(
                    "/home/{}/.kaggle/kaggle.json".format(self.remote_user_entry.text())
                ):
                    sftp.put(
                        os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json"),
                        "/home/{}/.kaggle/kaggle.json".format(
                            self.remote_user_entry.text()
                        ),
                    )
                    sftp.chmod(
                        "/home/{}/.kaggle/kaggle.json".format(
                            self.remote_user_entry.text()
                        ),
                        600,
                    )
                if self.remote_key_pass_entry.text() != "":
                    key = paramiko.RSAKey.from_private_key_file(
                        self.remote_key_entry.text(),
                        password=self.remote_key_pass_entry.text(),
                    )
                else:
                    key = paramiko.RSAKey.from_private_key_file(
                        self.remote_key_entry.text()
                    )
                venv_dir = os.path.join(
                    "/home/{}/".format(self.remote_user_entry.text()), "almasim_env"
                )
                commands = f"""
                source {venv_dir}/bin/activate
                python -c "from kaggle import api; \
                    api.dataset_download_files('jaimetrickz/galaxy-zoo-2-images', \
                        path='{self.galaxy_zoo_entry.text()}', unzip=True)"
                """
                paramiko_client = paramiko.SSHClient()
                paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                paramiko_client.connect(
                    self.remote_address_entry.text(),
                    username=self.remote_user_entry.text(),
                    pkey=key,
                )
                stdin, stdout, stderr = paramiko_client.exec_command(commands)
                self.terminal.add_log(stdout.read().decode())
                self.terminal.add_log(stderr.read().decode())

    def download_hubble(self):
        """
        Download 10GB of iconic Hubble images to hubble_image_path/top100.
        These are large in size which allows random cropping and
        scaling for data-augmentation.
        """
        baseurl = "https://esahubble.org/static/images/zip/top100/top100-original.zip"
        self.terminal.add_log("Hubble images not found on disk, downloading ...")
        zipfilename = os.path.join(self.hubble_entry.text(), os.path.basename(baseurl))
        if not os.path.exists(zipfilename):
            response = requests.Session().get(baseurl, stream=True)
            with open(zipfilename, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        with zipfile.ZipFile(zipfilename) as zf:
            zf.extractall(self.hubble_entry.text())
        os.remove(zipfilename)
        message = f"Hubble 100 Images downloaded to {self.hubble_entry.text()}"
        self.terminal.add_log(message)

    def download_hubble_on_remote(self):
        self.terminal.add_log("Not yet implemented")

    def check_tng_dirs(self):
        tng_dir = self.tng_entry.text()
        if not os.path.exists(os.path.join(tng_dir, "TNG100-1")):
            os.makedirs(os.path.join(tng_dir, "TNG100-1"))
        if not os.path.exists(os.path.join(tng_dir, "TNG100-1", "output")):
            os.makedirs(os.path.join(tng_dir, "TNG100-1", "output"))
        if not os.path.exists(os.path.join(tng_dir, "TNG100-1", "postprocessing")):
            os.makedirs(os.path.join(tng_dir, "TNG100-1", "postprocessing"))
        if not os.path.exists(
            os.path.join(tng_dir, "TNG100-1", "postprocessing", "offsets")
        ):
            os.makedirs(os.path.join(tng_dir, "TNG100-1", "postprocessing", "offsets"))
        if not isfile(os.path.join(tng_dir, "TNG100-1", "simulation.hdf5")):
            if self.terminal is not None:
                self.terminal.add_log("Downloading simulation file")
            url = "http://www.tng-project.org/api/TNG100-1/files/simulation.hdf5"
            cmd = "wget -nv --content-disposition --header=API-Key:{} -O {} {}".format(
                self.tng_api_key_entry.text(),
                os.path.join(tng_dir, "TNG100-1", "simulation.hdf5"),
                url,
            )
            subprocess.run(cmd, shell=True)
            if self.terminal is not None:
                self.terminal.add_log("Done.")

    def create_remote_environment(self):
        self.terminal.add_log("Checking ALMASim environment")
        repo_url = "https://github.com/MicheleDelliVeneri/ALMASim.git"
        if self.remote_dir_line.text() != "":
            work_dir = self.remote_dir_line.text()
            repo_dir = os.path.join(work_dir, "ALMASim")
            venv_dir = os.path.join(work_dir, "almasim_env")
        else:
            venv_dir = os.path.join(
                "/home/{}".format(self.remote_user_entry.text()), "almasim_env"
            )
            repo_dir = os.path.join(
                "/home/{}".format(self.remote_user_entry.text()), "ALMASim"
            )
        self.remote_main_dir = repo_dir
        self.remote_venv_dir = venv_dir
        if self.remote_key_pass_entry.text() != "":
            key = paramiko.RSAKey.from_private_key_file(
                self.remote_key_entry.text(), password=self.remote_key_pass_entry.text()
            )
        else:
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text())

        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
                private_key_pass=self.remote_key_pass_entry.text(),
            )

        else:
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
            )
        if not sftp.exists("/home/{}/.config".format(self.remote_user_entry.text())):
            sftp.mkdir("/home/{}/.config".format(self.remote_user_entry.text()))

        if not sftp.exists(
            "/home/.config/{}/{}".format(
                self.remote_user_entry.text(), self.settings_path.split(os.sep)[-1]
            )
        ):
            sftp.put(
                self.settings_path,
                "/home/{}/.config/{}".format(
                    self.remote_user_entry.text(), self.settings_path.split(os.sep)[-1]
                ),
            )
        sftp.chmod(
            "/home/{}/.config/{}".format(
                self.remote_user_entry.text(), self.settings_path.split(os.sep)[-1]
            ),
            600,
        )
        # Get the path to the Python executable
        paramiko_client = paramiko.SSHClient()
        paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        paramiko_client.connect(
            self.remote_address_entry.text(),
            username=self.remote_user_entry.text(),
            pkey=key,
        )
        stdin, stdout, stderr = paramiko_client.exec_command("which python3.12")
        python_path = stdout.read().decode().strip()
        if not python_path:
            self.terminal.add_log("Python 3.12 not found on remote machine.")
            paramiko_client.close()
            return

        commands = f"""
            if [ ! -d {repo_dir} ]; then
                git clone {repo_url} {repo_dir}
            fi
            cd {repo_dir}
            git pull
            if [ ! -d {venv_dir} ]; then
                {python_path} -m venv {venv_dir}
                source {venv_dir}/bin/activate
                pip install --upgrade pip
                pip install -e .
            fi
            """

        stdin, stdout, stderr = paramiko_client.exec_command(commands)
        self.terminal.add_log(stdout.read().decode())
        self.terminal.add_log(stderr.read().decode())

    def create_remote_output_dir(self):
        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
                private_key_pass=self.remote_key_pass_entry.text(),
            )

        else:
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
            )
        output_path = os.path.join(
            self.output_entry.text(), self.project_name_entry.text()
        )
        plot_path = os.path.join(output_path, "plots")
        if not sftp.exists(output_path):
            sftp.mkdir(output_path)
        if not sftp.exists(plot_path):
            sftp.mkdir(plot_path)

    def remote_check_tng_dirs(self):
        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
                private_key_pass=self.remote_key_pass_entry.text(),
            )

        else:
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
            )
        tng_dir = self.tng_entry.text()
        if not sftp.exists(os.path.join(tng_dir, "TNG100-1")):
            sftp.mkdir(os.path.join(tng_dir, "TNG100-1"))
        if not sftp.exists(os.path.join(tng_dir, "TNG100-1", "output")):
            sftp.mkdir(os.path.join(tng_dir, "TNG100-1", "output"))
        if not sftp.exists(os.path.join(tng_dir, "TNG100-1", "postprocessing")):
            sftp.mkdir(os.path.join(tng_dir, "TNG100-1", "postprocessing"))
        if not sftp.exists(
            os.path.join(tng_dir, "TNG100-1", "postprocessing", "offsets")
        ):
            sftp.mkdir(os.path.join(tng_dir, "TNG100-1", "postprocessing", "offsets"))
        if not sftp.exists(os.path.join(tng_dir, "TNG100-1", "simulation.hdf5")):
            self.terminal.add_log("Downloading simulation file")
            url = "http://www.tng-project.org/api/TNG100-1/files/simulation.hdf5"
            cmd = "wget -nv --content-disposition --header=API-Key:{} -O {} {}".format(
                self.tng_api_key_entry.text(),
                os.path.join(tng_dir, "TNG100-1", "simulation.hdf5"),
                url,
            )
            if self.remote_key_pass_entry.text() != "":
                key = paramiko.RSAKey.from_private_key_file(
                    self.remote_key_entry.text(),
                    password=self.remote_key_pass_entry.text(),
                )
            else:
                key = paramiko.RSAKey.from_private_key_file(
                    self.remote_key_entry.text()
                )
            paramiko_client = paramiko.SSHClient()
            paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            paramiko_client.connect(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                pkey=key,
            )
            stdin, stdout, stderr = paramiko_client.exec_command(cmd)
            self.terminal.add_log(stdout.read().decode())
            self.terminal.add_log(stderr.read().decode())
            self.terminal.add_log("Done.")

    def copy_metadata_on_remote(self):
        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
                private_key_pass=self.remote_key_pass_entry.text(),
            )

        else:
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
            )

        self.input_params.to_csv("input_params.csv", index=False, na_rep="None")
        sftp.put("input_params.csv", self.remote_main_dir + "/input_params.csv")
        os.remove("input_params.csv")

    def copy_settings_on_remote(self):
        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
                private_key_pass=self.remote_key_pass_entry.text(),
            )

        else:
            sftp = pysftp.Connection(
                self.remote_address_entry.text(),
                username=self.remote_user_entry.text(),
                private_key=self.remote_key_entry.text(),
            )
        sftp.put(self.settings_path, self.remote_main_dir + "/settings.plist")

    @staticmethod
    def nan_to_none(value):
        if pd.isna(value):
            return None
        return value

    def run_on_pbs_cluster(self):
        # pbs_config = self.remote_config_entry.text()
        if self.remote_key_pass_entry.text() != "":
            key = paramiko.RSAKey.from_private_key_file(
                self.remote_key_entry.text(), password=self.remote_key_pass_entry.text()
            )
        else:
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text())
        settings_path = os.path.join(self.remote_main_dir, "settings.plist")
        dask_commands = f"""
        cd {self.remote_main_dir}
        source {self.remote_venv_dir}/bin/activate
        """

        _QApplication = QApplication
        python_command = [
            'python -c "import sys; import os; import almasim.ui as ui; ',
            f"app = ui.{_QApplication}(sys.argv); ",
            f"ui.ALMASimulator.settings_file = '{settings_path}'; ",
            "window=ui.ALMASimulator(); ",
            "window.create_pbs_cluster_and_run()",
            'sys.exit(app.exec())"',
        ]
        python_command_str = "".join(python_command)
        # Add it as a new line to dask_commands, and add a newline character
        dask_commands += "\n" + python_command_str
        paramiko_client = paramiko.SSHClient()
        paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        paramiko_client.connect(
            self.remote_address_entry.text(),
            username=self.remote_user_entry.text(),
            pkey=key,
        )
        stdin, stdout, stderr = paramiko_client.exec_command(dask_commands)
        self.terminal.add_log(stdout.read().decode())
        self.terminal.add_log(stderr.read().decode())

    def run_on_mpi_machine(self):
        # slurm_config = self.remote_config_entry.text()
        if self.remote_key_pass_entry.text() != "":
            key = paramiko.RSAKey.from_private_key_file(
                self.remote_key_entry.text(), password=self.remote_key_pass_entry.text()
            )
        else:
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text())

        settings_path = os.path.join(self.remote_main_dir, "settings.plist")
        dask_commands = f"""
            cd {self.remote_main_dir}
            source {self.remote_venv_dir}/bin/activate
            export QT_QPA_PLATFORM=offscreen
        """
        # Separate the Python command for readability and maintainability
        python_command = (
            'python -c "import sys; import os; import almasim.ui as ui; '
            f"app = ui.QApplication(sys.argv); "
            f"ui.ALMASimulator.settings_file = '{settings_path}'; "
            "window=ui.ALMASimulator(); "
            "ui.ALMASimulator.initiate_parallel_simulation_remote(window); "
            'sys.exit(app.exec())"'
        )
        # Join the list elements into a single string
        python_command_str = "".join(python_command)
        # Add it as a new line to dask_commands, and add a newline character
        dask_commands += "\n" + python_command_str
        exclude_pattern = re.compile(
            r"""
            # Pattern 1: Specific command lines
            (^cd /home/astro/ALMASim$)
            |(source /home/astro/almasim_env/bin/activate$)
            |(export QT_QPA_PLATFORM=offscreen$)
            |(python -c "import sys.*initiate_parallel_simulation_remote\(window\);)
            |(sys.exit\(app.exec\(\)"\s*$)
            # Pattern 2: ANSI escape codes
            |(\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]))
        """,
            re.VERBOSE,
        )
        paramiko_client = paramiko.SSHClient()
        paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        paramiko_client.connect(
            self.remote_address_entry.text(),
            username=self.remote_user_entry.text(),
            pkey=key,
        )
        channel = paramiko_client.invoke_shell()

        # Continuously read and display output
        def read_output():
            while True:
                if channel.recv_ready():
                    output = channel.recv(1024).decode()
                    # Filter out unwanted lines
                    filtered_output = ""
                    for line in output.splitlines():
                        if not exclude_pattern.search(line):
                            filtered_output += line + "\n"
                    if (
                        filtered_output
                    ):  # Only add to the log if there's filtered output
                        self.terminal.add_log(filtered_output)
                if channel.exit_status_ready():
                    break

        output_thread = threading.Thread(target=read_output)
        output_thread.start()
        channel.send(dask_commands + "\n")

        # Wait for the command to finish
        output_thread.join()
        paramiko_client.close()
        output_thread.close()

    def run_on_slurm_cluster(self):
        # slurm_config = self.remote_config_entry.text()
        if self.remote_key_pass_entry.text() != "":
            key = paramiko.RSAKey.from_private_key_file(
                self.remote_key_entry.text(), password=self.remote_key_pass_entry.text()
            )
        else:
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text())

        settings_path = os.path.join(self.remote_main_dir, "settings.plist")
        dask_commands = f"""
            cd {self.remote_main_dir}
            source {self.remote_venv_dir}/bin/activate
            export QT_QPA_PLATFORM=offscreen
        """
        _QApplication = QApplication
        python_command = [
            'python -c "import sys; import os; import almasim.ui as ui; ',
            f"app = ui.{_QApplication}(sys.argv); ",
            f"ui.ALMASimulator.settings_file = '{settings_path}'; ",
            "window=ui.ALMASimulator(); ",
            "ui.ALMASimulator.initiate_slurm_simulation_remote(window); ",
            'sys.exit(app.exec())"',
        ]
        # Join the list elements into a single string
        python_command_str = "".join(python_command)
        # Add it as a new line to dask_commands, and add a newline character
        dask_commands += "\n" + python_command_str
        exclude_pattern = re.compile(
            r"""
            # Pattern 1: Specific command lines
            (^cd /home/astro/ALMASim$)
            |(source /home/astro/almasim_env/bin/activate$)
            |(export QT_QPA_PLATFORM=offscreen$)
            |(python -c "import sys.*initiate_slurm_simulation_remote\(window\);)
            |(sys.exit\(app.exec\(\)"\s*$)
            # Pattern 2: ANSI escape codes
            |(\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]))
        """,
            re.VERBOSE,
        )
        paramiko_client = paramiko.SSHClient()
        paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        paramiko_client.connect(
            self.remote_address_entry.text(),
            username=self.remote_user_entry.text(),
            pkey=key,
        )
        channel = paramiko_client.invoke_shell()

        # Continuously read and display output
        def read_output():
            while True:
                if channel.recv_ready():
                    output = channel.recv(1024).decode()
                    # Filter out unwanted lines
                    filtered_output = ""
                    for line in output.splitlines():
                        if not exclude_pattern.search(line):
                            filtered_output += line + "\n"
                    if (
                        filtered_output
                    ):  # Only add to the log if there's filtered output
                        self.terminal.add_log(filtered_output)
                if channel.exit_status_ready():
                    break

        output_thread = threading.Thread(target=read_output)
        output_thread.start()
        channel.send(dask_commands + "\n")

        # Wait for the command to finish
        output_thread.join()
        paramiko_client.close()
        output_thread.close()

    def transform_source_type_label(self):
        if self.model_combo.currentText() == "Galaxy Zoo":
            self.source_type = "galaxy-zoo"
        elif self.model_combo.currentText() == "Hubble 100":
            self.source_type = "hubble-100"
        elif self.model_combo.currentText() == "Molecular":
            self.source_type = "molecular"
        elif self.model_combo.currentText() == "Diffuse":
            self.source_type = "diffuse"
        elif self.model_combo.currentText() == "Gaussian":
            self.source_type = "gaussian"
        elif self.model_combo.currentText() == "Point":
            self.source_type = "point"
        elif self.model_combo.currentText() == "Extended":
            self.source_type = "extended"

    def sample_given_redshift(self, metadata, n, rest_frequency, extended, zmax=None):
        pd.options.mode.chained_assignment = None
        if isinstance(rest_frequency, np.ndarray) or isinstance(rest_frequency, list):
            rest_frequency = np.sort(np.array(rest_frequency))
        else:
            rest_frequency = np.array([rest_frequency])

        if self.terminal is not None:
            max_freq = np.max(metadata["Freq"].values)
            self.terminal.add_log(f"Max frequency recorded in metadata: {max_freq} GHz")
            min_freq = np.min(metadata["Freq"].values)
            self.terminal.add_log(f"Min frequency recorded in metadata: {min_freq} GHz")
            self.terminal.add_log("Filtering metadata based on line catalogue...")
        if self.terminal is not None:
            self.terminal.add_log(f"Remaining metadata: {len(metadata)}")
        freqs = metadata["Freq"].values
        closest_rest_frequencies = []
        for freq in freqs:
            # Calculate the absolute difference between the freq and all rest_frequencies
            differences = rest_frequency - freq
            # if the difference is negative, set it to a large number
            differences[differences < 0] = 1e10
            # Find the index of the minimum difference
            index_min = np.argmin(differences)
            # Append the closest rest frequency to the list
            closest_rest_frequencies.append(rest_frequency[index_min])
        rest_frequencies = np.array(closest_rest_frequencies)

        redshifts = [
            uas.compute_redshift(rest_frequency * U.GHz, source_freq * U.GHz)
            for source_freq, rest_frequency in zip(freqs, rest_frequencies)
        ]
        metadata.loc[:, "redshift"] = redshifts
        snapshots = [
            uas.redshift_to_snapshot(redshift)
            for redshift in metadata["redshift"].values
        ]
        metadata["rest_frequency"] = rest_frequencies
        n_metadata = 0
        z_save = zmax
        self.terminal.add_log("Computing redshifts")
        while n_metadata < ceil(n / 10):
            s_metadata = n_metadata
            if zmax is not None:
                f_metadata = metadata[
                    (metadata["redshift"] <= zmax) & (metadata["redshift"] >= 0)
                ]
            else:
                f_metadata = metadata[metadata["redshift"] >= 0]
            n_metadata = len(f_metadata)
            if n_metadata == s_metadata:
                zmax += 0.1
        if zmax is not None:
            metadata = metadata[
                (metadata["redshift"] <= zmax) & (metadata["redshift"] >= 0)
            ]
        else:
            metadata = metadata[metadata["redshift"] >= 0]
        if z_save != zmax:
            if self.terminal is not None:
                self.terminal.add_log(
                    f"Max redshift has been adjusted fit metadata,\
                         new max redshift: {round(zmax, 3)}"
                )
        if self.terminal is not None:
            self.terminal.add_log(f"Remaining metadata: {len(metadata)}")
        snapshots = [
            uas.redshift_to_snapshot(redshift)
            for redshift in metadata["redshift"].values
        ]
        metadata["snapshot"] = snapshots
        if extended is True:
            # metatada = metadata[metadata['redshift'] < 0.05]
            metadata = metadata[
                (metadata["snapshot"] == 99) | (metadata["snapshot"] == 95)
            ]
        sample = metadata.sample(n, replace=True)
        return sample

    def remove_non_numeric(self, text):
        """Removes non-numeric characters from a string.
        Args:
            text: The string to process.

        Returns:
            A new string containing only numeric characters and the decimal point (.).
        """
        numbers = "0123456789."
        return "".join(char for char in text if char in numbers)

    @staticmethod
    def closest_power_of_2(x):
        op = math.floor if bin(x)[3] != "1" else math.ceil
        return 2 ** op(math.log(x, 2))

    def freq_supp_extractor(self, freq_sup, obs_freq):
        freq_band, n_channels, freq_mins, freq_maxs, freq_ds = [], [], [], [], []
        freq_sup = freq_sup.split("U")
        for i in range(len(freq_sup)):
            sup = freq_sup[i][1:-1].split(",")
            sup = [su.split("..") for su in sup][:2]
            freq_min, freq_max = float(self.remove_non_numeric(sup[0][0])), float(
                self.remove_non_numeric(sup[0][1])
            )
            freq_d = float(self.remove_non_numeric(sup[1][0]))
            freq_min = freq_min * U.GHz
            freq_max = freq_max * U.GHz
            freq_d = freq_d * U.kHz
            freq_d = freq_d.to(U.GHz)
            freq_b = freq_max - freq_min
            n_chan = int(freq_b / freq_d)
            freq_band.append(freq_b)
            n_channels.append(n_chan)
            freq_mins.append(freq_min)
            freq_maxs.append(freq_max)
            freq_ds.append(freq_d)
        freq_ranges = np.array(
            [[freq_mins[i].value, freq_maxs[i].value] for i in range(len(freq_mins))]
        )
        idx_ = np.argwhere(
            (obs_freq.value >= freq_ranges[:, 0])
            & (obs_freq.value <= freq_ranges[:, 1])
        )[0][0]
        freq_range = freq_ranges[idx_]
        band_range = freq_range[1] - freq_range[0]
        n_channels = n_channels[idx_]
        central_freq = freq_range[0] + band_range / 2
        freq_d = freq_ds[idx_]
        return band_range * U.GHz, central_freq * U.GHz, n_channels, freq_d

    # -------- Simulation Functions ------------------------
    def start_simulation(self):
        # Implement the logic to start the simulation
        if self.local_mode_combo.currentText() == "local":
            self.terminal.add_log("Starting simulation on your local machine")
        else:
            self.terminal.add_log(
                f"Starting simulation on {self.remote_address_entry.text()}"
            )
            self.remote_simulation_finished = False
        n_sims = int(self.n_sims_entry.text())
        sim_idxs = np.arange(n_sims)
        self.transform_source_type_label()
        source_types = np.array([self.source_type] * n_sims)
        self.output_path = os.path.join(
            self.output_entry.text(), self.project_name_entry.text()
        )
        plot_path = os.path.join(self.output_path, "plots")
        # Output Directory
        if self.local_mode_combo.currentText() == "local":
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
        else:
            self.create_remote_output_dir()

        output_paths = np.array([self.output_path] * n_sims)
        tng_paths = np.array([self.tng_entry.text()] * n_sims)

        # Galaxy Zoo Directory
        if self.local_mode_combo.currentText() == "local":
            if self.model_combo.currentText() == "Galaxy Zoo":
                if self.galaxy_zoo_entry.text() and not os.path.exists(
                    os.path.join(self.galaxy_zoo_entry.text(), "images_gz2")
                ):
                    self.terminal.add_log("Downloading Galaxy Zoo")
                    # pool = QThreadPool.globalInstance()
                    runnable = DownloadGalaxyZooRunnable(self)
                    self.thread_pool.start(runnable)
            if self.model_combo.currentText() == "Hubble 100":
                if self.hubble_entry.text() and not os.path.exists(
                    os.path.join(self.hubble_entry.text(), "top100")
                ):

                    self.terminal.add_log("Downloading Hubble Images")
                    # pool = QThreadPool.globalInstance()
                    runnable = DownloadHubbleRunnable(self)
                    self.thread_pool.start(runnable)
        else:
            self.create_remote_environment()
            self.download_galaxy_zoo_on_remote()
            self.download_hubble_on_remote()

        galaxy_zoo_paths = np.array([self.galaxy_zoo_entry.text()] * n_sims)
        hubble_paths = np.array([self.hubble_entry.text()] * n_sims)
        if self.local_mode_combo.currentText() == "local":
            main_paths = np.array([self.main_path] * n_sims)
        else:
            main_paths = np.array(
                [
                    os.path.join(
                        "/home/{}/".format(self.remote_user_entry.text()),
                        "ALMASim/almasim/",
                    )
                ]
                * n_sims
            )
        ncpus = np.array([int(self.ncpu_entry.text())] * n_sims)
        project_names = np.array([self.project_name_entry.text()] * n_sims)
        save_mode = np.array([self.save_format_combo.currentText()] * n_sims)
        self.db_line = uas.read_line_emission_csv(
            os.path.join(self.main_path, "brightnes", "calibrated_lines.csv"),
            sep=",",
        )
        # parameter for c generations for artificial lines
        self.line_cs_mean = np.mean(self.db_line["c"].values)
        self.line_cs_std = np.std(self.db_line["c"].values)
        # Checking Line Mode
        if self.line_mode_checkbox.isChecked():
            line_indices = [int(i) for i in self.line_index_entry.text().split()]
            rest_freq, line_names = uas.get_line_info(self.main_path, line_indices)
            self.terminal.add_log("# ------------------------------------- #\n")
            self.terminal.add_log("The following lines have been selected\n")
            for line_name, r_freq in zip(line_names, rest_freq):
                self.terminal.add_log(f"Line: {line_name}: {r_freq} GHz")
            self.terminal.add_log("# ------------------------------------- #\n")
            if len(rest_freq) == 1:
                rest_freq = rest_freq[0]
            rest_freqs = np.array([rest_freq] * n_sims)
            redshifts = np.array([None] * n_sims)
            n_lines = np.array([None] * n_sims)
            line_names = np.array([line_names] * n_sims)
            z1 = None
        else:
            if self.redshift_entry.text() != "":
                redshifts = [float(z) for z in self.redshift_entry.text().split()]
                if len(redshifts) == 1:
                    redshifts = np.array([redshifts[0]] * n_sims)
                    z0, z1 = float(redshifts[0]), float(redshifts[0])
                else:
                    z0, z1 = float(redshifts[0]), float(redshifts[1])
                    redshifts = np.random.uniform(z0, z1, n_sims)
                n_lines = np.array([int(self.num_lines_entry.text())] * n_sims)
                rest_freq, _ = uas.get_line_info(self.main_path)
                rest_freqs = np.array([None] * n_sims)
                line_names = np.array([None] * n_sims)
            else:
                if self.terminal is not None:
                    self.terminal.add_log(
                        'Please fill the redshift and n lines fields or check "Line Mode"'
                    )
                return

        # Checking Infrared Luminosity
        if self.ir_luminosity_checkbox.isChecked():
            lum_infrared = [
                float(lum) for lum in self.ir_luminosity_entry.text().split()
            ]
            if len(lum_infrared) == 1:
                lum_ir = np.array([lum_infrared[0]] * n_sims)
            else:
                lum_ir = np.random.uniform(lum_infrared[0], lum_infrared[1], n_sims)
        else:
            lum_ir = np.array([None] * n_sims)

        # Checking SNR
        if self.snr_checkbox.isChecked():
            snr = [float(snr) for snr in self.snr_entry.text().split()]
            if len(snr) == 1:
                snr = np.array([snr[0]] * n_sims)
            else:
                snr = np.random.uniform(snr[0], snr[1], n_sims)
        else:
            snr = np.ones(n_sims)

        # Checking Number of Pixesl
        if self.fix_spatial_checkbox.isChecked():
            n_pixs = np.array([int(self.n_pix_entry.text())] * n_sims)
        else:
            n_pixs = np.array([None] * n_sims)

        # Checking Number of Channels
        if self.fix_spectral_checkbox.isChecked():
            n_channels = np.array([int(self.n_channels_entry.text())] * n_sims)
        else:
            n_channels = np.array([None] * n_sims)
        if self.model_combo.currentText() == "Extended":
            if self.local_mode_combo.currentText() == "local":
                self.terminal.add_log("Checking TNG Directories")
                # pool = QThreadPool.globalInstance()
                runnable = DownloadTNGStructureRunnable(self)
                self.thread_pool.start(runnable)
            else:
                self.remote_check_tng_dirs()
            tng_apis = np.array([self.tng_api_key_entry.text()] * n_sims)
            self.metadata = self.sample_given_redshift(
                self.metadata, n_sims, rest_freq, True, z1
            )
        else:
            tng_apis = np.array([None] * n_sims)
            self.metadata = self.sample_given_redshift(
                self.metadata, n_sims, rest_freq, False, z1
            )
        source_names = self.metadata["ALMA_source_name"].values
        member_ouids = self.metadata["member_ous_uid"].values
        ras = self.metadata["RA"].values
        decs = self.metadata["Dec"].values
        bands = self.metadata["Band"].values
        ang_ress = self.metadata["Ang.res."].values
        vel_ress = self.metadata["Vel.res."].values
        fovs = self.metadata["FOV"].values
        obs_dates = self.metadata["Obs.date"].values
        pwvs = self.metadata["PWV"].values
        int_times = self.metadata["Int.Time"].values
        bandwidths = self.metadata["Bandwidth"].values
        freqs = self.metadata["Freq"].values
        freq_supports = self.metadata["Freq.sup."].values
        antenna_arrays = self.metadata["antenna_arrays"].values
        cont_sens = self.metadata["Cont_sens_mJybeam"].values
        self.terminal.add_log("Metadata retrived successfully\n")
        if self.serendipitous_checkbox.isChecked():
            inject_serendipitous = np.array([True] * n_sims)
        else:
            inject_serendipitous = np.array([False] * n_sims)
        if self.local_mode_combo.currentText() == "local":
            remote = np.array([False] * n_sims)
        else:
            remote = np.array([True] * n_sims)
        self.input_params = pd.DataFrame(
            zip(
                sim_idxs,
                source_names,
                member_ouids,
                main_paths,
                output_paths,
                tng_paths,
                galaxy_zoo_paths,
                hubble_paths,
                project_names,
                ras,
                decs,
                bands,
                ang_ress,
                vel_ress,
                fovs,
                obs_dates,
                pwvs,
                int_times,
                bandwidths,
                freqs,
                freq_supports,
                cont_sens,
                antenna_arrays,
                n_pixs,
                n_channels,
                source_types,
                tng_apis,
                ncpus,
                rest_freqs,
                redshifts,
                lum_ir,
                snr,
                n_lines,
                line_names,
                save_mode,
                inject_serendipitous,
                remote,
            ),
            columns=[
                "idx",
                "source_name",
                "member_ouid",
                "main_path",
                "output_dir",
                "tng_dir",
                "galaxy_zoo_dir",
                "hubble_dir",
                "project_name",
                "ra",
                "dec",
                "band",
                "ang_res",
                "vel_res",
                "fov",
                "obs_date",
                "pwv",
                "int_time",
                "bandwidth",
                "freq",
                "freq_support",
                "cont_sens",
                "antenna_array",
                "n_pix",
                "n_channels",
                "source_type",
                "tng_api_key",
                "ncpu",
                "rest_frequency",
                "redshift",
                "lum_infrared",
                "snr",
                "n_lines",
                "line_names",
                "save_mode",
                "inject_serendipitous",
                "remote",
            ],
        )
        if self.local_mode_combo.currentText() == "remote":
            self.copy_metadata_on_remote()
            self.copy_settings_on_remote()
        if self.comp_mode_combo.currentText() == "parallel":
            if self.local_mode_combo.currentText() == "local":
                self.initiate_parallel_simulation()
            else:
                if self.remote_mode_combo.currentText() == "SLURM":
                    self.run_on_slurm_cluster()
                elif self.remote_mode_combo.currentText() == "PBS":
                    self.run_on_pbs_cluster()
                elif self.remote_mode_combo.currentText() == "MPI":
                    thread = threading.Thread(
                        target=self.run_on_mpi_machine, daemon=True
                    )
                    thread.start()
                else:
                    self.terminal.add_log("Please select a valid remote mode")
        else:
            if self.local_mode_combo.currentText() == "local":
                self.run_simulator_sequentially()
            else:
                self.terminal.add_log(
                    "Cannot run on remote in sequential mode, changing it to parallel"
                )
                self.comp_mode_combo.setCurrentText("parallel")
        os.chdir(self.main_path)

    def stop_simulation(self):
        # Implement the logic to stop the simulation
        self.stop_simulation_flag = True
        self.progress_bar_entry.setText("Simulation Stopped")
        self.update_progress_bar(0)
        self.terminal.add_log("# ------------------------------------- #\n")

    def run_simulator_sequentially(self):
        self.stop_simulation_flag = False
        self.current_sim_index = 0
        self.nextSimulation.connect(self.run_next_simulation)
        self.run_next_simulation()

    def run_next_simulation(self):
        if self.current_sim_index >= int(self.n_sims_entry.text()):
            self.progress_bar_entry.setText("Simluation Finished")
            # self.send_email()
            return
        # pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(int(self.ncpu_entry.text()))
        runnable = SimulatorRunnable(
            self, *self.input_params.iloc[self.current_sim_index]
        )
        self.update_progress.connect(self.update_progress_bar)
        # runnable.signals.simulationFinished.connect(self.plot_simulation_results)
        runnable.signals.simulationFinished.connect(self.start_plot_runnable)
        runnable.signals.simulationFinished.connect(self.nextSimulation.emit)
        self.thread_pool.start(runnable)
        self.current_sim_index += 1
        # for i in range(int(self.n_sims_entry.text())):
        #    runnable = SimulatorRunnable(self, *self.input_params.iloc[i])
        #    self.update_progress.connect(self.update_progress_bar)
        #    runnable.signals.simulationFinished.connect(
        #        self.plot_simulation_results
        #    )  # Connect the signal
        #    pool.start(runnable)

    def run_simulator_parallel_remote(self, input_params):
        # Access instance attributes here using `self`
        self.output_path = os.path.join(
            self.output_entry.text(), self.project_name_entry.text()
        )
        dask.config.set({"temporary_directory": self.output_path})
        total_memory = psutil.virtual_memory().total
        num_workers = int(self.ncpu_entry.text()) // 4
        memory_limit = int(0.9 * total_memory / num_workers)

        ddf = dd.from_pandas(input_params, npartitions=num_workers)
        with LocalCluster(
            n_workers=num_workers, threads_per_worker=4, dashboard_address=None
        ) as cluster:
            with Client(cluster) as client:
                client.register_plugin(MemoryLimitPlugin(memory_limit))
                futures = []

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    for df in ddf.partitions:
                        worker = SimulatorWorker(self, df)
                        # Connect signals using the instance ('self')
                        self.update_progress.connect(self.update_progress_bar)
                        worker.signals.simulationFinished.connect(
                            self.plot_simulation_results
                        )
                        futures.append(executor.submit(worker.run))

                for future in futures:
                    future.result()
            client.close()
        cluster.close()
        self.remote_simulation_finished = True
        # self.send_email()

    def run_simulator_slurm_remote(self, input_params):
        self.output_path = os.path.join(
            self.output_entry.text(), self.project_name_entry.text()
        )
        with open("slurm_config.json", "r") as f:
            config = json.load(f)
        with SLURMCluster(
            queue=config["queue"],
            account=config["account"],
            cores=config["cores"],
            memory=config["memory"],
            job_extra_directives=config["job_extra"],
        ) as cluster:
            with Client(cluster) as client:

                self.terminal.add_log(
                    "Dashboard Link: {}".format(client.dashboard_link)
                )
                self.terminal.add_log(
                    "Workers: {}".format(len(client.scheduler_info()["workers"]))
                )
                self.terminal.add_log(
                    "Total threads: {}".format(
                        sum(
                            w["nthreads"]
                            for w in client.scheduler_info()["workers"].values()
                        )
                    )
                )
                self.terminal.add_log(
                    "Total memory: {}".format(
                        sum(
                            w["memory_limit"]
                            for w in client.scheduler_info()["workers"].values()
                        )
                    )
                )
                cluster.scale(jobs=int(int(self.ncpu_entry.text()) // 4))
                ddf = dd.from_pandas(input_params, npartitions=cluster.n_workers)
                futures = []
                with ThreadPoolExecutor(max_workers=cluster.n_workers) as executor:
                    for df in ddf.partitions:
                        worker = SimulatorWorker(self, df)
                        self.update_progress.connect(self.update_progress_bar)
                        worker.signals.simulationFinished.connect(
                            self.plot_simulation_results
                        )
                        futures.append(executor.submit(worker.run))

                for future in futures:
                    future.result()
            client.close()
        cluster.close()

    def run_simulator_parallel(self):
        dask.config.set({"temporary_directory": self.output_path})
        total_memory = psutil.virtual_memory().total
        num_workers = int(self.ncpu_entry.text()) // 4
        memory_limit = int(0.9 * total_memory / num_workers)

        ddf = dd.from_pandas(self.input_params, npartitions=num_workers)
        with LocalCluster(
            n_workers=num_workers, threads_per_worker=4, dashboard_address=None
        ) as cluster:
            with Client(cluster) as client:
                client.register_plugin(MemoryLimitPlugin(memory_limit))
                futures = []
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    for df in ddf.partitions:
                        worker = SimulatorWorker(self, df)
                        self.update_progress.connect(self.update_progress_bar)
                        worker.signals.simulationFinished.connect(
                            self.plot_simulation_results
                        )
                        futures.append(executor.submit(worker.run))

                # Optionally wait for all workers to complete before proceeding
                for future in futures:
                    future.result()  # This blocks until the worker is done
        # self.send_email()

    def initiate_parallel_simulation(self):
        # pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(int(self.ncpu_entry.text()))
        runnable = ParallelSimulatorRunnable(self)
        self.thread_pool.start(runnable)

    @classmethod
    def initiate_parallel_simulation_remote(cls, window_instance):
        input_params = pd.read_csv("input_params.csv")
        # pool = QThreadPool.globalInstance()
        runnable = ParallelSimulatorRunnableRemote(window_instance, input_params)
        window_instance.thread_pool.start(runnable)

    @classmethod
    def initialize_slurm_simulation_remote(cls, window_instance):
        input_params = pd.read_csv("input_params.csv")
        pool = QThreadPool.globalInstance()
        runnable = SlurmSimulatorRunnableRemote(window_instance, input_params)
        pool.start(runnable)

    def cont_finder(self, cont_frequencies, line_frequency):
        # cont_frequencies=sed['GHz'].values
        distances = np.abs(
            cont_frequencies - np.ones(len(cont_frequencies)) * line_frequency
        )
        return np.argmin(distances)

    def normalize_sed(
        self,
        sed,
        lum_infrared,
        solid_angle,
        cont_sens,
        freq_min,
        freq_max,
        remote=False,
    ):
        so_to_erg_s = 3.846e33  # Solar luminosity to erg/s -XX
        lum_infrared_erg_s = lum_infrared * so_to_erg_s  # luminosity in erg/s -XX
        sed["Jy"] = lum_infrared_erg_s * sed["erg/s/Hz"] * 1e23 / solid_angle
        cont_mask = (sed["GHz"].values >= freq_min) & (sed["GHz"].values <= freq_max)
        if sum(cont_mask) > 0:
            cont_fluxes = sed["Jy"].values[cont_mask]
            min_ = np.min(cont_fluxes)
        else:
            freq_point = np.argmin(np.abs(sed["GHz"].values - freq_min))
            cont_fluxes = sed["Jy"].values[freq_point]
            min_ = cont_fluxes
        if remote is True:
            print("Minimum continum flux: {:.2e}".format(min_))
            print("Continum sensitivity: {:.2e}".format(cont_sens))
        else:
            self.terminal.add_log("Minimum continum flux: {:.2e}".format(min_))
            self.terminal.add_log("Continum sensitivity: {:.2e}".format(cont_sens))
        lum_save = lum_infrared

        if min_ < cont_sens:
            while min_ < cont_sens:
                lum_infrared += 0.1 * lum_infrared
                lum_infrared_erg_s = so_to_erg_s * lum_infrared
                sed["Jy"] = lum_infrared_erg_s * sed["erg/s/Hz"] * 1e23 / solid_angle
                cont_mask = (sed["GHz"] >= freq_min) & (sed["GHz"] <= freq_max)
                if sum(cont_mask) > 0:
                    cont_fluxes = sed["Jy"].values[cont_mask]
                    min_ = np.min(cont_fluxes)
                else:
                    freq_point = np.argmin(np.abs(sed["GHz"].values - freq_min))
                    cont_fluxes = sed["Jy"].values[freq_point]
                    min_ = cont_fluxes

        if lum_save != lum_infrared:
            if remote is True:
                print(
                    "To observe the source, luminosity has been set to {:.2e}".format(
                        lum_infrared
                    )
                )
                print("# ------------------------------------- #\n")
            else:
                self.terminal.add_log(
                    "To observe the source, luminosity has been set to {:.2e}".format(
                        lum_infrared
                    )
                )
                self.terminal.add_log("# ------------------------------------- #\n")
        return sed, lum_infrared_erg_s, lum_infrared

    def sed_reading(
        self,
        type_,
        path,
        cont_sens,
        freq_min,
        freq_max,
        remote,
        lum_infrared=None,
        redshift=None,
    ):
        cosmo = FlatLambdaCDM(H0=70 * U.km / U.s / U.Mpc, Tcmb0=2.725 * U.K, Om0=0.3)
        if (
            type_ == "extended"
            or type_ == "diffuse"
            or type_ == "molecular"
            or type_ == "galaxy-zoo"
            or type_ == "hubble-100"
        ):
            file_path = os.path.join(path, "SED_low_z_warm_star_forming_galaxy.dat")
            if redshift is None:
                redshift = 10 ** (-4)
            if lum_infrared is None:
                lum_infrared = 1e12  # luminosity in solar luminosities
        elif type_ == "point" or type_ == "gaussian":
            file_path = os.path.join(path, "SED_low_z_type2_AGN.dat")
            if redshift is None:
                redshift = 0.05
            if lum_infrared is None:
                lum_infrared = 1e12  # luminosity in solar luminosities
        else:
            return "Not valid type"
        # L (erg/s/Hz) = 4 pi d^2(cm) * 10^-23 Flux (Jy)
        #  Flux (Jy) =L (erg/s/Hz) * 10^23 /  * 4 pi d^2(cm)
        # To normalize we multiply by lum_infrared_jy
        distance_Mpc = cosmo.luminosity_distance(redshift).value  # distance in Mpc
        Mpc_to_cm = 3.086e24  # Mpc to cm
        distance_cm = distance_Mpc * Mpc_to_cm  # distance in cm  -XX
        solid_angle = 4 * pi * distance_cm**2  # solid angle in cm^2 -XX
        # Load the SED
        sed = pd.read_csv(file_path, sep=r"\s+")
        # Convert to GHz
        sed["GHz"] = sed["um"].apply(
            lambda x: (x * U.um).to(U.GHz, equivalencies=U.spectral()).value
        )
        # Re normalize the SED and convert to Jy from erg/s/Hz
        sed, lum_infrared_erg_s, lum_infrared = self.normalize_sed(
            sed, lum_infrared, solid_angle, cont_sens, freq_min, freq_max, remote
        )
        #  Flux (Jy) =L (erg/s/Hz) * 10^23 /  * 4 pi d^2(cm)
        flux_infrared = lum_infrared_erg_s * 1e23 / solid_angle  # Jy * Hz
        # flux_infrared_jy = flux_infrared  / (sed['GHz'].values *
        # U.GHz).to(U.Hz).value  # Jy
        sed.drop(columns=["um", "erg/s/Hz"], inplace=True)
        sed = sed.sort_values(by="GHz", ascending=True)
        return sed, flux_infrared, lum_infrared

    def find_compatible_lines(
        self,
        db_line,
        source_freq,
        redshift,
        n,
        line_names,
        freq_min,
        freq_max,
        band_range,
    ):
        """
        Found the lines at given configuration, if real lines are not possibile, it will
        generate fakes lines to reach the desidered number of lines.

        Parameter:
        db_line (pandas.Dataframe): It is the database of lines from which the user can
            choose real ones.
        redshift (float) : Redshift value of the source
        n (int): Number of lines that want to simulate
        line_names (str):
        freq_min, freq_max (float) : Minimum frequency and Maximum frequency of the source
        band_range : The band range around the central frequency

        Return:
        compatible_lines (pandas.Dataframe) : Dataframe with n lines that will be
        simulated.
        """
        c_km_s = c.to(U.km / U.s)
        min_delta_v = float(self.min_line_width_slider.value())
        max_delta_v = float(self.max_line_width_slider.value())
        db_line = db_line.copy()
        db_line["redshift"] = (db_line["freq(GHz)"].values - source_freq) / source_freq
        db_line = db_line.loc[~((db_line["redshift"] < 0) | (db_line["redshift"] > 20))]
        delta_v = np.random.uniform(min_delta_v, max_delta_v, len(db_line)) * U.km / U.s
        db_line["shifted_freq(GHz)"] = db_line["freq(GHz)"] / (1 + db_line["redshift"])
        fwhms = (
            0.84
            * (db_line["shifted_freq(GHz)"].values * (delta_v / c_km_s) * 1e9)
            * U.Hz
        )
        fwhms_GHz = fwhms.to(U.GHz).value
        db_line["fwhm_GHz"] = fwhms_GHz
        found_lines = 0
        i = 0
        lines_fitted, lines_fitted_redshifts = [], []
        if redshift is not None:
            db_line["redshift_distance"] = np.abs(db_line["redshift"] - redshift)
            db_line = db_line.sort_values(by="redshift_distance")
        for i in range(len(db_line)):
            db = db_line.copy()
            first_line = db.iloc[i]
            db["shifted_freq(GHz)"] = db["freq(GHz)"] / (1 + first_line["redshift"])
            db["distance(GHz)"] = abs(
                db["shifted_freq(GHz)"] - first_line["shifted_freq(GHz)"]
            )
            compatible_lines = db.loc[db["distance(GHz)"] < band_range]
            compatible_lines.loc[:, "redshift"] = (
                np.ones(len(compatible_lines)) * first_line["redshift"]
            )
            found_lines = len(compatible_lines)
            lines_fitted.append(found_lines)
            lines_fitted_redshifts.append(first_line["redshift"])
            i += 1
        if redshift is None:
            found_lines = np.max(lines_fitted)
        else:
            found_lines = np.argmin(np.abs(np.array(lines_fitted_redshifts) - redshift))

        if found_lines < n:
            if redshift is None:
                i = np.argmax(lines_fitted)
            else:
                i = np.argmin(np.abs(np.array(lines_fitted_redshifts) - redshift))
            first_line = db_line.iloc[i]
            db_line["shifted_freq(GHz)"] = db_line["freq(GHz)"] / (
                1 + first_line["redshift"]
            )
            db_line["distance(GHz)"] = abs(
                db_line["shifted_freq(GHz)"] - first_line["shifted_freq(GHz)"]
            )
            compatible_lines = db_line.loc[db_line["distance(GHz)"] < band_range]
            compatible_lines.loc[:, "redshift"] = first_line["redshift"]
            found_lines = len(compatible_lines)
            if found_lines > 1:
                mean, std = np.mean(compatible_lines["freq(GHz)"]), np.std(
                    compatible_lines["freq(GHz)"]
                )
            else:
                mean = np.mean(compatible_lines["freq(GHz)"])
                std = np.random.uniform(0.1, 0.3) * band_range
            freqs = np.array(list(np.random.normal(mean, std, n - found_lines)))
            if found_lines > 1:
                mean, std = np.mean(compatible_lines["c"]), np.std(
                    compatible_lines["c"]
                )
            else:
                mean = self.line_cs_mean
                std = self.line_cs_std
            cs = np.array(list(np.random.normal(mean, std, n - found_lines)))
            mean, std = np.mean(compatible_lines["err_c"]), np.std(
                compatible_lines["err_c"]
            )
            err_cs = np.array(list(np.random.normal(mean, std, n - found_lines)))
            line_names = np.array([f"fake_line {i}" for i in range(n - found_lines)])
            redshifts = np.array(list(np.ones(len(freqs)) * first_line["redshift"]))
            shifted_freqs = np.array(freqs / (1 + first_line["redshift"]))
            distances = np.array(abs(shifted_freqs - first_line["shifted_freq(GHz)"]))
            fwhms = np.array(
                list(np.ones(len(line_names)) * first_line["fwhm_GHz"].astype(float))
            )
            if redshift is None:
                data = np.column_stack(
                    (
                        line_names,
                        np.round(freqs, 2).astype(float),
                        np.round(cs, 2).astype(float),
                        np.round(err_cs, 2).astype(float),
                        np.round(redshifts, 6).astype(float),
                        np.round(shifted_freqs, 6).astype(float),
                        np.round(fwhms, 6).astype(float),
                        np.round(distances, 6).astype(float),
                    )
                )
            else:
                redshift_distance = np.array(
                    list(
                        np.ones(len(line_names))
                        * first_line["redshift_distance"].astype(float)
                    )
                )
                data = np.column_stack(
                    (
                        line_names,
                        np.round(freqs, 2).astype(float),
                        np.round(cs, 2).astype(float),
                        np.round(err_cs, 2).astype(float),
                        np.round(redshifts, 6).astype(float),
                        np.round(shifted_freqs, 6).astype(float),
                        np.round(fwhms, 6).astype(float),
                        redshift_distance,
                        np.round(distances, 6).astype(float),
                    )
                )
            fake_db = pd.DataFrame(data=data, columns=db_line.columns)
            for col in fake_db.columns[1:]:
                fake_db[col] = pd.to_numeric(fake_db[col])
            compatible_lines = pd.concat(
                (compatible_lines, fake_db),
                ignore_index=True,
            )
        elif found_lines > n:
            compatible_lines = compatible_lines.iloc[:n]
        compatible_lines = compatible_lines.reset_index(drop=True)
        for index, row in compatible_lines.iterrows():
            lower_bound, upper_bound = (
                row["shifted_freq(GHz)"] - row["fwhm_GHz"] / 2,
                row["shifted_freq(GHz)"] + row["fwhm_GHz"] / 2,
            )
            #print(f'lower check : {lower_bound- freq_min}, upper check : {upper_bound-freq_max}')
            while lower_bound < freq_min and upper_bound > freq_max:
                row["fwhm_GHz"] -= 0.1
                lower_bound = row["shifted_freq(GHz)"] - row["fwhm_GHz"] / 2
                upper_bound = row["shifted_freq(GHz)"] + row["fwhm_GHz"] / 2
            if row["fwhm_GHz"] != compatible_lines["fwhm_GHz"].iloc[index]:
                compatible_lines.loc[i, "fwhm_GHz"] = row["fwhm_GHz"]
        #print(f'lower check : {lower_bound- freq_min}, upper check : {upper_bound-freq_max}')
        return compatible_lines

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
        line_names=None,
        n_lines=None,
        remote=False,
    ):
        """
        Process spectral data based on the type of source, wavelength conversion,
        line ratios, and given frequency bands.

        Prameters:

        redshift: Redshift value to adjust the spectral lines and cont.
        central_frequency: Central frequency of the observation band (GHz).
        delta_freq: Bandwidth around the central frequency (GHz).
        source_frequency: Frequency of the source obtained from metadata (GHz).
        lines: Optional list of line names provided by the user.
        n_lines: Number of additional lines to consider if lines is None.

        Output:


        """
        # Define the frequency range based on central frequency and bandwidth
        freq_min = central_frequency - delta_freq / 2
        freq_max = central_frequency + delta_freq / 2
        sed, flux_infrared, lum_infrared = self.sed_reading(
            type_,
            os.path.join(master_path, "brightnes"),
            cont_sens,
            freq_min,
            freq_max,
            remote,
            lum_infrared,
        )
        self.db_line = uas.read_line_emission_csv(
            os.path.join(self.main_path, "brightnes", "calibrated_lines.csv"),
            sep=",",
        )
        self.line_cs_mean = np.mean(self.db_line["c"].values)
        self.line_cs_std = np.std(self.db_line["c"].values)
        # line_ratio, line_error
        if line_names is None:
            if n_lines is not None:
                n = n_lines
            else:
                n = 1
        else:
            n = len(line_names)
            self.db_line = self.db_line[self.db_line["Line"].isin(line_names)]
        filtered_lines = self.find_compatible_lines(
            self.db_line,
            source_frequency,
            redshift,
            n,
            line_names,
            freq_min,
            freq_max,
            delta_freq,
        )

        cont_mask = (sed["GHz"] >= freq_min) & (sed["GHz"] <= freq_max)
        if sum(cont_mask) > 0:
            cont_fluxes = sed["Jy"].values[cont_mask]
            cont_frequencies = sed["GHz"].values[cont_mask]
        else:
            freq_point = np.argmin(np.abs(sed["GHz"].values - freq_min))
            cont_fluxes = [sed["Jy"].values[freq_point]]
            cont_frequencies = [sed["GHz"].values[freq_point]]

        line_names = filtered_lines["Line"].values
        cs = filtered_lines["c"].values
        cdeltas = filtered_lines["err_c"].values
        line_ratios = np.array([np.random.normal(c, cd) for c, cd in zip(cs, cdeltas)])
        line_frequencies = filtered_lines["shifted_freq(GHz)"].values
        # line_rest_frequencies = filtered_lines["freq(GHz)"].values * U.GHz
        new_cont_freq = np.linspace(freq_min, freq_max, n_channels)
        if len(cont_fluxes) > 1:
            int_cont_fluxes = np.interp(new_cont_freq, cont_frequencies, cont_fluxes)
        else:
            int_cont_fluxes = np.ones(n_channels) * cont_fluxes[0]
        line_indexes = filtered_lines["shifted_freq(GHz)"].apply(
            lambda x: self.cont_finder(new_cont_freq, float(x))
        )
        fwhms_GHz = filtered_lines["fwhm_GHz"].values
        freq_steps = (
            np.array(
                [
                    new_cont_freq[line_index] + fwhm - new_cont_freq[line_index]
                    for fwhm, line_index in zip(fwhms_GHz, line_indexes)
                ]
            )
            * U.GHz
        )
        freq_steps = freq_steps.to(U.Hz).value
        line_fluxes = 10 ** (np.log10(flux_infrared) + line_ratios) / freq_steps
        bandwidth = freq_max - freq_min
        freq_support = bandwidth / n_channels
        fwhms = []
        for fwhm in fwhms_GHz / freq_support:
            if fwhm >= 1:
                fwhms.append(fwhm)
            else:
                fwhms.append(1)
        # fwhms = [int(fwhm) for fwhm in fwhms_GHz.value / freq_support]
        return (
            int_cont_fluxes,
            line_fluxes,
            line_names,
            redshift,
            line_frequencies,
            line_indexes,
            n_channels,
            bandwidth,
            freq_support,
            new_cont_freq,
            fwhms,
            lum_infrared,
        )

    def simulator(self, *args, **kwargs):
        """
        Simulates the ALMA observations for the given input parameters.

        Parameters:
        idx (int): Index of the simulation.
        source_name (str): Name of the metadata source.
        member_ouid (str): Member OUID of the metadata source.
        main_path (str): Path to the directory where the file.csv is stored.
        output_dir (str): Path to the output directory.
        tng_dir (str): Path to the TNG directory.
        galaxy_zoo_dir (str): Path to the Galaxy Zoo directory.
        hubble_dir (str): Path to the Hubble Top 100 directory.
        project_name (str): Name of the project.
        ra (float): Right ascension.
        dec (float): Declination.
        band (str): Observing band.
        ang_res (float): Angular resolution.
        vel_res (float): Velocity resolution.
        fov (float): Field of view.
        obs_date (str): Observation date.
        pwv (float): Precipitable water vapor.
        int_time (float): Integration time.
        bandwidth (float): Bandwidth.
        freq (float): Frequency.
        freq_support (float): Frequency support.
        cont_sens (float): Continuum sensitivity.
        antenna_array (str): Antenna array.
        n_pix (int): Number of pixels.
        n_channels (int): Number of channels.
        source_type (str): Type of source.
        tng_api_key (str): TNG API key.
        ncpu (int): Number of CPUs.
        rest_frequency (float): Rest frequency.
        redshift (float): Redshift.
        lum_infrared (float): Infrared luminosity.
        snr (float): Signal-to-noise ratio.
        n_lines (int): Number of lines.
        line_names (str): Names of the lines.
        save_mode (str): Save mode.
        inject_serendipitous (bool): Inject serendipitous sources.

        Returns:
        str: Path to the output file.
        """
        (
            inx,
            source_name,
            member_ouid,
            main_dir,
            output_dir,
            tng_dir,
            galaxy_zoo_dir,
            hubble_dir,
            project_name,
            ra,
            dec,
            band,
            ang_res,
            vel_res,
            fov,
            obs_date,
            pwv,
            int_time,
            bandwidth,
            freq,
            freq_support,
            cont_sens,
            antenna_array,
            n_pix,
            n_channels,
            source_type,
            tng_api_key,
            ncpu,
            rest_frequency,
            redshift,
            lum_infrared,
            snr,
            n_lines,
            line_names,
            save_mode,
            inject_serendipitous,
            remote,
        ) = args
        if remote is True:
            print("\nRunning simulation {}".format(inx))
            print("Source Name: {}".format(source_name))
            if pd.isna(n_pix):
                n_pix = None
            if pd.isna(n_channels):
                n_channels = None
            if pd.isna(tng_api_key):
                tng_api_key = None
            if pd.isna(rest_frequency):
                rest_frequency = None
            if pd.isna(redshift):
                redshift = None
            if pd.isna(lum_infrared):
                lum_infrared = None
            if pd.isna(snr):
                snr = None
            if pd.isna(n_lines):
                n_lines = None
            if pd.isna(line_names):
                line_names = None
        else:
            self.terminal.add_log("\nRunning simulation {}".format(inx))
            self.terminal.add_log("Source Name: {}".format(source_name))

        if isinstance(line_names, str):
            # Remove brackets and split into elements
            line_names = line_names.strip("[]").split(
                ","
            )  # Or .split() if single space delimited
            # Convert to NumPy array
            line_names = np.array([name.strip("' ") for name in line_names])
        remote = bool(remote)
        start = time.time()
        second2hour = 1 / 3600
        ra = ra * U.deg
        dec = dec * U.deg
        fov = fov * 3600 * U.arcsec
        ang_res = ang_res * U.arcsec
        vel_res = vel_res * U.km / U.s
        int_time = int_time * U.s
        source_freq = freq * U.GHz
        band_range, central_freq, t_channels, delta_freq = self.freq_supp_extractor(
            freq_support, source_freq
        )
        sim_output_dir = os.path.join(output_dir, project_name + "_{}".format(inx))
        if not os.path.exists(sim_output_dir):
            os.makedirs(sim_output_dir)
        os.chdir(output_dir)

        if remote is True:
            print("RA: {}".format(ra))
            print("DEC: {}".format(dec))
            print("Integration Time: {}".format(int_time))
        else:
            self.terminal.add_log("RA: {}".format(ra))
            self.terminal.add_log("DEC: {}".format(dec))
            self.terminal.add_log("Integration Time: {}".format(int_time))
        ual.generate_antenna_config_file_from_antenna_array(
            antenna_array, main_dir, sim_output_dir
        )
        antennalist = os.path.join(sim_output_dir, "antenna.cfg")
        self.progress_bar_entry.setText("Computing Max baseline")
        max_baseline = (
            ual.get_max_baseline_from_antenna_config(self.update_progress, antennalist)
            * U.km
        )
        if remote is True:
            print("Field of view: {} arcsec".format(round(fov.value, 3)))
        else:
            self.terminal.add_log(
                "Field of view: {} arcsec".format(round(fov.value, 3))
            )
        beam_size = ual.estimate_alma_beam_size(
            central_freq, max_baseline, return_value=False
        )
        beam_area = 1.1331 * beam_size**2
        beam_solid_angle = np.pi * (beam_size / 2) ** 2
        cont_sens = cont_sens * U.mJy / (U.arcsec**2)
        cont_sens_jy = (cont_sens * beam_solid_angle).to(U.Jy)
        # cont_sens = cont_sens_jy * snr
        if remote is True:
            print("Minimum detectable continum: {}".format(cont_sens_jy))
        else:
            self.terminal.add_log(
                "Minimum detectable continum: {}".format(cont_sens_jy)
            )
        cell_size = beam_size / 5
        if n_pix is None:
            # cell_size = beam_size / 5
            n_pix = self.closest_power_of_2(int(1.5 * fov.value / cell_size.value))
        else:
            n_pix = self.closest_power_of_2(n_pix)
            cell_size = fov / n_pix
            # just added
            # beam_size = cell_size * 5
        if n_channels is None:
            n_channels = t_channels
        else:
            band_range = n_channels * delta_freq
        if redshift is None:
            if isinstance(rest_frequency, np.ndarray):
                rest_frequency = np.sort(np.array(rest_frequency))[0]
            rest_frequency = rest_frequency * U.GHz
            redshift = uas.compute_redshift(rest_frequency, source_freq)
        else:
            rest_frequency = (
                uas.compute_rest_frequency_from_redshift(
                    main_dir, source_freq.value, redshift
                )
                * U.GHz
            )
            self.progress_bar_entry.setText("Computing spectral lines and properties")
        (
            continum,
            line_fluxes,
            line_names,
            redshift,
            line_frequency,
            source_channel_index,
            n_channels_nw,
            bandwidth,
            freq_sup_nw,
            cont_frequencies,
            fwhm_z,
            lum_infrared,
        ) = self.process_spectral_data(
            source_type,
            main_dir,
            redshift,
            central_freq.value,
            band_range.value,
            source_freq.value,
            n_channels,
            lum_infrared,
            cont_sens_jy.value,
            line_names,
            n_lines,
            remote,
        )
        if n_channels_nw != n_channels:
            freq_sup = freq_sup_nw * U.MHz
            n_channels = n_channels_nw
            band_range = n_channels * freq_sup
        if remote is True:
            print("Beam size: {} arcsec\n".format(round(beam_size.value, 4)))
            print("Central Frequency: {}\n".format(central_freq))
            print("Spectral Window: {}\n".format(band_range))
            print("Freq Support: {}\n".format(delta_freq))
            print("Cube Dimensions: {} x {} x {}\n".format(n_pix, n_pix, n_channels))
            print("Redshift: {}\n".format(round(redshift, 3)))
            print("Source frequency: {} GHz\n".format(round(source_freq.value, 2)))
            print("Band: {}\n".format(band))
            print("Velocity resolution: {} Km/s\n".format(round(vel_res.value, 2)))
            print("Angular resolution: {} arcsec\n".format(round(ang_res.value, 3)))
            print("Infrared Luminosity: {:.2e}\n".format(lum_infrared))
        else:
            self.terminal.add_log("Central Frequency: {}".format(central_freq))
            self.terminal.add_log(
                "Beam size: {} arcsec".format(round(beam_size.value, 4))
            )
            self.terminal.add_log("Spectral Window: {}".format(band_range))
            self.terminal.add_log("Freq Support: {}".format(delta_freq))
            self.terminal.add_log(
                "Cube Dimensions: {} x {} x {}".format(n_pix, n_pix, n_channels)
            )
            self.terminal.add_log("Redshift: {}".format(round(redshift, 3)))
            self.terminal.add_log(
                "Source frequency: {} GHz".format(round(source_freq.value, 2))
            )
            self.terminal.add_log("Band: {}".format(band))
            self.terminal.add_log(
                "Velocity resolution: {} Km/s".format(round(vel_res.value, 2))
            )
            self.terminal.add_log(
                "Angular resolution: {} arcsec".format(round(ang_res.value, 3))
            )
            self.terminal.add_log("Infrared Luminosity: {:.2e}".format(lum_infrared))
        if source_type == "extended":
            snapshot = uas.redshift_to_snapshot(redshift)
            tng_subhaloid = uas.get_subhaloids_from_db(1, main_dir, snapshot)
        else:
            snapshot = None
            tng_subhaloid = None
        if isinstance(line_names, list) or isinstance(line_names, np.ndarray):
            for line_name, line_flux in zip(line_names, line_fluxes):
                if remote is True:
                    print(
                        "Simulating Line {} Flux: {:.3e} at z {}".format(
                            line_name, line_flux, redshift
                        )
                    )
                else:
                    self.terminal.add_log(
                        "Simulating Line {} Flux: {:.3e} at z {}".format(
                            line_name, line_flux, redshift
                        )
                    )
        else:
            if remote is True:
                print(
                    "Simulating Line {} Flux: {} at z {}".format(
                        line_names[0], line_fluxes[0], redshift
                    )
                )
            else:
                self.terminal.add_log(
                    "Simulating Line {} Flux: {} at z {}".format(
                        line_names[0], line_fluxes[0], redshift
                    )
                )
        if remote is True:
            print("Simulating Continum Flux: {:.2e}".format(np.mean(continum)))
            print("Continuum Sensitity: {:.2e}".format(cont_sens))
            print("Generating skymodel cube ...\n")
        else:
            self.terminal.add_log(
                "Simulating Continum Flux: {:.2e}".format(np.mean(continum))
            )
            self.terminal.add_log("Continuum Sensitity: {:.2e}".format(cont_sens))
            self.terminal.add_log("Generating skymodel cube ...")
        datacube = usm.DataCube(
            n_px_x=n_pix,
            n_px_y=n_pix,
            n_channels=n_channels,
            px_size=cell_size,
            channel_width=delta_freq,
            spectral_centre=central_freq,
            ra=ra,
            dec=dec,
        )
        wcs = datacube.wcs
        fwhm_x, fwhm_y, angle = None, None, None
        mean_shift = (np.sqrt(2) / 2) * 22
        std_shift = (np.sqrt(2) / 2) * 44
        shift_x = int(np.random.normal(loc=mean_shift, scale=std_shift))
        shift_y = int(np.random.normal(loc=mean_shift, scale=std_shift))
        if abs(shift_x) > 0.8 * n_pix / 2:
            if shift_y > 0:
                shift_x = int(0.8 * n_pix / 2)
            else:
                shift_x = -int(0.8 * n_pix / 2)
        if abs(shift_y) > 0.8 * n_pix / 2:
            if shift_y > 0:
                shift_y = int(0.8 * n_pix / 2)
            else:
                shift_y = -int(0.8 * n_pix / 2)
        if source_type == "point":
            pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
            pos_x = pos_x + shift_x
            pos_y = pos_x + shift_y
            pos_z = [int(index) for index in source_channel_index]
            self.progress_bar_entry.setText("Inserting Point Source Model")
            datacube = usm.insert_pointlike(
                self.update_progress,
                datacube,
                continum,
                line_fluxes,
                int(pos_x),
                int(pos_y),
                pos_z,
                fwhm_z,
                n_channels,
            )
        elif source_type == "gaussian":
            self.progress_bar_entry.setText("Inserting Gaussian Source Model")
            pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
            pos_x = pos_x + shift_x
            pos_y = pos_x + shift_y
            pos_z = [int(index) for index in source_channel_index]
            fwhm_x = np.random.randint(3, 10)
            fwhm_y = np.random.randint(3, 10)
            angle = np.random.randint(0, 180)
            datacube = usm.insert_gaussian(
                self.update_progress,
                datacube,
                continum,
                line_fluxes,
                int(pos_x),
                int(pos_y),
                pos_z,
                fwhm_x,
                fwhm_y,
                fwhm_z,
                angle,
                n_pix,
                n_channels,
            )
        elif source_type == "extended":
            self.progress_bar_entry.setText("Inserting Extended Source Model")
            datacube = usm.insert_extended(
                self.update_progress,
                self.terminal,
                datacube,
                tng_dir,
                snapshot,
                int(tng_subhaloid),
                redshift,
                ra,
                dec,
                tng_api_key,
                ncpu,
            )
        elif source_type == "diffuse":
            datacube = usm.insert_diffuse(
                self.update_progress,
                datacube,
                continum,
                line_fluxes,
                pos_z,
                fwhm_z,
                n_pix,
                n_channels,
            )
        elif source_type == "galaxy-zoo":
            self.progress_bar_entry.setText("Inserting Galaxy Zoo Source Model")
            galaxy_path = os.path.join(galaxy_zoo_dir, "images_gz2", "images")
            pos_z = [int(index) for index in source_channel_index]
            datacube = usm.insert_galaxy_zoo(
                self.update_progress,
                datacube,
                continum,
                line_fluxes,
                pos_z,
                fwhm_z,
                n_pix,
                n_channels,
                galaxy_path,
            )
        elif source_type == "molecular":
            self.progress_bar_entry.setText("Inserting Molecular Cloud Source Model")
            pos_z = [int(index) for index in source_channel_index]
            datacube = usm.insert_molecular_cloud(
                self.update_progress,
                datacube,
                continum,
                line_fluxes,
                pos_z,
                fwhm_z,
                n_pix,
                n_channels,
            )
        elif source_type == "hubble-100":
            self.progress_bar_entry.setText("Insert Hubble Top 100 Source Model")
            hubble_path = os.path.join(hubble_dir, "top100")
            pos_z = [int(index) for index in source_channel_index]
            datacube = usm.insert_hubble(
                self.update_progress,
                datacube,
                continum,
                line_fluxes,
                pos_z,
                fwhm_z,
                n_pix,
                n_channels,
                hubble_path,
            )
        uas.write_sim_parameters(
            os.path.join(output_dir, "sim_params_{}.txt".format(inx)),
            source_name,
            member_ouid,
            ra,
            dec,
            ang_res,
            vel_res,
            int_time,
            band,
            band_range,
            central_freq,
            redshift,
            line_fluxes,
            line_names,
            line_frequency,
            continum,
            fov,
            beam_size,
            cell_size,
            n_pix,
            n_channels,
            snapshot,
            tng_subhaloid,
            lum_infrared,
            fwhm_z,
            source_type,
            fwhm_x,
            fwhm_y,
            angle,
        )
        if bool(inject_serendipitous) is True and not self.stop_simulation_flag:
            self.progress_bar_entry.setText("Inserting Serendipitous Sources")
            if source_type != "gaussian":
                fwhm_x = np.random.randint(3, 10)
                fwhm_y = np.random.randint(3, 10)
            datacube = usm.insert_serendipitous(
                self.terminal,
                self.update_progress,
                datacube,
                continum,
                cont_sens.value,
                line_fluxes,
                line_names,
                line_frequency,
                delta_freq.value,
                pos_z,
                fwhm_x,
                fwhm_y,
                fwhm_z,
                n_pix,
                n_channels,
                os.path.join(output_dir, "sim_params_{}.txt".format(inx)),
            )
        header = usm.get_datacube_header(datacube, obs_date)
        model = datacube._array.to_value(datacube._array.unit).T
        model = model / beam_area.value
        totflux = np.sum(model)
        if remote is True:
            print("Total Flux injected in model cube: {:.3f} Jy\n".format(totflux))
            print("Done\n")
        else:
            self.terminal.add_log(
                f"Total Flux injected in model cube: {round(totflux, 3)} Jy"
            )
            self.terminal.add_log("Done")
        del datacube
        if remote is True:
            print("Observing with ALMA\n")
        else:
            self.terminal.add_log("Observing with ALMA")
        min_line_flux = np.min(line_fluxes)
        interferometer = uin.Interferometer(
            inx,
            model,
            main_dir,
            output_dir,
            ra,
            dec,
            central_freq,
            band_range,
            fov,
            antenna_array,
            (min_line_flux / beam_area.value) / snr,
            snr,
            int_time.value * second2hour,
            obs_date,
            header,
            save_mode,
            self.terminal,
            self.stop_simulation_flag,
            float(self.robust_slider.value()) / 10,
        )
        interferometer.progress_signal.connect(self.handle_progress)
        self.terminal.add_log(
            "Setting Brigg's robust parameter to {}".format(
                float(self.robust_slider.value()) / 10
            )
        )
        self.progress_bar_entry.setText("Observing with ALMA")
        simulation_results = interferometer.run_interferometric_sim()
        if remote is True:
            print("Finished")
        else:
            self.terminal.add_log("Finished")
        stop = time.time()
        if remote is True:
            print(
                "Execution took {} seconds".format(
                    strftime("%H:%M:%S", gmtime(stop - start))
                )
            )
        else:
            self.terminal.add_log(
                "Simulation took {} seconds, creating plots".format(
                    strftime("%H:%M:%S", gmtime(stop - start))
                )
            )
            self.progress_bar_entry.setText("Simulation Finished")
        shutil.rmtree(sim_output_dir)
        return simulation_results

    # ------- Progress Bar ---------------------------------
    @pyqtSlot(int)
    def handle_progress(self, value):
        self.update_progress.emit(value)

    @pyqtSlot(int)
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    # -------- Plotting Functions ---------------------------
    @pyqtSlot(object)
    def plot_simulation_results(self, simulation_results):
        if simulation_results is not None:
            self.progress_bar_entry.setText("Generating Plots")
            # Extract data from the simulation_results dictionary
            self.modelCube = simulation_results["modelCube"]
            self.dirtyCube = simulation_results["dirtyCube"]
            self.visCube = simulation_results["visCube"]
            self.dirtyvisCube = simulation_results["dirtyvisCube"]
            self.Npix = simulation_results["Npix"]
            self.Np4 = simulation_results["Np4"]
            self.Nchan = simulation_results["Nchan"]
            self.gamma = simulation_results["gamma"]
            self.currcmap = simulation_results["currcmap"]
            self.Xaxmax = simulation_results["Xaxmax"]
            self.lfac = simulation_results["lfac"]
            self.u = simulation_results["u"]
            self.v = simulation_results["v"]
            self.UVpixsize = simulation_results["UVpixsize"]
            self.w_min = simulation_results["w_min"]
            self.w_max = simulation_results["w_max"]
            self.plot_dir = simulation_results["plot_dir"]
            self.idx = simulation_results["idx"]
            self.wavelength = simulation_results["wavelength"]
            self.totsampling = simulation_results["totsampling"]
            self.beam = simulation_results["beam"]
            self.fmtB = simulation_results["fmtB"]
            self.curzoom = simulation_results["curzoom"]
            self.Nphf = simulation_results["Nphf"]
            self.Xmax = simulation_results["Xmax"]
            self.antPos = simulation_results["antPos"]
            self.Nant = simulation_results["Nant"]
            self._plot_beam()
            self._plot_uv_coverage()
            self._plot_antennas()
            self._plot_sim()
            self.progress_bar_entry.setText("Done")
            self.terminal.add_log("Plotting Finished")

    @pyqtSlot(object)
    def start_plot_runnable(self, simulation_results):
        runnable = PlotResultsRunnable(self, simulation_results)
        self.thread_pool.start(runnable)

    def _plot_antennas(self):
        plt.figure(figsize=(8, 8))
        toplot = np.array(self.antPos[: self.Nant])
        plt.plot([0], [0], "-b")[0]
        plt.plot(toplot[:, 0], toplot[:, 1], "o", color="lime", picker=5)[0]
        plt.xlim(-self.Xmax, self.Xmax)
        plt.ylim(-self.Xmax, self.Xmax)
        plt.xlabel("East-West offset (Km)")
        plt.ylabel("North-South offset (Km)")
        plt.title("Antenna Configuration")
        plt.savefig(
            os.path.join(self.plot_dir, "antenna_config_{}.png".format(str(self.idx)))
        )
        plt.close()

    def _plot_uv_coverage(self):
        self.ulab = r"U (k$\lambda$)"
        self.vlab = r"V (k$\lambda$)"
        plt.figure(figsize=(8, 8))
        UVPlotPlot = []
        toplotu = self.u.flatten() / self.lfac
        toplotv = self.v.flatten() / self.lfac
        UVPlotPlot.append(
            plt.plot(toplotu, toplotv, ".", color="lime", markersize=1, picker=2)[0]
        )
        UVPlotPlot.append(
            plt.plot(-toplotu, -toplotv, ".", color="lime", markersize=1, picker=2)[0]
        )
        plt.xlim(
            (
                2.0 * self.Xmax / self.wavelength[2] / self.lfac,
                -2.0 * self.Xmax / self.wavelength[2] / self.lfac,
            )
        )
        plt.ylim(
            (
                2.0 * self.Xmax / self.wavelength[2] / self.lfac,
                -2.0 * self.Xmax / self.wavelength[2] / self.lfac,
            )
        )
        plt.xlabel(self.ulab)
        plt.ylabel(self.vlab)
        plt.title("UV Coverage")
        plt.savefig(
            os.path.join(self.plot_dir, "uv_coverage_{}.png".format(str(self.idx)))
        )
        plt.close()

    def _plot_beam(self):
        plt.figure(figsize=(8, 8))
        beamPlotPlot = plt.imshow(
            self.beam[self.Np4 : self.Npix - self.Np4, self.Np4 : self.Npix - self.Np4],
            picker=True,
            interpolation="nearest",
            cmap=self.currcmap,
        )
        beamText = plt.text(
            0.80,
            0.80,
            self.fmtB % (1.0, 0.0, 0.0),
            bbox=dict(facecolor="white", alpha=0.7),
        )
        plt.ylabel("Dec offset (as)")
        plt.xlabel("RA offset (as)")
        plt.setp(
            beamPlotPlot,
            extent=(
                self.Xaxmax / 2.0,
                -self.Xaxmax / 2.0,
                -self.Xaxmax / 2.0,
                self.Xaxmax / 2.0,
            ),
        )
        self.curzoom[0] = (
            self.Xaxmax / 2.0,
            -self.Xaxmax / 2.0,
            -self.Xaxmax / 2.0,
            self.Xaxmax / 2.0,
        )
        plt.title("DIRTY BEAM")
        plt.colorbar()
        nptot = np.sum(self.totsampling[:])
        beamPlotPlot.norm.vmin = np.min(self.beam)
        beamPlotPlot.norm.vmax = 1.0
        if (
            np.sum(
                self.totsampling[
                    self.Nphf - 4 : self.Nphf + 4, self.Nphf - 4 : self.Nphf + 4
                ]
            )
            == nptot
        ):
            warn = "WARNING!\nToo short baselines for such a small image\nPLEASE, \
                INCREASE THE IMAGE SIZE!\nAND/OR DECREASE THE WAVELENGTH"
            beamText.set_text(warn)

        plt.savefig(os.path.join(self.plot_dir, "beam_{}.png".format(str(self.idx))))
        plt.close()

    def _plot_sim(self):
        simPlot, ax = plt.subplots(2, 3, figsize=(18, 12))
        sim_img = np.sum(self.modelCube, axis=0)
        simPlotPlot = ax[0, 0].imshow(
            np.power(
                sim_img[
                    self.Np4 : self.Npix - self.Np4, self.Np4 : self.Npix - self.Np4
                ],
                self.gamma,
            ),
            picker=True,
            interpolation="nearest",
            vmin=0.0,
            vmax=np.max(sim_img) ** self.gamma,
            cmap=self.currcmap,
        )
        plt.setp(
            simPlotPlot,
            extent=(
                self.Xaxmax / 2.0,
                -self.Xaxmax / 2.0,
                -self.Xaxmax / 2.0,
                self.Xaxmax / 2.0,
            ),
        )
        ax[0, 0].set_ylabel("Dec offset (as)")
        ax[0, 0].set_xlabel("RA offset (as)")
        totflux = np.sum(
            sim_img[self.Np4 : self.Npix - self.Np4, self.Np4 : self.Npix - self.Np4]
        )
        ax[0, 0].set_title("MODEL IMAGE: %.2e Jy/beam" % totflux)
        simPlotPlot.norm.vmin = np.min(sim_img)
        simPlotPlot.norm.vmax = np.max(sim_img)
        dirty_img = np.sum(self.dirtyCube, axis=0)
        dirtyPlotPlot = ax[1, 0].imshow(
            dirty_img[self.Np4 : self.Npix - self.Np4, self.Np4 : self.Npix - self.Np4],
            picker=True,
            interpolation="nearest",
        )
        plt.setp(
            dirtyPlotPlot,
            extent=(
                self.Xaxmax / 2.0,
                -self.Xaxmax / 2.0,
                -self.Xaxmax / 2.0,
                self.Xaxmax / 2.0,
            ),
        )
        ax[1, 0].set_ylabel("Dec offset (as)")
        ax[1, 0].set_xlabel("RA offset (as)")
        totflux = np.sum(
            dirty_img[self.Np4 : self.Npix - self.Np4, self.Np4 : self.Npix - self.Np4]
        )
        ax[1, 0].set_title("DIRTY IMAGE: %.2e Jy/beam" % totflux)
        dirtyPlotPlot.norm.vmin = np.min(dirty_img)
        dirtyPlotPlot.norm.vmax = np.max(dirty_img)
        self.UVmax = self.Npix / 2.0 / self.lfac * self.UVpixsize
        self.UVSh = -self.UVmax / self.Npix
        toplot = np.sum(np.abs(self.visCube), axis=0)
        mval = np.min(toplot)
        Mval = np.max(toplot)
        dval = (Mval - mval) / 2.0
        UVPlotFFTPlot = ax[0, 1].imshow(
            toplot, cmap=self.currcmap, vmin=0.0, vmax=Mval + dval, picker=5
        )
        plt.setp(
            UVPlotFFTPlot,
            extent=(
                -self.UVmax + self.UVSh,
                self.UVmax + self.UVSh,
                -self.UVmax - self.UVSh,
                self.UVmax - self.UVSh,
            ),
        )

        ax[0, 1].set_ylabel("V (k$\\lambda$)")
        ax[0, 1].set_xlabel("U (k$\\lambda$)")
        ax[0, 1].set_title("MODEL VISIBILITY")

        toplot = np.sum(np.abs(self.dirtyvisCube), axis=0)
        mval = np.min(toplot)
        Mval = np.max(toplot)
        dval = (Mval - mval) / 2.0
        UVPlotDirtyFFTPlot = ax[1, 1].imshow(
            toplot, cmap=self.currcmap, vmin=0.0, vmax=Mval + dval, picker=5
        )
        plt.setp(
            UVPlotDirtyFFTPlot,
            extent=(
                -self.UVmax + self.UVSh,
                self.UVmax + self.UVSh,
                -self.UVmax - self.UVSh,
                self.UVmax - self.UVSh,
            ),
        )
        ax[1, 1].set_ylabel("V (k$\\lambda$)")
        ax[1, 1].set_xlabel("U (k$\\lambda$)")
        ax[1, 1].set_title("DIRTY VISIBILITY")

        phaseplot = np.sum(np.angle(self.visCube), axis=0)
        PhasePlotFFTPlot = ax[0, 2].imshow(
            phaseplot, cmap="twilight", vmin=-np.pi, vmax=np.pi, picker=5
        )
        plt.setp(
            PhasePlotFFTPlot,
            extent=(
                -self.UVmax + self.UVSh,
                self.UVmax + self.UVSh,
                -self.UVmax - self.UVSh,
                self.UVmax - self.UVSh,
            ),
        )
        ax[0, 2].set_ylabel("V (k$\\lambda$)")
        ax[0, 2].set_xlabel("U (k$\\lambda$)")
        ax[0, 2].set_title("MODEL VISIBILITY PHASE")

        phaseplot = np.sum(np.angle(self.dirtyvisCube), axis=0)
        PhasePlotFFTPlot = ax[1, 2].imshow(
            phaseplot, cmap="twilight", vmin=-np.pi, vmax=np.pi, picker=5
        )
        plt.setp(
            PhasePlotFFTPlot,
            extent=(
                -self.UVmax + self.UVSh,
                self.UVmax + self.UVSh,
                -self.UVmax - self.UVSh,
                self.UVmax - self.UVSh,
            ),
        )
        ax[1, 2].set_ylabel("V (k$\\lambda$)")
        ax[1, 2].set_xlabel("U (k$\\lambda$)")
        ax[1, 2].set_title("DIRTY VISIBILITY PHASE")
        plt.savefig(os.path.join(self.plot_dir, "sim_{}.png".format(str(self.idx))))
        plt.close()

        sim_spectrum = np.sum(self.modelCube, axis=(1, 2))
        dirty_spectrum = np.sum(self.dirtyCube, axis=(1, 2))
        wavelenghts = np.linspace(self.w_min, self.w_max, self.Nchan)
        specPlot, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(wavelenghts, sim_spectrum)
        ax[0].set_ylabel("Jy/beam")
        ax[0].set_xlabel("$\\lambda$ [mm]")
        ax[0].set_title("MODEL SPECTRUM")
        ax[1].plot(wavelenghts, dirty_spectrum)
        ax[1].set_ylabel("Jy/beam")
        ax[1].set_xlabel("$\\lambda$ [mm]")
        ax[1].set_title("DIRTY SPECTRUM")
        plt.savefig(os.path.join(self.plot_dir, "spectra_{}.png".format(str(self.idx))))
        plt.close()

    # -------- Utility Functions ---------------------------
    # def send_email(self):
    #    """
    #    Sends an email to the user with the simulation results.
    #    """
    #    path = os.path.dirname(self.main_path)
    #    print(path)
    #    yag = yagmail.SMTP(
    #        user="almasimulator@gmail.com",
    #        oauth2_file=os.path.join(path, 'email_client.json'),
    #    )
    #    subject = f"Simulation {self.project_name_entry.text()} Finished"
    #    body1 = "The simulations have finished\n"
    #    location = self.output_path
    #    if self.local_mode_combo.currentText() == "local":
    #        body2 = f"You can find your results on your local machine in {location}"
    #    else:
    #        address = self.remote_address_entry.text()
    #        body2 = f"You can find your results on {address} in {location}"
    #    body = body1 + body2
    #    to_email = self.mail_entry.text()
    #    yag.send(
    #        to=to_email,
    #        contents=body,
    #        subject=subject
    #    )
