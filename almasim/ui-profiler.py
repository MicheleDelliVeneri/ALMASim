import numpy as np
import pandas as pd
import os
from PyQt6.QtWidgets import (
    QApplication,
    QFormLayout,
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
from qtrangeslider import QRangeSlider
from qtrangeslider.qtcompat import QtCore
from qtrangeslider.qtcompat import QtWidgets as QtW
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
import sys

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

class SignalEmitter(QObject):
    simulationFinished = pyqtSignal(object)
    queryFinished = pyqtSignal(object)
    progress = pyqtSignal(int)



class QueryKeyword(QRunnable, QObject):
    finished = pyqtSignal()
    def __init__(self, alma_simulator_instance):
        QRunnable.__init__(self)
        QObject.__init__(self)  # QObject for signals
        self.alma_simulator = (
            alma_simulator_instance  # Store a reference to the main UI class
        )

    def run(self):
        try:
            self.alma_simulator_instance.terminal.add_log(("Querying metadata by science type..."))
            science_keywords, scientific_categories = ual.get_science_types()
        finally: 
            self.finished.emit()
            for i, keyword in enumerate(science_keywords):
                self.alma_simulator_instance.terminal.add_log(f"{i}: {keyword}")
            self.alma_simulator_instance.terminal.add_log("\nAvailable scientific categories:")
            for i, category in enumerate(scientific_categories):
                self.alma_simulator_instance.terminal.add_log(f"{i}: {category}")

class ALMASimulator(QMainWindow):
    settings_file = None
    ncpu_entry = None
    terminal = None
    thread_pool = None
    update_progress = pyqtSignal(int)

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
        self.initialize_ui()
        self.stop_simulation_flag = False
        self.terminal.add_log(f"Setting file path is {self.settings_path}")

    # -------- Widgets and UI -------------------------
    def initialize_ui(self):
        
        self.setWindowTitle("ALMASim: Set up your simulation parameters")
        #self.setMinimumSize(800, 600)  # Set minimum size for laptop usage
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)  # Horizontal layout for main wind
        self.build_left_layout()
        self.build_right_layout()
        main_layout.addWidget(self.left_scroll_area)
        main_layout.addLayout(self.right_layout)
        main_layout.setStretch(0, 2)  # Stretch factor for the left side
        main_layout.setStretch(1, 1)  # Stretch factor for the right side
        self.metadata_path_label = QLabel("Metadata Path:")
        self.metadata_path_entry = QLineEdit()
        self.metadata_path_button = QPushButton("Browse")
        self.metadata_path_button.clicked.connect(self.browse_metadata_path)
        self.metadata_path_row = QHBoxLayout()
        self.metadata_path_row.addWidget(self.metadata_path_label)
        self.metadata_path_row.addWidget(self.metadata_path_entry)
        self.metadata_path_row.addWidget(self.metadata_path_button)
        self.line_displayed = False
        ALMASimulator.populate_class_variables(
            self.terminal, self.ncpu_entry, self.thread_pool
        )
        #if self.on_remote is True:
        #    self.load_settings_on_remote()
        #else:
        #    self.load_settings()
        self.toggle_line_mode_widgets()
        self.metadata_mode_combo.currentTextChanged.connect(self.toggle_metadata_browse)
        if self.metadata_path_entry.text() != "" and isfile(
            self.metadata_path_entry.text()
        ):
            self.load_metadata(self.metadata_path_entry.text())
        current_mode = self.metadata_mode_combo.currentText()
        self.toggle_metadata_browse(current_mode)  # Call here
        ALMASimulator.populate_class_variables(
            self.terminal, self.ncpu_entry, self.thread_pool
        )
        self.set_window_size()

    @classmethod
    def populate_class_variables(cls, terminal, ncpu_entry, thread_pool):
        cls.terminal = terminal
        cls.ncpu_entry = ncpu_entry
        cls.thread_pool = thread_pool

    def build_right_layout(self):
        self.right_layout = QVBoxLayout()
        self.term = QTextEdit()
        self.term.setReadOnly(True)
        self.terminal = TerminalLogger(self.term)
        self.right_layout.addWidget(self.term)
        self.progress_bar_entry = QLabel()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.right_layout.addWidget(self.progress_bar_entry)
        self.right_layout.addWidget(self.progress_bar)

    def build_left_layout(self):
        # Create the scrollable left layout
        self.left_scroll_area = QScrollArea(self)
        self.left_scroll_area.setWidgetResizable(True)
        self.left_scroll_content = QWidget()
        self.left_scroll_area.setWidget(self.left_scroll_content)
        self.left_content_layout = QVBoxLayout(self.left_scroll_content)
        self.left_layout = QFormLayout(self.left_scroll_content)
        self.left_layout.setSpacing(5) 
        self.find_label_width()
        self.add_folder_widgets()
        self.add_line_widgets()
        self.add_width_slider()
        self.add_robust_slider()
        self.add_dim_widgets()
        self.add_model_widgets()
        self.add_meta_widgets()
        self.add_query_widgets()
        self.left_content_layout.addLayout(self.left_layout)
        # Add a stretch to push the content upwards
        self.left_content_layout.addStretch()
        # Add footer buttons at the bottom
        self.add_footer_buttons()
        
    # -------- Window Browsing Functions -------------------------
    def map_to_remote_directory(self, directory):
        directory_name = directory.split(os.path.sep)[-1]
        if self.remote_folder_line.text() != "":
            if not self.remote_folder_line.text().startswith("/"):
                self.remote_folder_line.setText("/" + self.remote_folder_line.text())
            directory_path = os.path.join(self.remote_folder_line.text(), directory_name)
        else:
            directory_path = os.path.join(
                "/home", self.remote_user_entry.text(), directory_name
            )
        return directory_path
    
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

    def select_metadata_path(self):
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Select Metadata File",
            os.path.join(self.main_path, "metadata"),
            "CSV Files (*.csv)",
        )
        if file:
            self.query_save_entry.setText(file)

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
            
    # -------- UI Widgets Functions -------------------------

    # Utility Functions
    def find_label_width(self):
        labels = [
        QLabel("Output Directory:"),
        QLabel("TNG Directory:"),
        QLabel("Galaxy Zoo Directory:"),
        QLabel("Hubble Top 100 Directory:"),
        QLabel("Project Name:"),
        QLabel("Number of Simulations:"),
        QLabel("Number of Cores:"),
        QLabel("Save Format:"),
        QLabel("Local / Remote Mode:"),
        QLabel("Line Widths in Km/s:"),
        QLabel("Briggs Robustness:"),
        QLabel("Metadata Mode:")
        ]
        # Find the maximum width of the labels
        self.max_label_width = max(label.sizeHint().width() for label in labels)

    def has_widget(self, layout, widget_type):
        """Check if the layout contains a widget of a specific type."""
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(item.widget(), widget_type):
                return True
        return False

    def show_hide_widgets(self, layout, show=True):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                if show:
                    item.widget().show()
                else:
                    item.widget().hide()

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

    def metadata_path_set(self):
        metadata_path = self.metadata_path_entry.text()
        self.load_metadata(metadata_path)
    
    def update_width_labels(self, tuple_values):
        # Update the labels when the range slider values change
        self.line_width_value_label_min.setText(f"Min: {tuple_values[0]} km/s")
        self.line_width_value_label_max.setText(f"Max: {tuple_values[1]} km/s")

    def update_robust_label(self, value):
        self.robust_value_label.setText(f"{value / 10}")
    
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

    # Add Widget Functions
    def add_folder_widgets(self):
        line_edit_max_width = 400  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
        # Output Directory
        self.output_label = QLabel("Output Directory:")
        self.output_label.setFixedWidth(self.max_label_width)
        self.output_entry = QLineEdit()
        self.output_entry.setMaximumWidth(line_edit_max_width)
        self.output_button = QPushButton("Browse")
        self.output_button.setFixedHeight(button_height)
        self.output_button.setFixedWidth(button_width)
        self.output_button.setStyleSheet(button_style)
        self.output_button.clicked.connect(self.browse_output_directory)
        self.output_row = QHBoxLayout()
        self.output_row.addWidget(self.output_label)
        self.output_row.addWidget(self.output_entry)
        self.output_row.addWidget(self.output_button)
        self.output_row.addStretch()
        self.left_layout.addRow(self.output_row)

        # TNG Directory
        self.tng_label = QLabel("TNG Directory:")
        self.tng_label.setFixedWidth(self.max_label_width)
        self.tng_entry = QLineEdit()
        self.tng_entry.setMaximumWidth(line_edit_max_width)
        self.tng_button = QPushButton("Browse")
        self.tng_button.clicked.connect(self.browse_tng_directory)
        self.tng_button.setFixedHeight(button_height)
        self.tng_button.setFixedWidth(button_width)
        self.tng_button.setStyleSheet(button_style)
        self.tng_row = QHBoxLayout()
        self.tng_row.addWidget(self.tng_label)
        self.tng_row.addWidget(self.tng_entry)
        self.tng_row.addWidget(self.tng_button)
        self.tng_row.addStretch()
        self.left_layout.addRow(self.tng_row)

        # Galaxy Zoo Directory
        self.galaxy_zoo_label = QLabel("Galaxy Zoo Directory:")
        self.galaxy_zoo_label.setFixedWidth(self.max_label_width)
        self.galaxy_zoo_entry = QLineEdit()
        self.galaxy_zoo_entry.setMaximumWidth(line_edit_max_width)
        self.galaxy_zoo_button = QPushButton("Browse")
        self.galaxy_zoo_button.clicked.connect(self.browse_galaxy_zoo_directory)
        self.galaxy_zoo_button.setFixedHeight(button_height)
        self.galaxy_zoo_button.setFixedWidth(button_width)
        self.galaxy_zoo_button.setStyleSheet(button_style)
        self.galaxy_zoo_checkbox = QCheckBox("Get Galaxy Zoo")
        self.galaxy_row = QHBoxLayout()
        self.galaxy_row.addWidget(self.galaxy_zoo_label)
        self.galaxy_row.addWidget(self.galaxy_zoo_entry)
        self.galaxy_row.addWidget(self.galaxy_zoo_button)
        self.galaxy_chechbox_row = QHBoxLayout()
        self.galaxy_chechbox_row.addWidget(self.galaxy_zoo_checkbox)
        self.galaxy_row.addStretch()
        self.galaxy_chechbox_row.addStretch()
        self.left_layout.addRow(self.galaxy_row)
        self.left_layout.addRow(self.galaxy_chechbox_row)

        # Hubble Top 100 Directory
        self.hubble_label = QLabel("Hubble Top 100 Directory:")
        self.hubble_label.setFixedWidth(self.max_label_width)
        self.hubble_entry = QLineEdit()
        self.hubble_entry.setMaximumWidth(line_edit_max_width)
        self.hubble_button = QPushButton("Browse")
        self.hubble_button.clicked.connect(self.browse_hubble_directory)
        self.hubble_button.setFixedHeight(button_height)
        self.hubble_button.setFixedWidth(button_width)
        self.hubble_button.setStyleSheet(button_style)
        self.hubble_checkbox = QCheckBox("Get Hubble Data")
        self.hubble_row = QHBoxLayout()
        self.hubble_row.addWidget(self.hubble_label)
        self.hubble_row.addWidget(self.hubble_entry)
        self.hubble_row.addWidget(self.hubble_button)
        self.hubble_chechbox_row = QHBoxLayout()
        self.hubble_chechbox_row.addWidget(self.hubble_checkbox)
        self.hubble_row.addStretch()
        self.hubble_chechbox_row.addStretch()
        self.left_layout.addRow(self.hubble_row)
        self.left_layout.addRow(self.hubble_chechbox_row)

        # Project Name
        self.project_name_label = QLabel("Project Name:")
        self.project_name_label.setFixedWidth(self.max_label_width)
        self.project_name_entry = QLineEdit()
        self.project_name_entry.setMaximumWidth(line_edit_max_width)
        self.project_name_row = QHBoxLayout()
        self.project_name_row.addWidget(self.project_name_label)
        self.project_name_row.addWidget(self.project_name_entry)
        self.project_name_row.addStretch()
        self.left_layout.addRow(self.project_name_row)

        # Number of Simulaitons 
        self.n_sims_label = QLabel("Number of Simulations:")
        self.n_sims_label.setFixedWidth(self.max_label_width)
        self.n_sims_entry = QLineEdit()
        self.n_sims_entry.setMaximumWidth(line_edit_max_width)
        self.n_sims_row = QHBoxLayout()
        self.n_sims_row.addWidget(self.n_sims_label)
        self.n_sims_row.addWidget(self.n_sims_entry)
        self.n_sims_row.addStretch()
        self.left_layout.addRow(self.n_sims_row)

        # Number of Cores
        self.ncpu_label = QLabel("Number of Cores:")
        self.ncpu_label.setFixedWidth(self.max_label_width)
        self.ncpu_entry = QLineEdit()
        self.ncpu_entry.setMaximumWidth(line_edit_max_width)
        self.ncpu_row = QHBoxLayout()
        self.ncpu_row.addWidget(self.ncpu_label)
        self.ncpu_row.addWidget(self.ncpu_entry)
        self.ncpu_row.addStretch()
        self.left_layout.addRow(self.ncpu_row)


        # Save Format 
        self.save_format_label = QLabel("Save Format:")
        self.save_format_label.setFixedWidth(self.max_label_width)
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["npz", "fits", "h5"])
        self.save_format_row = QHBoxLayout()
        self.save_format_row.addWidget(self.save_format_label)
        self.save_format_row.addWidget(self.save_format_combo)
        self.save_format_row.addStretch()
        self.left_layout.addRow(self.save_format_row)
        
        # Local / Remote Mode
        self.local_mode_label = QLabel("Local / Remote Mode:")
        self.local_mode_label.setFixedWidth(self.max_label_width)
        self.local_mode_combo = QComboBox()
        self.local_mode_combo.addItems(["local", "remote"])
        # Remote Compute Mode 
        self.remote_mode_label = QLabel("Remote Compute Mode:")
        self.remote_mode_label.setFixedWidth(self.max_label_width)
        self.remote_mode_combo = QComboBox()
        self.remote_mode_combo.addItems(["MPI", "SLURM", "PBS"])
        self.remote_folder_checkbox = QCheckBox("Set Working Directory")
        self.remote_folder_checkbox.stateChanged.connect(self.toggle_remote_folder_line)
        self.remote_folder_line = QLineEdit()
        self.remote_folder_line.setMaximumWidth(line_edit_max_width)
        self.local_mode_row = QHBoxLayout()
        self.local_mode_row.addWidget(self.local_mode_label)
        self.local_mode_row.addWidget(self.local_mode_combo)
        self.local_mode_row.addWidget(self.remote_mode_label)
        self.local_mode_row.addWidget(self.remote_mode_combo)
        self.local_mode_row.addWidget(self.remote_folder_checkbox)
        self.local_mode_row.addWidget(self.remote_folder_line)
        self.local_mode_row.addStretch()
        self.left_layout.addRow(self.local_mode_row)
        self.remote_mode_label.hide()
        self.remote_mode_combo.hide()
        self.remote_folder_line.hide()
        self.remote_folder_checkbox.hide()
        self.remote_mode_combo.currentTextChanged.connect(self.toggle_remote_row)   


        # Remote info 
        self.remote_address_label = QLabel("Remote Host:")
        self.remote_address_entry = QLineEdit()
        self.remote_address_entry.setMaximumWidth(line_edit_max_width)
        self.remote_config_label = QLabel("SLURM Config:")
        self.remote_config_entry = QLineEdit()
        self.remote_config_button = QPushButton("Browse", self)
        self.remote_config_button.setFixedHeight(button_height)
        self.remote_config_button.setFixedWidth(button_width)
        self.remote_config_button.setStyleSheet(button_style)
        
        self.remote_config_button.clicked.connect(self.browse_slurm_config)
        self.remote_user_label = QLabel("User:")
        self.remote_user_entry = QLineEdit()
        self.remote_user_entry.setMaximumWidth(line_edit_max_width)
        self.remote_key_label = QLabel("SSH Key:")
        self.remote_key_entry = QLineEdit()
        self.remote_key_entry.setMaximumWidth(line_edit_max_width)
        self.key_button = QPushButton("Browse", self)
        self.key_button.setFixedHeight(button_height)
        self.key_button.setFixedWidth(button_width)
        self.key_button.setStyleSheet(button_style)
        self.key_button.clicked.connect(self.browse_ssh_key)
        self.remote_key_pass_label = QLabel("Key Password:")
        self.remote_key_pass_entry = QLineEdit()
        self.remote_key_pass_entry.setMaximumWidth(line_edit_max_width)
        self.remote_key_pass_entry.setEchoMode(QLineEdit.EchoMode.Password)
        self.remote_address_row = QHBoxLayout()
        self.remote_address_row.addWidget(self.remote_address_label)
        self.remote_address_row.addWidget(self.remote_address_entry)
        self.remote_address_row.addWidget(self.remote_config_label)
        self.remote_address_row.addWidget(self.remote_config_entry)
        self.remote_address_row.addWidget(self.remote_config_button)
        self.remote_address_row.addStretch()
        self.left_layout.addRow(self.remote_address_row)
        self.show_hide_widgets(self.remote_address_row, show=False)
        self.remote_info_row = QHBoxLayout()
        self.remote_info_row.addWidget(self.remote_user_label)
        self.remote_info_row.addWidget(self.remote_user_entry)
        self.remote_info_row.addWidget(self.remote_key_label)
        self.remote_info_row.addWidget(self.remote_key_entry)
        self.remote_info_row.addWidget(self.key_button)
        self.remote_info_row.addWidget(self.remote_key_pass_label)
        self.remote_info_row.addWidget(self.remote_key_pass_entry)
        self.remote_info_row.addStretch()
        self.left_layout.addRow(self.remote_info_row)
        self.show_hide_widgets(self.remote_info_row, show=False)
        self.local_mode_combo.currentTextChanged.connect(self.toggle_remote_row)

    def add_line_widgets(self):
        line_edit_max_width = 400  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
        # Line Mode
        self.line_mode_checkbox = QCheckBox("Line Mode")
        self.line_mode_checkbox.stateChanged.connect(self.toggle_line_mode_widgets)
        self.line_mode_row = QHBoxLayout()
        self.line_mode_row.addWidget(self.line_mode_checkbox)
        self.line_mode_row.addStretch()
        self.left_layout.addRow(self.line_mode_row)
        self.line_index_label = QLabel("Select Line Indices (space-separated):")
        self.line_index_label.setFixedWidth(self.max_label_width)
        self.line_index_entry = QLineEdit()
        self.redshift_label = QLabel("Redshift (space-separated):")
        self.redshift_label.setFixedWidth(self.max_label_width)
        self.redshift_entry = QLineEdit()
        self.redshift_entry.setMaximumWidth(line_edit_max_width)
        self.num_lines_label = QLabel("Number of Lines:")
        self.num_lines_label.setFixedWidth(self.max_label_width)
        self.num_lines_entry = QLineEdit()
        self.num_lines_entry.setMaximumWidth(line_edit_max_width)
        self.non_line_mode_row1 = QHBoxLayout()
        self.non_line_mode_row1.addWidget(self.redshift_label)
        self.non_line_mode_row1.addWidget(self.redshift_entry)
        self.non_line_mode_row1.addStretch()
        self.left_layout.addRow(self.non_line_mode_row1)
        self.non_line_mode_row2 = QHBoxLayout()
        self.non_line_mode_row2.addWidget(self.num_lines_label)
        self.non_line_mode_row2.addWidget(self.num_lines_entry)
        self.non_line_mode_row2.addStretch()
        self.left_layout.addRow(self.non_line_mode_row2)
        self.show_hide_widgets(self.non_line_mode_row1, show=False)
        self.show_hide_widgets(self.non_line_mode_row2, show=False)

    def add_width_slider(self):
        line_edit_max_width = 250  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
        # Line Width
        szp = QtW.QSizePolicy.Maximum
        self.line_width_label = QtW.QLabel("Line Widths in Km/s:")
        self.line_width_label.setFixedWidth(self.max_label_width)
        self.line_width_label.setSizePolicy(szp, szp)
        self.line_width_value_label_min = QLabel("Min: 50 km/s")
        self.line_width_value_label_max = QLabel("Max: 1700 km/s")
        self.line_width_slider = QRangeSlider(QtCore.Qt.Horizontal)
        self.line_width_slider.setRange(50, 1700)  # Set min and max range for the slider
        self.line_width_slider.setValue((100, 500))# Initial values for min/max
        self.line_width_slider.setMinimumSize(line_edit_max_width, 20)
        self.line_width_slider.setMaximumWidth(line_edit_max_width + button_width)
        self.line_width_slider.valueChanged.connect(self.update_width_labels)
        self.line_width_row = QHBoxLayout()
        self.line_width_row.addWidget(self.line_width_label)
        self.line_width_row.addWidget(self.line_width_value_label_min)
        self.line_width_row.addWidget(self.line_width_slider)
        self.line_width_row.addWidget(self.line_width_value_label_max)
        self.line_width_row.addStretch()
        self.left_layout.addRow(self.line_width_row)

    def add_robust_slider(self):
        line_edit_max_width = 300  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
        # Briggs Robustness
        self.robust_label = QLabel("Briggs Robustness:")
        self.robust_slider = QSlider(Qt.Orientation.Horizontal)
        self.robust_value_label = QLabel(f"{self.robust_slider.value()}")
        self.robust_slider.setRange(-20, 20)
        self.robust_slider.setTickInterval(1)
        self.robust_slider.setSingleStep(1)
        self.robust_slider.setValue(0)
        self.robust_slider.setMaximumWidth(line_edit_max_width + button_width)
        self.robust_slider.valueChanged.connect(self.update_robust_label)
        self.robust_row = QHBoxLayout()
        self.robust_row.addWidget(self.robust_label)
        self.robust_row.addWidget(self.robust_slider)
        self.robust_row.addWidget(self.robust_value_label)
        self.left_layout.addRow(self.robust_row)
    
    def add_dim_widgets(self):
        line_edit_max_width = 400  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
        # Set SNR
        self.snr_checkbox = QCheckBox("Set SNR")
        self.snr_entry = QLineEdit()
        self.snr_entry.setMaximumWidth(line_edit_max_width)
        self.snr_entry.setVisible(False)
        self.snr_checkbox.stateChanged.connect(
            lambda: self.toggle_dim_widgets_visibility(self.snr_entry)
        )

        # Set Infrared Luminosity
        self.ir_luminosity_checkbox = QCheckBox("Set IR Luminosity")
        self.ir_luminosity_entry = QLineEdit()
        self.ir_luminosity_entry.setMaximumWidth(line_edit_max_width)
        self.ir_luminosity_entry.setVisible(False)
        self.ir_luminosity_checkbox.stateChanged.connect(
            lambda: self.toggle_dim_widgets_visibility(self.ir_luminosity_entry)
        )

        # Fix Spatial Dimensions
        self.fix_spatial_checkbox = QCheckBox("Fix Spatial Dimensions")
        self.n_pix_entry = QLineEdit()
        self.n_pix_entry.setMaximumWidth(line_edit_max_width)
        self.n_pix_entry.setVisible(False)
        self.fix_spatial_checkbox.stateChanged.connect(
            lambda: self.toggle_dim_widgets_visibility(self.n_pix_entry)
        )

        # Fix Spectral Dimensions
        self.fix_spectral_checkbox = QCheckBox("Fix Spectral Dimensions")
        self.n_channels_entry = QLineEdit()
        self.n_channels_entry.setMaximumWidth(line_edit_max_width)
        self.n_channels_entry.setVisible(False)
        self.fix_spectral_checkbox.stateChanged.connect(
            lambda: self.toggle_dim_widgets_visibility(self.n_channels_entry)
        )
        
        # Serendipitous Sources
        self.serendipitous_checkbox = QCheckBox("Serendipitous Sources")

        self.chechbox_row = QHBoxLayout()
        self.chechbox_row_2 = QHBoxLayout()
        self.chechbox_row.addWidget(self.snr_checkbox)
        self.chechbox_row.addWidget(self.snr_entry)
        self.chechbox_row.addWidget(self.ir_luminosity_checkbox)
        self.chechbox_row.addWidget(self.ir_luminosity_entry)
        self.chechbox_row_2.addWidget(self.fix_spatial_checkbox)
        self.chechbox_row_2.addWidget(self.n_pix_entry)
        self.chechbox_row_2.addWidget(self.fix_spectral_checkbox)
        self.chechbox_row_2.addWidget(self.n_channels_entry)
        self.chechbox_row_2.addWidget(self.serendipitous_checkbox)
        self.chechbox_row.addStretch()
        self.chechbox_row_2.addStretch()
        self.left_layout.addRow(self.chechbox_row)
        self.left_layout.addRow(self.chechbox_row_2)

    def add_model_widgets(self):
        line_edit_max_width = 400  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
        self.model_label = QLabel("Select Model:")
        self.model_label.setFixedWidth(self.max_label_width)
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
        self.model_combo.currentTextChanged.connect(self.toggle_tng_api_key_row)
        self.model_row = QHBoxLayout()
        self.model_row.addWidget(self.model_label)
        self.model_row.addWidget(self.model_combo)
        self.model_row.addStretch()
        self.left_layout.addRow(self.model_row)

        self.tng_api_key_label = QLabel("TNG API Key:")
        self.tng_api_key_label.setFixedWidth(self.max_label_width)
        self.tng_api_key_entry = QLineEdit()
        self.tng_api_key_entry.setMaximumWidth(line_edit_max_width)
        self.tng_api_key_row = QHBoxLayout()
        self.tng_api_key_row.addWidget(self.tng_api_key_label)
        self.tng_api_key_row.addWidget(self.tng_api_key_entry)
        self.tng_api_key_row.addStretch()
        self.show_hide_widgets(self.tng_api_key_row, show=False)
        self.left_layout.addRow(self.tng_api_key_row)

    def add_meta_widgets(self):
        line_edit_max_width = 400  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
         # Metadata 
        self.metadata_mode_label = QLabel("Metadata Mode:")
        self.metadata_mode_label.setFixedWidth(self.max_label_width)
        self.metadata_mode_combo = QComboBox()
        self.metadata_mode_combo.addItems(["get", "query"])
        self.metadata_mode_combo.currentTextChanged.connect(self.toggle_metadata_browse)
        self.metadata_mode_row = QHBoxLayout()
        self.metadata_mode_row.addWidget(self.metadata_mode_label)
        self.metadata_mode_row.addWidget(self.metadata_mode_combo)
        self.metadata_mode_row.addStretch()
        self.left_layout.addRow(self.metadata_mode_row)

    def add_metadata_widgets(self):
        line_edit_max_width = 400  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
        self.metadata_path_label = QLabel("Metadata Path:")
        self.metadata_path_label.setFixedWidth(self.max_label_width)
        self.metadata_path_entry = QLineEdit()
        self.metadata_path_entry.setMaximumWidth(line_edit_max_width)
        self.metadata_path_button = QPushButton("Browse")
        self.metadata_path_button.setFixedHeight(button_height)
        self.metadata_path_button.setFixedWidth(button_width)
        self.metadata_path_button.setStyleSheet(button_style)
        
        self.metadata_path_button.clicked.connect(self.browse_metadata_path)
        self.metadata_path_row = QHBoxLayout()
        self.metadata_path_row.addWidget(self.metadata_path_label)
        self.metadata_path_row.addWidget(self.metadata_path_entry)
        self.metadata_path_row.addWidget(self.metadata_path_button)
        self.metadata_path_row.addStretch()
        self.left_layout.addRow(self.metadata_path_row)
        self.left_layout.update()

    def add_metadata_query_widgets(self):
        line_edit_max_width = 400  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
        science_keyword_label = QLabel(
            "Select Science Keyword by number (space-separated):"
        )
        self.science_keyword_entry = QLineEdit()  # Use QLineEdit instead of input
        self.science_keyword_entry.setMaximumWidth(line_edit_max_width)
        scientific_category_label = QLabel(
            "Select Scientific Category by number (space-separated):"
        )
        self.scientific_category_entry = QLineEdit()
        self.scientific_category_entry.setMaximumWidth(line_edit_max_width)

        band_label = QLabel("Select observing bands (space-separated):")
        self.band_entry = QLineEdit()
        self.band_entry.setMaximumWidth(line_edit_max_width)

        fov_label = QLabel("Select FOV range (min max) or max only (space-separated):")
        self.fov_entry = QLineEdit()
        self.fov_entry.setMaximumWidth(line_edit_max_width)

        time_resolution_label = QLabel(
            "Select integration time  range (min max) or max only (space-separated):"
        )
        self.time_resolution_entry = QLineEdit()
        self.time_resolution_entry.setMaximumWidth(line_edit_max_width)

        frequency_label = QLabel(
            "Select source frequency range (min max) or max only (space-separated):"
        )
        self.frequency_entry = QLineEdit()
        self.frequency_entry.setMaximumWidth(line_edit_max_width)

        self.continue_query_button = QPushButton("Continue Query")
        self.continue_query_button.clicked.connect(self.execute_query)
        self.continue_query_button.setFixedHeight(button_height)
        self.continue_query_button.setFixedWidth(2 * button_width)
        self.continue_query_button.setStyleSheet(button_style)

        # Create layouts and add widgets
        self.science_keyword_row = QHBoxLayout()
        self.science_keyword_row.addWidget(science_keyword_label)
        self.science_keyword_row.addWidget(self.science_keyword_entry)
        self.science_keyword_row.addStretch()

        self.scientific_category_row = QHBoxLayout()
        self.scientific_category_row.addWidget(scientific_category_label)
        self.scientific_category_row.addWidget(self.scientific_category_entry)
        self.scientific_category_row.addStretch()

        self.band_row = QHBoxLayout()
        self.band_row.addWidget(band_label)
        self.band_row.addWidget(self.band_entry)
        self.band_row.addStretch()

        self.fov_row = QHBoxLayout()
        self.fov_row.addWidget(fov_label)
        self.fov_row.addWidget(self.fov_entry)
        self.fov_row.addStretch()

        self.time_resolution_row = QHBoxLayout()
        self.time_resolution_row.addWidget(time_resolution_label)
        self.time_resolution_row.addWidget(self.time_resolution_entry)
        self.time_resolution_row.addStretch()

        self.frequency_row = QHBoxLayout()
        self.frequency_row.addWidget(frequency_label)
        self.frequency_row.addWidget(self.frequency_entry)
        self.frequency_row.addStretch()

        #self.continue_query_row = QHBoxLayout()
        self.query_save_row.addWidget(self.continue_query_button)

        # Insert rows into left_layout (adjust index if needed)
        self.left_layout.addRow(self.science_keyword_row)
        self.left_layout.addRow(self.scientific_category_row)
        self.left_layout.addRow(self.band_row)
        self.left_layout.addRow(self.fov_row)
        self.left_layout.addRow(self.time_resolution_row)
        self.left_layout.addRow(self.frequency_row)
        self.terminal.add_log(
            "\n\nFill out the fields and click 'Continue Query' to proceed."
        )
        self.query_execute_button.hide()  # Hide 
        self.continue_query_button.show()  # Show

    def add_query_widgets(self):
        line_edit_max_width = 400  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
        self.query_type_label = QLabel("Query Type:")
        self.query_type_label.setFixedWidth(self.max_label_width)
        self.query_type_combo = QComboBox()
        self.query_type_combo.addItems(["science", "target"])
        self.query_type_combo.currentTextChanged.connect(self.update_query_save_label)
        self.query_type_row = QHBoxLayout()
        self.query_type_row.addWidget(self.query_type_label)
        self.query_type_row.addWidget(self.query_type_combo)
        self.query_type_row.addStretch()
        self.left_layout.addRow(self.query_type_row)

        self.target_list_label = QLabel("Load Target List:")
        self.target_list_entry = QLineEdit()
        self.target_list_entry.setMaximumWidth(line_edit_max_width)
        self.target_list_button = QPushButton("Browse")
        self.target_list_button.setFixedHeight(button_height)
        self.target_list_button.setFixedWidth(button_width)
        self.target_list_button.setStyleSheet(button_style)
        self.target_list_button.clicked.connect(self.browse_target_list)
        self.target_list_row = QHBoxLayout()
        self.target_list_row.addWidget(self.target_list_label)
        self.target_list_row.addWidget(self.target_list_entry)
        self.target_list_row.addWidget(self.target_list_button)
        self.target_list_row.addStretch()
        self.left_layout.addRow(self.target_list_row)
        self.show_hide_widgets(self.target_list_row, show=False)
        
        self.query_save_label = QLabel("Save Metadata to:")
        self.query_save_entry = QLineEdit()
        self.query_save_entry.setMaximumWidth(line_edit_max_width)
        self.query_save_button = QPushButton("Browse")
        self.query_save_button.setFixedHeight(button_height)
        self.query_save_button.setFixedWidth(button_width)
        self.query_save_button.setStyleSheet(button_style)
        self.query_save_button.clicked.connect(self.select_metadata_path)
        self.query_execute_button = QPushButton("Execute Query")
        self.query_execute_button.setFixedHeight(button_height)
        self.query_execute_button.setFixedWidth(2 * button_width)
        self.query_execute_button.setStyleSheet(button_style)
        self.query_execute_button.clicked.connect(self.execute_query)
        self.query_save_row = QHBoxLayout()
        self.query_save_row.addWidget(self.query_save_label)
        self.query_save_row.addWidget(self.query_save_entry)
        self.query_save_row.addWidget(self.query_save_button)
        self.query_save_row.addStretch()
        self.query_save_row.addWidget(self.query_execute_button)
        self.query_save_row.addStretch()
        self.left_layout.addRow(self.query_save_row)
        #self.left_layout.addRow(self.query_execute_button)

    def add_footer_buttons(self):
        line_edit_max_width = 400  # Example width (you can adjust this)
        button_width = 80
        button_height = 20
        border_radius = 10 
        button_style = f"""
            QPushButton {{
                background-color: #5DADE2;
                color: white;
                border-radius: {border_radius}px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #3498DB;
            }}
        """
        # Start / Stop Simulation
        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        self.start_button.setFixedHeight(button_height)
        self.start_button.setFixedWidth(2 * button_width)
        self.start_button.setStyleSheet(button_style)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setFixedHeight(button_height)
        self.stop_button.setFixedWidth(2 * button_width)
        self.stop_button.setStyleSheet(button_style)

        self.reset_button = QPushButton("Reset Fields")
        self.reset_button.clicked.connect(self.reset_fields)
        self.reset_button.setFixedHeight(button_height)
        self.reset_button.setFixedWidth(2 * button_width)
        self.reset_button.setStyleSheet(button_style)

        self.button_row = QHBoxLayout()
        self.button_row.addWidget(self.start_button)
        self.button_row.addWidget(self.stop_button)
        self.button_row.addWidget(self.reset_button)
        self.button_row.addStretch()
        # Create a layout for the buttons
        self.footer_layout = QHBoxLayout()
        self.footer_layout.addWidget(self.start_button)
        self.footer_layout.addWidget(self.stop_button)
        self.footer_layout.addWidget(self.reset_button)

        # Add the footer layout to the bottom of the left_content_layout
        self.left_content_layout.addLayout(self.footer_layout)
    
    # Remove Widgets Functions 
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
            self.metadata = None 
    
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

    # Toggle Widgets Functions
    def toggle_config_label(self):
        if self.remote_mode_combo.currentText() == "SLURM":
            self.remote_config_label.setText("Slurm Config:")
        elif self.remote_mode_combo.currentText() == "PBS":
            self.remote_config_label.setText("PBS Config:")
        else:
            self.remote_config_label.setText("MPI Config")

    def toggle_remote_folder_line(self):
        if self.remote_folder_checkbox.isChecked():
            self.remote_folder_line.show()
        else:
            self.remote_folder_line.hide()

    def toggle_remote_folder(self):
        if self.remote_folder_checkbox.isChecked():
            self.remote_folder_line.show()
        else:
            self.remote_folder_line.hide()

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
            self.remote_folder_line.hide()
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
            self.line_mode_row.addStretch()
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
        
    def toggle_dim_widgets_visibility(self, widget):
        widget.setVisible(self.sender().isChecked())

    def toggle_tng_api_key_row(self):
        """Shows/hides the TNG API key row based on the selected model."""
        if self.model_combo.currentText() == "Extended":
            self.show_hide_widgets(self.tng_api_key_row, show=True)
        else:
            self.show_hide_widgets(self.tng_api_key_row, show=False)

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
        self.left_layout.update()
    
    
    # -------- Metadata Query Functions ---------------------
    def on_query_finished(self):
        self.terminal.add_log("Query finished!")

    def execute_query(self):
        self.terminal.add_log("Executing query...")
        if self.metadata_mode_combo.currentText() == "query":
            query_type = self.query_type_combo.currentText()
            if query_type == "science":
                if (
                    not hasattr(self, "metadata_query_widgets_added")
                    or not self.metadata_query_widgets_added
                ):
                    runnable = QueryKeyword(self)
                    runnable.finished.connect(self.on_query_finished)
                    self.thread_pool.start(runnable)
                    self.add_metadata_query_widgets()
                    self.metadata_query_widgets_added = True
                   
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


    # -------- Simulation Functions -------------------------
    def start_simulation(self):
        return
    
    def stop_simulation(self):
        # Implement the logic to stop the simulation
        self.stop_simulation_flag = True
        self.progress_bar_entry.setText("Simulation Stopped")
        self.update_progress_bar(0)
        self.terminal.add_log("# ------------------------------------- #\n")


    def reset_fields(self):
        self.output_entry.clear()
        self.tng_entry.clear()
        self.galaxy_zoo_entry.clear()
        self.galaxy_zoo_checkbox.setChecked(False)
        self.hubble_entry.clear()
        self.hubble_checkbox.setChecked(False)
        self.ncpu_entry.clear()
        self.n_sims_entry.clear()
        self.metadata_path_entry.clear()
        if self.local_mode_combo.currentText() == "remote":
            self.remote_address_entry.clear()
            self.remote_user_entry.clear()
            self.remote_key_entry.clear()
            self.remote_key_pass_entry.clear()
            self.remote_config_entry.clear()
            self.remote_mode_combo.setCurrentText("MPI")
            self.remote_folder_line.clear()
            self.remote_folder_checkbox.setChecked(False)
        if self.metadata_mode_combo.currentText() == "query":
            self.query_save_entry.clear()
        self.metadata_mode_combo.setCurrentText("get")
        self.project_name_entry.clear()
        self.save_format_combo.setCurrentText("npz")
        self.redshift_entry.clear()
        self.num_lines_entry.clear()
        self.snr_checkbox.setChecked(False)
        #self.line_width_slider.setValue(200)
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
    # -------- Load and Save functions -----------------------
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
        self.n_sims_entry.setText(self.settings.value("n_sims", ""))
        self.ncpu_entry.setText(self.settings.value("ncpu", ""))
        self.metadata_mode_combo.setCurrentText(
            self.settings.value("metadata_mode", "")
        )
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
                                runnable.finished.connect(
                                    self.on_download_finished
                                )  # Connect signal
                                self.thread_pool.start(runnable)
                                self.terminal.add_log(
                                    "Waiting for download to finish..."
                                )
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
    
    
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ALMASimulator()
    window.show()
    sys.exit(app.exec())