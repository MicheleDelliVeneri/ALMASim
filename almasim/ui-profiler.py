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
        #self.stop_simulation_flag = False
        #self.remote_simulation_finished = True
        #self.terminal.add_log(f"Setting file path is {self.settings_path}")

    # -------- Widgets and UI -------------------------
    def initialize_ui(self):
        self.setWindowTitle("ALMASim: Set up your simulation parameters")
        self.setMinimumSize(800, 600)  # Set minimum size for laptop usage

        # Create main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.build_left_layout()
        self.build_right_layout()
        main_layout.addLayout(self.left_layout)
        main_layout.addLayout(self.right_layout)


    def build_right_layout(self):
        self.right_layout = QVBoxLayout()
        self.term = QTextEdit()
        self.term.setReadOnly(True)
        self.terminal = TerminalLogger(self.term)
        self.right_layout.addWidget(self.term)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)


    def build_left_layout(self):
        self.left_layout = QFormLayout()
        self.left_layout.setSpacing(10)
        self.left_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.left_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.left_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        self.left_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        # Output Directory
        self.output_label = QLabel("Output Directory:")
        self.output_entry = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.browse_output_directory)
        self.output_row = QHBoxLayout()
        self.output_row.addWidget(self.output_entry)
        self.output_row.addWidget(self.output_entry)
        self.output_row.addWidget(self.output_button)
        self.left_layout.addRow(self.output_row)

        # TNG Directory
        self.tng_label = QLabel("TNG Directory:")
        self.tng_entry = QLineEdit()
        self.tng_button = QPushButton("Browse")
        self.tng_button.clicked.connect(self.browse_tng_directory)
        self.tng_row = QHBoxLayout()
        self.tng_row.addWidget(self.tng_label)
        self.tng_row.addWidget(self.tng_entry)
        self.tng_row.addWidget(self.tng_button)
        self.left_layout.addRow(self.tng_row)

        # Galaxy Zoo Directory
        self.galaxy_zoo_label = QLabel("Galaxy Zoo Directory:")
        self.galaxy_zoo_entry = QLineEdit()
        self.galaxy_zoo_button = QPushButton("Browse")
        self.galaxy_zoo_button.clicked.connect(self.browse_galaxy_zoo_directory)
        self.galaxy_zoo_checkbox = QCheckBox("Get Galaxy Zoo")
        self.galaxy_row = QHBoxLayout()
        self.galaxy_row.addWidget(self.galaxy_zoo_label)
        self.galaxy_row.addWidget(self.galaxy_zoo_entry)
        self.galaxy_row.addWidget(self.galaxy_zoo_button)
        self.galaxy_row.addWidget(self.galaxy_zoo_checkbox)
        self.left_layout.addRow(self.galaxy_row)

        # Hubble Top 100 Directory
        self.hubble_label = QLabel("Hubble Top 100 Directory:")
        self.hubble_entry = QLineEdit()
        self.hubble_button = QPushButton("Browse")
        self.hubble_button.clicked.connect(self.browse_hubble_directory)
        self.hubble_checkbox = QCheckBox("Get Hubble Data")
        self.hubble_row = QHBoxLayout()
        self.hubble_row.addWidget(self.hubble_label)
        self.hubble_row.addWidget(self.hubble_entry)
        self.hubble_row.addWidget(self.hubble_button)
        self.hubble_row.addWidget(self.hubble_checkbox)
        self.left_layout.addRow(self.hubble_row)

        # Project Name
        self.project_name_label = QLabel("Project Name:")
        self.project_name_entry = QLineEdit()
        self.project_row = QHBoxLayout()
        self.project_row.addWidget(self.project_name_label)
        self.project_row.addWidget(self.project_name_entry)
        self.left_layout.addRow(self.project_row)

        # Number of Simulaitons 
        self.n_sims_label = QLabel("Number of Simulations:")
        self.n_sims_entry = QLineEdit()
        self.n_sims_row = QHBoxLayout()
        self.n_sims_row.addWidget(self.n_sims_label)
        self.n_sims_row.addWidget(self.n_sims_entry)
        self.left_layout.addRow(self.n_sims_row)

        # Number of Cores
        self.ncpu_label = QLabel("Number of Cores:")
        self.ncpu_entry = QLineEdit()
        self.ncpu_row = QHBoxLayout()
        self.ncpu_row.addWidget(self.ncpu_label)
        self.ncpu_row.addWidget(self.ncpu_entry)
        self.left_layout.addRow(self.ncpu_row)


        # Save Format 
        self.save_format_label = QLabel("Save Format:")
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["npz", "fits", "h5"])
        self.save_format_row = QHBoxLayout()
        self.save_format_row.addWidget(self.save_format_label)
        self.save_format_row.addWidget(self.save_format_combo)
        self.left_layout.addRow(self.save_format_row)
        
        # Local / Remote Mode
        self.local_mode_label = QLabel("Local / Remote Mode:")
        self.local_mode_combo = QComboBox()
        self.local_mode_combo.addItems(["local", "remote"])
        self.left_layout.addRow(self.local_mode_label, self.local_mode_combo)
        self.local_mode_combo.currentTextChanged.connect(self.toggle_local_remote)
        
        # Remote Compute Mode 
        self.remote_mode_label = QLabel("Remote Compute Mode:")
        self.remote_mode_combo = QComboBox()
        self.remote_mode_combo.addItems(["MPI", "SLURM", "PBS"])
        self.remote_folder_checkbox = QCheckBox("Set Working Directory")
        self.remote_folder_line = QLineEdit()
        self.remote_folder_checkbox.stateChanged.connect(self.toggle_remote_folder)
        self.remote_mode_combo.currentTextChanged.connect(self.toggle_config)
        self.left_layout.addRow(self.local_mode_label, self.local_mode_combo, self.remote_mode_label, self.remote_mode_combo, self.remote_folder_checkbox, self.remote_folder_line)
        
        # Remote info 
        self.remote_address_label = QLabel("Remote Host:")
        self.remote_address_entry = QLineEdit()
        self.remote_config_label = QLabel("SLURM Config:")
        self.remote_config_entry = QLineEdit()
        self.remote_config_button = QPushButton("Browse", self)
        self.remote_config_button.clicked.connect(self.browse_slurm_config)
        self.remote_user_label = QLabel("User:")
        self.remote_user_entry = QLineEdit()
        self.remote_key_label = QLabel("SSH Key:")
        self.remote_key_entry = QLineEdit()
        self.key_button = QPushButton("Browse", self)
        self.key_button.clicked.connect(self.browse_ssh_key)
        self.remote_key_pass_label = QLabel("Key Password:")
        self.remote_key_pass_entry = QLineEdit()
        self.remote_key_pass_entry.setEchoMode(QLineEdit.EchoMode.Password)
        self.left_layout.addRow(
            self.remote_address_label, self.remote_address_entry)
        self.left_layout.addRow(
            self.remote_user_label, self.remote_user_entry)
        self.left_layout.addRow(
            self.remote_key_label, self.remote_key_entry, self.key_button)
        self.left_layout.addRow(
            self.remote_key_pass_label, self.remote_key_pass_entry)
        self.left_layout.addRow(
            self.remote_config_label, self.remote_config_entry, self.remote_config_button)
        
        # Line Mode
        self.line_mode_checkbox = QCheckBox("Line Mode")
        self.line_mode_checkbox.stateChanged.connect(self.toggle_line_mode)
        self.line_index_label = QLabel("Select Line Indices (space-separated):")
        self.line_index_entry = QLineEdit()
        self.redshift_label = QLabel("Redshift (space-separated):")
        self.redshift_entry = QLineEdit()
        self.num_lines_label = QLabel("Number of Lines:")
        self.num_lines_entry = QLineEdit()
        self.left_layout.addRow(self.line_mode_checkbox)
        self.left_layout.addRow(self.line_index_label, self.line_index_entry)
        self.left_layout.addRow(self.redshift_label, self.redshift_entry)
        self.left_layout.addRow(self.num_lines_label, self.num_lines_entry)
        
        # Line Width
        self.line_width_label = QLabel("Line Widths in Km/s:")
        self.line_width_value_label_min = QLabel("Min: 200 km/s")
        self.line_width_value_label_max = QLabel("Max: 400 km/s")
        self.line_width_slider = QRangeSlider()
        self.line_width_slider.setMin(50)    # Minimum value
        self.line_width_slider.setMax(1700)  # Maximum value
        self.line_width_slider.setRange(200, 400)  # Initial values for min/max
        self.line_width_slider.setTickInterval(25)
        self.line_width_slider.rangeChanged.connect(self.update_width_labels)
        self.left_layout.addRow(self.line_width_value_label_min, self.line_width_slider, self.line_width_value_label_max)
        
        # Briggs Robustness
        self.robust_label = QLabel("Briggs Robustness:")
        self.robust_slider = QSlider(Qt.Orientation.Horizontal)
        self.robust_value_label = QLabel(f"{self.robust_slider.value()}")
        self.robust_slider.setRange(-20, 20)
        self.robust_slider.setTickInterval(1)
        self.robust_slider.setSingleStep(1)
        self.robust_slider.setValue(0)
        self.robust_slider.valueChanged.connect(self.update_robust_label)
        self.left_layout.addRow(self.robust_label, self.robust_slider, self.robust_value_label)

        # Set SNR
        self.snr_checkbox = QCheckBox("Set SNR")
        self.snr_entry = QLineEdit()
        self.snr_checkbox.stateChanged.connect(self.toggle_snr)

        # Set Infrared Luminosity
        self.ir_luminosity_checkbox = QCheckBox("Set IR Luminosity")
        self.ir_luminosity_entry = QLineEdit()
        self.ir_luminosity_checkbox.stateChanged.connect(self.toggle_ir_luminosity)

        # Fix Spatial Dimensions
        self.fix_spatial_checkbox = QCheckBox("Fix Spatial Dimensions")
        self.n_pix_entry = QLineEdit()
        self.fix_spatial_checkbox.stateChanged.connect(self.toggle_spatial)

        # Fix Spectral Dimensions
        self.fix_spectral_checkbox = QCheckBox("Fix Spectral Dimensions")
        self.n_channels_entry = QLineEdit()
        self.fix_spectral_checkbox.stateChanged.connect(self.toggle_spectral)
        
        # Serendipitous Sources
        self.serendipitous_checkbox = QCheckBox("Serendipitous Sources")

        self.left_layout.addRow(
            self.snr_checkbox, self.snr_entry, 
            self.ir_luminosity_checkbox, self.ir_luminosity_entry,
            self.fix_spatial_checkbox, self.n_pix_entry,
            self.fix_spectral_checkbox, self.n_channels_entry,
            self.serendipitous_checkbox
            )

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
        self.model_combo.currentTextChanged.connect(self.toggle_tng_api)
        self.left_layout.addRow(self.model_label, self.model_combo)
        self.tng_api_key_label = QLabel("TNG API Key:")
        self.tng_api_key_entry = QLineEdit()
        self.left_layout.addRow(self.tng_api_key_label, self.tng_api_key_entry)
        
        # Metadata 
        self.metadata_mode_label = QLabel("Metadata Mode:")
        self.metadata_mode_combo = QComboBox()
        self.metadata_mode_combo.addItems(["get", "query"])
        self.metadata_mode_combo.currentTextChanged.connect(self.toggle_metadata_mode)
        self.metadata_path_label = QLabel("Metadata Path:")
        self.metadata_path_entry = QLineEdit()
        self.metadata_path_button = QPushButton("Browse")
        self.metadata_path_button.clicked.connect(self.browse_metadata_path)
        self.left_layout.addRow(
            self.metadata_mode_label, self.metadata_mode_combo)
        self.left_layout.addRow(
            self.metadata_path_label, self.metadata_path_entry, self.metadata_path_button
        )
        # Query 
        self.query_type_label = QLabel("Query Type:")
        self.query_type_combo = QComboBox()
        self.query_type_combo.addItems(["science", "target"])
        self.query_type_combo.currentTextChanged.connect(self.toggle_query_type)
        self.query_save_label = QLabel("Save Metadata to:")
        self.query_save_entry = QLineEdit()
        self.query_save_button = QPushButton("Browse")
        self.query_save_button.clicked.connect(self.select_metadata_path)
        self.query_execute_button = QPushButton("Execute Query")
        self.query_execute_button.clicked.connect(self.execute_query)
        self.target_list_label = QLabel("Load Target List:")
        self.target_list_entry = QLineEdit()
        self.target_list_button = QPushButton("Browse")
        self.target_list_button.clicked.connect(self.browser_target_list)
        self.left_layout.addRow(
            self.query_type_label, self.query_type_combo)
        self.left_layout.addRow(
            self.query_save_label, self.query_save_entry, self.query_save_button
        )
        self.left_layout.addRow(
            self.target_list_label, self.target_list_entry, self.target_list_button
        )
        self.left_layout.addRow(self.query_execute_button)


        # Start / Stop Simulation
        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.stop_simulation)

        self.reset_button = QPushButton("Reset Fields")
        self.reset_button.clicked.connect(self.reset_fields)

        self.botton_row = QHBoxLayout()
        self.botton_row.addWidget(self.start_button)
        self.botton_row.addWidget(self.stop_button)
        self.botton_row.addWidget(self.reset_button)
        self.button_row.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.button_row.addStretch()
        self.left_layout.addRow(self.botton_row)

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
            
    # -------- UI Toggle Functions -------------------------
    def toggle_config_label(self):
        if self.remote_mode_combo.currentText() == "SLURM":
            self.remote_config_label.setText("Slurm Config:")
        elif self.remote_mode_combo.currentText() == "PBS":
            self.remote_config_label.setText("PBS Config:")
        else:
            self.remote_config_label.setText("MPI Config")

    def toggle_remote_folder(self):
        if self.remote_folder_checkbox.isChecked():
            self.remote_folder_line.show()
        else:
            self.remote_folder_line.hide()

    def toggle_remote_row(self):
        if self.local_mode_combo.currentText() == "remote":
            self.toggle_config_label()
            self.remote_address_label.show()
            self.remote_address_entry.show()
            self.remote_user_label.show()
            self.remote_user_entry.show()
            self.remote_key_label.show()
            self.remote_key_entry.show()
            self.key_button.show()
            self.remote_key_pass_label.show()
            self.remote_key_pass_entry.show()
            self.remote_mode_label.show()
            self.remote_mode_combo.show()
            self.remote_config_label.show()
            self.remote_config_entry.show()
            self.remote_config_button.show()
            self.remote_folder_checkbox.show()
            self.remote_folder_line.show()
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
            self.remote_address_label.hide()
            self.remote_address_entry.hide()
            self.remote_user_label.hide()
            self.remote_user_entry.hide()
            self.remote_key_label.hide()
            self.remote_key_entry.hide()
            self.key_button.hide()
            self.remote_key_pass_label.hide()
            self.remote_key_pass_entry.hide()
            self.remote_mode_label.hide()
            self.remote_mode_combo.hide()
            self.remote_config_label.hide()
            self.remote_config_entry.hide()
            self.remote_config_button.hide()
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
    
    def toggle_line_mode(self):
        if self.line_mode_checkbox.isChecked():
            self.line_index_label.show()
            self.line_index_entry.show()
            self.redshift_label.hide()
            self.redshift_entry.hide()
            self.num_lines_label.hide()
            self.num_lines_entry.hide()
            if self.line_displayed is False:
                self.line_display()
        else:
            self.line_index_label.hide()
            self.line_index_entry.hide()
            self.redshift_label.show()
            self.redshift_entry.show()
            self.num_lines_label.show()
            self.num_lines_entry.show()
        
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

    def metadata_path_set(self):
        metadata_path = self.metadata_path_entry.text()
        self.load_metadata(metadata_path)
    
    def update_width_labels(self, min_value, max_value):
        # Update the labels when the range slider values change
        self.line_width_value_label_min.setText(f"Min: {min_value} km/s")
        self.line_width_value_label_max.setText(f"Max: {max_value} km/s")

    def update_robust_label(self, value):
        self.robust_value_label.setText(f"{value / 10}")
    
    def toggle_snr(self):
        if self.snr_checkbox.isChecked():
            self.snr_entry.show()
        else:
            self.snr_entry.hide()

    def toggle_ir_luminosity(self):
        if self.ir_luminosity_checkbox.isChecked():
            self.ir_luminosity_entry.show()
        else:
            self.ir_luminosity_entry.hide()
    
    def toggle_spatial(self):
        if self.fix_spatial_checkbox.isChecked():
            self.n_pix_entry.show()
        else:
            self.n_pix_entry.hide()
    
    def toggle_spectral(self):
        if self.fix_spectral_checkbox.isChecked():
            self.n_channels_entry.show()
        else:
            self.n_channels_entry.hide()
    
    def toggle_tng_api(self):
        if self.model_combo.currentText() == "Extended":
            self.tng_api_key_entry.show()
        else:
            self.tng_api_key_entry.hide()

    def toggle_metadata_mode(self):
        if self.metadata_mode_combo.currentText() == 'get':
            self.metadata_path_label.show()
            self.metadata_path_button.show()
            self.metadata_path_entry.show()
        else:
            self.metadata_path_label.hide()
            self.metadata_path_button.hide()
            self.metadata_path_entry.hide()

    def toggle_query_type(self):
        if self.query_type_combo.currentText() == 'science':
            self.query_save_label.show()
            self.query_save_entry.show()
            self.query_save_button.show()
            self.target_list_label.hide()
            self.target_list_entry.hide()
            self.target_list_button.hide()
        else:
            self.query_save_label.hide()
            self.query_save_entry.hide()
            self.query_save_button.hide()
            self.target_list_label.show()
            self.target_list_entry.show()
            self.target_list_button.show()
    
    # -------- Metadata Query Functions ---------------------
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
    
    def execute_query(self):
        self.terminal.add_log("Executing query...")
        if self.metadata_mode_combo.currentText() == "query":
            query_type = self.query_type_combo.currentText()
            if query_type == "science":
                self.show_scientific_keywords()




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
            self.remote_folder_line.clear()
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
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ALMASimulator()
    window.show()
    sys.exit(app.exec())