import sys
import numpy as np
import pandas as pd
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QScrollArea, QGridLayout, 
    QGroupBox, QCheckBox, QRadioButton, QButtonGroup, QSizePolicy, QCheckBox, QSplitter,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QMessageBox, QPlainTextEdit  
)
from PyQt6.QtCore import QSettings, QIODevice, QTextStream, QProcess, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QGuiApplication
from kaggle import api
from os.path import isfile
import dask
from distributed import Client, LocalCluster, WorkerPlugin
from dask_jobqueue import SLURMCluster
import json
import astropy.units as U
from astropy.constants import c
from astropy.time import Time
import math
from math import pi, ceil
from datetime import date
import time
import shutil
from time import strftime, gmtime
import paramiko
import pysftp
import plistlib
import psutil
import dask.dataframe as dd
import utility.alma as ual
import utility.astro as uas
import utility.compute as uc
import utility.skymodels as usm
import utility.plotting as upl
import utility.interferometer as uin


class LogView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._process = QProcess()
        self._process.readyReadStandardOutput.connect(self.handle_stdout)
        self._process.readyReadStandardError.connect(self.handle_stderr)

    def start_log(self, program, arguments=None):
        if arguments is None:
            arguments = []
        self._process.start(program, arguments)

    def add_log(self, message):
        self.appendPlainText(message.rstrip())
        self.ensureCursorVisible()  # Ensure the cursor is visible after adding text
        QApplication.processEvents()  # Process all pending events to force UI update

    def handle_stdout(self):
        message = self._process.readAllStandardOutput().data().decode()
        self.add_log(message)

    def handle_stderr(self):
        message = self._process.readAllStandardError().data().decode()
        self.add_log(message)

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
                scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.AspectRatioMode.KeepAspectRatio)  # Scale with aspect ratio
                label.setPixmap(scaled_pixmap)

    def create_science_keyword_plots(self):
        """Creates and displays plots of science keyword distributions in the window."""
        try:
            plot_dir = os.path.join(os.getcwd(), "plots")  # Get the directory for plots
            expected_plots = [
                'science_vs_bands.png', 'science_vs_int_time.png', 
                'science_vs_FoV.png', 'science_vs_source_freq.png'
            ]

            # Check if plots need to be generated
            if not all(os.path.exists(os.path.join(plot_dir, plot)) for plot in expected_plots):
                self.plot_science_keywords_distributions(os.getcwd())  # Generate plots if not found

            row, col = 0, 0
            for plot_file in expected_plots:  # Iterate through the expected plot files
                plot_path = os.path.join(plot_dir, plot_file)

                pixmap = QPixmap()
                if not pixmap.load(plot_path):  # Load the image for the plot
                    self.terminal.add_log(f"Error loading plot: {plot_path}")  
                    continue 

                max_width = self.width() // 2
                max_height = self.height() // 2

                scaled_pixmap = pixmap.scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio)  # Scale the image while maintaining aspect ratio

                label = QLabel()
                label.setPixmap(scaled_pixmap)
                label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)  # Allow resizing
                self.scroll_layout.addWidget(label, row, col)
                self.scroll_layout.setRowStretch(row, 1)
                self.scroll_layout.setColumnStretch(col, 1)

                col += 1
                if col == 2:  # 2 columns per row
                    col = 0
                    row += 1

            self.adjustSize() 
        
        except Exception as e:  # Catch any potential exceptions
            self.terminal.add_log(f"Error in create_science_keyword_plots: {e}")  # Log the error

class ALMASimulatorUI(QMainWindow):
    settings_file = None
    ncpu_entry = None
    terminal = None
    def __init__(self):
        super().__init__()
        self.settings = QSettings("INFN Section of Naples", "ALMASim")
        if ALMASimulatorUI.settings_file is not None:
            with open(ALMASimulatorUI.settings_file, 'rb') as f:
                settings_data = plistlib.load(f)
                for key, value in settings_data.items():
                    self.settings.setValue(key, value)
            self.on_remote = True
        else:
            self.on_remote = False
        
        self.settings_path = self.settings.fileName()
        self.initialize_ui()
        self.terminal.add_log('Setting file path is {}'.format(self.settings_path))
        
    # -------- Widgets and UI -------------------------
    def initialize_ui(self):
        self.setWindowTitle("ALMASim: set up your simulation parameters")

        # --- Create Widgets ---
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


        #self.terminal = QPlainTextEdit()
        #self.terminal.setReadOnly(True)
        self.terminal = LogView()
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
        self.left_layout.addStretch(1)
        self.left_layout.addLayout(button_row)

        right_layout.addWidget(self.terminal)
        
        main_layout.addLayout(self.left_layout)
        main_layout.addLayout(right_layout)
        main_layout.setStretch(0, 3)  # left_layout stretch factor
        main_layout.setStretch(1, 2)  # right_layout stretch factor

        self.line_displayed = False
        self.add_folder_widgets()
        self.add_line_widgets()
        self.add_dim_widgets()
        self.add_model_widgets()
        self.add_meta_widgets()
        self.add_query_widgets()
        # Load saved settings
        if self.on_remote is True:
            self.load_settings_on_remote()
        else:
            self.load_settings()
        self.terminal.start_log("")
        # Check metadata mode on initialization
        self.toggle_line_mode_widgets()
        self.metadata_mode_combo.currentTextChanged.connect(self.toggle_metadata_browse)
        if self.metadata_path_entry.text() != "" and isfile(self.metadata_path_entry.text()):
            self.load_metadata(self.metadata_path_entry.text())
        current_mode = self.metadata_mode_combo.currentText()
        self.toggle_metadata_browse(current_mode)  # Call here
        self.set_window_size()
        ALMASimulatorUI.populate_class_variables(self.terminal, self.ncpu_entry)
    
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
            window_height
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
        self.project_name_label = QLabel("Project Name:")
        self.project_name_entry = QLineEdit()
        # 5
        self.n_sims_label = QLabel("Number of Simulations:")
        self.n_sims_entry = QLineEdit()
        # 6
        self.ncpu_label = QLabel("N. CPUs / Processes:")
        self.ncpu_entry = QLineEdit()
        # 7
        self.save_format_label = QLabel("Save Format:")
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["npz", "fits", 'h5'])
        # 8
        self.comp_mode_label = QLabel("Computation Mode:")
        self.comp_mode_combo = QComboBox()
        self.comp_mode_combo.addItems(["sequential", "parallel"])

        # 9
        self.local_mode_label = QLabel('Local or Remote:')
        self.local_mode_combo = QComboBox()
        self.local_mode_combo.addItems(["local", 'remote'])

        self.remote_mode_label = QLabel('Mode:')
        self.remote_mode_combo = QComboBox()
        self.remote_mode_combo.addItems(['MPI', 'SLURM', 'PBS'])
        self.remote_folder_checkbox = QCheckBox("Set Work Directory:")
        self.remote_dir_line = QLineEdit()
        self.remote_folder_checkbox.stateChanged.connect(self.toggle_remote_dir_line)
        
        self.remote_address_label = QLabel('Remote Host:')
        self.remote_address_entry = QLineEdit()
        self.remote_config_label = QLabel('Slurm Config:')
        self.remote_config_entry = QLineEdit()
        self.remote_config_button = QPushButton('Browse', self)
        self.remote_config_button.clicked.connect(self.browse_slurm_config)
        self.remote_user_label = QLabel('Username')
        self.remote_user_entry = QLineEdit()
        self.remote_key_label = QLabel('SSH Key:')
        self.remote_key_entry = QLineEdit()
        self.key_button = QPushButton("Browse", self)
        self.key_button.clicked.connect(self.browse_ssh_key)
        self.remote_key_pass_label = QLabel('Key Passwd:')
        self.remote_key_pass_entry = QLineEdit()
        self.remote_key_pass_entry.setEchoMode(QLineEdit.EchoMode.Password)
        
        ## 10  
        #self.flux_mode_label = QLabel('Flux Simulation Mode:')
        #self.flux_mode_combo = QComboBox()
        #self.flux_mode_combo.addItems(["direct", 'line-ratios'])
        
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
        galaxy_row.addWidget(self.galaxy_zoo_label)
        galaxy_row.addWidget(self.galaxy_zoo_entry)
        galaxy_row.addWidget(self.galaxy_zoo_button)
        self.left_layout.insertLayout(3, galaxy_row)

        # Project Name Row
        project_name_row = QHBoxLayout()
        project_name_row.addWidget(self.project_name_label)
        project_name_row.addWidget(self.project_name_entry)
        self.left_layout.insertLayout(4, project_name_row)

        # Number of Simulations Row
        n_sims_row = QHBoxLayout()
        n_sims_row.addWidget(self.n_sims_label)
        n_sims_row.addWidget(self.n_sims_entry)
        self.left_layout.insertLayout(5, n_sims_row)

        # Number of CPUs Row
        ncpu_row = QHBoxLayout()
        ncpu_row.addWidget(self.ncpu_label)
        ncpu_row.addWidget(self.ncpu_entry)
        self.left_layout.insertLayout(6, ncpu_row)

        # Save format Row
        save_format_row = QHBoxLayout()
        save_format_row.addWidget(self.save_format_label)
        save_format_row.addWidget(self.save_format_combo)
        self.left_layout.insertLayout(7, save_format_row)

        # Computation Mode Row
        comp_mode_row = QHBoxLayout()
        comp_mode_row.addWidget(self.comp_mode_label)
        comp_mode_row.addWidget(self.comp_mode_combo)
        self.left_layout.insertLayout(8, comp_mode_row)
        

        # Local Mode Row
        local_mode_row = QHBoxLayout()
        local_mode_row.addWidget(self.local_mode_label)
        local_mode_row.addWidget(self.local_mode_combo)
        local_mode_row.addWidget(self.remote_mode_label)
        local_mode_row.addWidget(self.remote_mode_combo)
        local_mode_row.addWidget(self.remote_folder_checkbox)
        local_mode_row.addWidget(self.remote_dir_line)
        self.left_layout.insertLayout(9, local_mode_row)
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
        self.left_layout.insertLayout(10, self.remote_address_row)
        self.show_hide_widgets(self.remote_address_row, show=False)

        self.remote_info_row = QHBoxLayout()
        self.remote_info_row.addWidget(self.remote_user_label)
        self.remote_info_row.addWidget(self.remote_user_entry)
        self.remote_info_row.addWidget(self.remote_key_label)
        self.remote_info_row.addWidget(self.remote_key_entry)
        self.remote_info_row.addWidget(self.key_button)
        self.remote_info_row.addWidget(self.remote_key_pass_label)
        self.remote_info_row.addWidget(self.remote_key_pass_entry)
        self.left_layout.insertLayout(11, self.remote_info_row)
        self.show_hide_widgets(self.remote_info_row, show=False)
        self.local_mode_combo.currentTextChanged.connect(self.toggle_remote_row)

    def toggle_config_label(self):
        if self.remote_mode_combo.currentText() == 'SLURM':
            self.remote_config_label.setText('Slurm Config:')
        elif self.remote_mode_combo.currentText() == 'PBS':
            self.remote_config_label.setText('PBS Config:')
        else:
            self.remote_config_label.setText('MPI Config')

    def toggle_remote_row(self):
        if self.local_mode_combo.currentText() == 'remote':
            self.show_hide_widgets(self.remote_address_row, show=True)
            self.show_hide_widgets(self.remote_info_row, show=True)
            self.toggle_config_label()
            self.remote_mode_label.show()
            self.remote_mode_combo.show()
            self.remote_folder_checkbox.show()
        else:
            self.show_hide_widgets(self.remote_address_row, show=False)
            self.show_hide_widgets(self.remote_info_row, show=False)
            self.remote_mode_label.hide()
            self.remote_mode_combo.hide()
            self.remote_folder_checkbox.hide()
            self.remote_dir_line.hide()
    
    def toggle_remote_dir_line(self):
        if self.remote_folder_checkbox.isChecked():
            self.remote_dir_line.show()
        else:
            self.remote_dir_line.hide()

    def add_line_widgets(self): 
        self.line_mode_checkbox = QCheckBox("Line Mode")
        self.line_mode_checkbox.stateChanged.connect(self.toggle_line_mode_widgets)
        #self.left_layout.insertWidget(8, self.line_mode_checkbox) 
        # Widgets for Line Mode
        self.line_index_label = QLabel('Select Line Indices (space-separated):')
        self.line_index_entry = QLineEdit()
        self.line_mode_row = QHBoxLayout()
        self.line_mode_row.addWidget(self.line_mode_checkbox)
        self.left_layout.insertLayout(12, self.line_mode_row)    # Insert at the end
        # Widgets for Non-Line Mode
        redshift_label = QLabel('Redshifts (space-separated):')
        self.redshift_entry = QLineEdit()
        num_lines_label = QLabel('Number of Lines to Simulate:')
        self.num_lines_entry = QLineEdit()
        self.non_line_mode_row1 = QHBoxLayout()
        self.non_line_mode_row1.addWidget(redshift_label)
        self.non_line_mode_row1.addWidget(self.redshift_entry)
        self.non_line_mode_row2 = QHBoxLayout()
        self.non_line_mode_row2.addWidget(num_lines_label)
        self.non_line_mode_row2.addWidget(self.num_lines_entry)
        self.left_layout.insertLayout(13, self.non_line_mode_row1) # Insert at the end
        self.left_layout.insertLayout(14, self.non_line_mode_row2) # Insert at the end
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
            if self.line_displayed == False:
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
    
    def add_dim_widgets(self):
        # --- Set SNR ---
        self.snr_checkbox = QCheckBox("Set SNR")
        self.snr_entry = QLineEdit()
        self.snr_entry.setVisible(False) 
        self.snr_checkbox.stateChanged.connect(lambda: self.toggle_dim_widgets_visibility(self.snr_entry))

        # --- Set Infrared Luminosity ---
        self.ir_luminosity_checkbox = QCheckBox("Set IR Luminosity")
        self.ir_luminosity_entry = QLineEdit()
        self.ir_luminosity_entry.setVisible(False)
        self.ir_luminosity_checkbox.stateChanged.connect(lambda: self.toggle_dim_widgets_visibility(self.ir_luminosity_entry))

        # --- Fix Spatial Dimension Checkbox and Field ---
        self.fix_spatial_checkbox = QCheckBox("Fix Spatial Dim")
        self.n_pix_entry = QLineEdit()
        self.n_pix_entry.setVisible(False)
        self.fix_spatial_checkbox.stateChanged.connect(lambda: self.toggle_dim_widgets_visibility(self.n_pix_entry))

        # --- Fix Spectral Dimension Checkbox and Field ---
        self.fix_spectral_checkbox = QCheckBox("Fix Spectral Dim")
        self.n_channels_entry = QLineEdit()
        self.n_channels_entry.setVisible(False)
        self.fix_spectral_checkbox.stateChanged.connect(lambda: self.toggle_dim_widgets_visibility(self.n_channels_entry))


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
        self.left_layout.insertLayout(15, checkbox_row)

    def add_model_widgets(self):
        self.model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Point", "Gaussian", "Extended", 'Diffuse', 'Galaxy Zoo', 'Hubble 100', 'Molecular'])
        self.model_row = QHBoxLayout()
        self.model_row.addWidget(self.model_label)
        self.model_row.addWidget(self.model_combo)
        self.left_layout.insertLayout(16, self.model_row)
        self.tng_api_key_label = QLabel("TNG API Key:")
        self.tng_api_key_entry = QLineEdit()
        self.tng_api_key_row = QHBoxLayout()
        self.tng_api_key_row.addWidget(self.tng_api_key_label)
        self.tng_api_key_row.addWidget(self.tng_api_key_entry)

        # Initially hide the TNG API key row
        self.show_hide_widgets(self.tng_api_key_row, show=False)

        self.left_layout.insertLayout(17, self.tng_api_key_row)  # Insert after model_row
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
        self.left_layout.insertLayout(18, self.metadata_mode_row)

    def add_metadata_widgets(self):
        self.metadata_path_label = QLabel("Metadata Path:")
        self.metadata_path_entry = QLineEdit()
        self.metadata_path_button = QPushButton("Browse")
        self.metadata_path_button.clicked.connect(self.browse_metadata_path)
        self.metadata_path_row = QHBoxLayout()
        self.metadata_path_row.addWidget(self.metadata_path_label)
        self.metadata_path_row.addWidget(self.metadata_path_entry)
        self.metadata_path_row.addWidget(self.metadata_path_button)
        self.left_layout.insertLayout(19, self.metadata_path_row)
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
        self.target_list_button.clicked.connect(self.browse_target_list)  # Add function for browsing
        self.target_list_row = QHBoxLayout()
        self.target_list_row.addWidget(self.target_list_label)
        self.target_list_row.addWidget(self.target_list_entry)
        self.target_list_row.addWidget(self.target_list_button)
        #self.target_list_row.hide()  # Initially hide the row
        self.show_hide_widgets(self.target_list_row, show=False)

        # Insert layouts at the correct positions
        self.left_layout.insertLayout(19, self.query_type_row)
        self.left_layout.insertLayout(20, self.target_list_row)  # Insert target list row
        self.left_layout.insertLayout(21, self.query_save_row)
        self.left_layout.insertWidget(22, self.query_execute_button)
        self.query_type_combo.currentTextChanged.connect(self.update_query_save_label)

    def remove_metadata_query_widgets(self):
        # Similar to remove_query_widgets from the previous response, but remove
        # all the rows and widgets added in add_metadata_query_widgets.
        widgets_to_remove = [
            self.science_keyword_row, self.scientific_category_row, self.band_row,
            self.fov_row, self.time_resolution_row,  self.frequency_row
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

            layout.removeItem(self.metadata_path_row)  # Remove the row layout from its parent
            self.metadata_path_row.setParent(None)  # Set the parent to None
            self.metadata = None  # Clear any loaded metadata

    def toggle_metadata_browse(self, mode):
        if mode == "get":
            if self.metadata_path_row.parent() is None:  # Check if already added
                #self.left_layout.insertLayout(8, self.metadata_path_row) # Re-insert at correct position
                #self.left_layout.update()  # Force layout update to show the row
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
            layout.removeItem(self.query_type_row)   # Remove the row
            self.query_type_row.setParent(None)      # Disassociate the row from parent
            
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
            layout = self.left_layout # Directly access the main vertical layout 
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
        self.ncpu_entry.clear()
        self.n_sims_entry.clear()
        self.metadata_path_entry.clear()
        self.comp_mode_combo.setCurrentText("sequential")
        if self.local_mode_combo.currentText() == 'remote':
            self.remote_address_entry.clear()
            self.remote_user_entry.clear()
            self.remote_key_entry.clear()
            self.remote_key_pass_entry.clear()
            self.remote_config_entry.clear()
            self.remote_mode_combo.setCurrentText('MPI')
            self.remote_dir_line.clear()
            self.remote_folder_checkbox.setChecked(False)
        self.local_mode_combo.setCurrentText('local')
        if self.metadata_mode_combo.currentText() == 'query':
            self.query_save_entry.clear()
        self.metadata_mode_combo.setCurrentText("get")
        self.project_name_entry.clear()
        self.save_format_combo.setCurrentText("npz")
        self.redshift_entry.clear()
        self.num_lines_entry.clear()
        self.snr_checkbox.setChecked(False)
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
        self.n_sims_entry.setText(self.settings.value("n_sims", ""))
        self.ncpu_entry.setText(self.settings.value("ncpu", ""))
        self.metadata_mode_combo.setCurrentText(self.settings.value("metadata_mode", ""))
        self.comp_mode_combo.setCurrentText(self.settings.value("comp_mode", ""))
        self.local_mode_combo.setCurrentText(self.settings.value("local_mode", ""))
        if self.local_mode_combo.currentText() == "remote":
            self.remote_address_entry.setText(self.settings.value("remote_address", ""))
            self.remote_user_entry.setText(self.settings.value("remote_user", ""))
            self.remote_key_entry.setText(self.settings.value("remote_key", ""))
            self.remote_key_pass_entry.setText(self.settings.value('remote_key_pass', ""))
            self.remote_config_entry.setText(self.settings.value('remote_config', ''))
            self.remote_mode_combo.setCurrentText(self.settings.value('remote_mode', ''))
            remote_folder = self.settings.value('remote_folder', False, type=bool)
            self.remote_folder_checkbox.setChecked(remote_folder)
            if remote_folder:
                self.remote_dir_line.setText(self.settings.value('remote_dir', ''))
        self.metadata_path_entry.setText(self.settings.value("metadata_path", ""))
        self.project_name_entry.setText(self.settings.value("project_name", ""))
        self.save_format_combo.setCurrentText(self.settings.value("save_format", ""))
        if self.metadata_mode_combo.currentText() == "get" and self.metadata_path_entry.text() != '':
            self.load_metadata(self.metadata_path_entry.text())
        elif self.metadata_mode_combo.currentText() == "query":
            self.query_save_entry.setText(self.settings.value("query_save_entry", ""))
        if self.galaxy_zoo_entry.text() != '':
            if self.local_mode_combo.currentText() == 'local':
                if os.path.exists(self.galaxy_zoo_entry.text()):
                    if not os.path.exists(os.path.join(self.galaxy_zoo_entry.text(), 'images_gz2')):
                        print('Downloading Galaxy Zoo')
                        self.download_galaxy_zoo()
            else:
                if self.remote_address_entry.text() != '' and self.remote_user_entry.text() != '' and self.remote_key_entry.text() != '':
                    self.download_galaxy_zoo_on_remote()
        line_mode = self.settings.value("line_mode", False, type=bool)
        self.tng_api_key_entry.setText(self.settings.value("tng_api_key", ""))
        self.line_mode_checkbox.setChecked(line_mode)
        if line_mode:
            self.line_index_entry.setText(self.settings.value("line_indices", ""))
        else:
            # Load non-line mode values
            self.redshift_entry.setText(self.settings.value("redshifts", ""))
            self.num_lines_entry.setText(self.settings.value("num_lines", ""))
        self.snr_entry.setText(self.settings.value("snr", ""))
        self.snr_checkbox.setChecked(self.settings.value("set_snr", False, type=bool))
        self.fix_spatial_checkbox.setChecked(self.settings.value("fix_spatial", False, type=bool))
        self.n_pix_entry.setText(self.settings.value("n_pix", ""))
        self.fix_spectral_checkbox.setChecked(self.settings.value("fix_spectral", False, type=bool))
        self.n_channels_entry.setText(self.settings.value("n_channels", ""))
        self.serendipitous_checkbox.setChecked(self.settings.value("inject_serendipitous", False, type=bool))
        self.model_combo.setCurrentText(self.settings.value("model", ""))
        self.tng_api_key_entry.setText(self.settings.value("tng_api_key", ""))
        self.toggle_tng_api_key_row()
        self.ir_luminosity_checkbox.setChecked(self.settings.value("set_ir_luminosity", False, type=bool))
        self.ir_luminosity_entry.setText(self.settings.value("ir_luminosity", ""))

    def load_settings_on_remote(self):
        self.output_entry.setText(self.settings.value("output_directory", ""))
        self.tng_entry.setText(self.settings.value("tng_directory", ""))
        self.galaxy_zoo_entry.setText(self.settings.value("galaxy_zoo_directory", ""))
        self.n_sims_entry.setText(self.settings.value("n_sims", ""))
        self.ncpu_entry.setText(self.settings.value("ncpu", ""))
        self.metadata_path_entry.setText("")
    
    @classmethod
    def populate_class_variables(cls, terminal, ncpu_entry):
        cls.terminal = terminal
        cls.ncpu_entry = ncpu_entry

    def closeEvent(self, event):
        if hasattr(self, "pool") and self.pool:
            self.pool.close()  # Signal to the pool to stop accepting new tasks
            self.pool.join()   # Wait for all tasks to complete
        self.settings.setValue("output_directory", self.output_entry.text())
        self.settings.setValue("tng_directory", self.tng_entry.text())
        self.settings.setValue("galaxy_zoo_directory", self.galaxy_zoo_entry.text())
        self.settings.setValue('n_sims', self.n_sims_entry.text())
        self.settings.setValue("ncpu", self.ncpu_entry.text())
        self.settings.setValue("project_name", self.project_name_entry.text())
        if self.metadata_mode_combo.currentText() == "get":
            self.settings.setValue("metadata_path", self.metadata_path_entry.text())
        elif self.metadata_mode_combo.currentText() == "query":
            self.settings.setValue("query_save_entry", self.query_save_entry.text())
        self.settings.setValue("metadata_mode", self.metadata_mode_combo.currentText())
        self.settings.setValue("comp_mode", self.comp_mode_combo.currentText())
        self.settings.setValue("local_mode", self.local_mode_combo.currentText())
        if self.local_mode_combo.currentText() == 'remote':
            self.settings.setValue('remote_address', self.remote_address_entry.text())
            self.settings.setValue('remote_user', self.remote_user_entry.text())
            self.settings.setValue('remote_key', self.remote_key_entry.text())
            self.settings.setValue('remote_key_pass', self.remote_key_pass_entry.text())
            self.settings.setValue('remote_config', self.remote_config_entry.text())
            self.settings.setValue('remote_mode', self.remote_mode_combo.currentText())
            self.settings.setValue('remote_folder', self.remote_dir_line.text())
        self.settings.setValue("save_format", self.save_format_combo.currentText())
        self.settings.setValue("line_mode", self.line_mode_checkbox.isChecked())
        if self.line_mode_checkbox.isChecked():
            self.settings.setValue("line_indices", self.line_index_entry.text())
        else:
            # Save non-line mode values
            self.settings.setValue("redshifts", self.redshift_entry.text())
            self.settings.setValue("num_lines", self.num_lines_entry.text())
        self.settings.setValue('set_snr', self.snr_checkbox.isChecked())
        self.settings.setValue("snr", self.snr_entry.text())
        self.settings.setValue("fix_spatial", self.fix_spatial_checkbox.isChecked())
        self.settings.setValue("n_pix", self.n_pix_entry.text())
        self.settings.setValue("fix_spectral", self.fix_spectral_checkbox.isChecked())
        self.settings.setValue("n_channels", self.n_channels_entry.text())
        self.settings.setValue("inject_serendipitous", self.serendipitous_checkbox.isChecked())
        self.settings.setValue("model", self.model_combo.currentText())
        self.settings.setValue("tng_api_key", self.tng_api_key_entry.text())
        self.settings.setValue("set_ir_luminosity", self.ir_luminosity_checkbox.isChecked())
        self.settings.setValue("ir_luminosity", self.ir_luminosity_entry.text())
        super().closeEvent(event)
  
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
        science_keyword_label = QLabel('Select Science Keyword by number (space-separated):')
        self.science_keyword_entry = QLineEdit()  # Use QLineEdit instead of input

        scientific_category_label = QLabel('Select Scientific Category by number (space-separated):')
        self.scientific_category_entry = QLineEdit()

        band_label = QLabel('Select observing bands (space-separated):')
        self.band_entry = QLineEdit()

        fov_label = QLabel('Select FOV range (min max) or max only (space-separated):')
        self.fov_entry = QLineEdit()

        time_resolution_label = QLabel('Select integration time  range (min max) or max only (space-separated):')
        self.time_resolution_entry = QLineEdit()

        frequency_label = QLabel('Select source frequency range (min max) or max only (space-separated):')
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

        self.time_resolution_row  = QHBoxLayout()
        self.time_resolution_row.addWidget(time_resolution_label)
        self.time_resolution_row.addWidget(self.time_resolution_entry)

        self.frequency_row = QHBoxLayout()
        self.frequency_row.addWidget(frequency_label)
        self.frequency_row.addWidget(self.frequency_entry)

        self.continue_query_row = QHBoxLayout()
        self.continue_query_row.addWidget(self.continue_query_button)

        # Insert rows into left_layout (adjust index if needed)
        self.left_layout.insertLayout(22, self.science_keyword_row)
        self.left_layout.insertLayout(23, self.scientific_category_row)
        self.left_layout.insertLayout(24, self.band_row)
        self.left_layout.insertLayout(25, self.fov_row)
        self.left_layout.insertLayout(26, self.time_resolution_row)
        self.left_layout.insertLayout(27, self.frequency_row)
        self.left_layout.insertWidget(28, self.continue_query_button)
        self.terminal.add_log("\n\nFill out the fields and click 'Continue Query' to proceed.")
        self.query_execute_button.hide()  # Hide the execute query button

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if self.local_mode_combo.currentText() == 'remote' and self.remote_address_entry.text() != "" and self.remote_key_entry.text() != "" and self.remote_user_entry != "":
            if directory: 
                remote_dir = self.map_to_remote_directory(directory)
                if self.remote_key_pass_entry.text() != "":
                    with pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text(), private_key_pass=self.remote_key_pass_entry.text()) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                else:
                    with pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text()) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                self.output_entry.setText(remote_dir)
        else:
            if directory:
                self.output_entry.setText(directory)

    def map_to_remote_directory(self, directory):
        directory_name = directory.split(os.path.sep)[-1]
        if self.remote_dir_line.text() != "":
            if not self.remote_dir_line.text().startswith('/'):
                self.remote_dir_line.setText('/' + self.remote_dir_line.text())
            directory_path = os.path.join(self.remote_dir_line.text(), directory_name)
        else:
            directory_path = os.path.join('/home', self.remote_user_entry.text(), directory_name)
        return directory_path

    def browse_tng_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select TNG Directory")
        if self.local_mode_combo.currentText() == 'remote' and self.remote_address_entry.text() != "" and self.remote_key_entry.text() != "" and self.remote_user_entry != "":
            if directory: 
                remote_dir = self.map_to_remote_directory(directory)
                if self.remote_key_pass_entry.text() != "":
                    with pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text(), private_key_pass=self.remote_key_pass_entry.text()) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                else:
                    with pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text()) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                self.tng_entry.setText(remote_dir)
        else:
            if directory:
                self.tng_entry.setText(directory)

    def browse_galaxy_zoo_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Galaxy Zoo Directory")
        if self.local_mode_combo.currentText() == 'remote' and self.remote_address_entry.text() != "" and self.remote_key_entry.text() != "" and self.remote_user_entry != "":
            if directory: 
                remote_dir = self.map_to_remote_directory(directory)
                if self.remote_key_pass_entry.text() != "":
                    with pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text(), private_key_pass=self.remote_key_pass_entry.text()) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                else:
                    with pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text()) as sftp:
                        if not sftp.exists(remote_dir):
                            sftp.mkdir(remote_dir)
                self.galaxy_zoo_entry.setText(remote_dir)
        else:
            if directory:
                self.galaxy_zoo_entry.setText(directory)

    def browse_metadata_path(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Metadata File", os.path.join(os.getcwd(), 'metadata'), "CSV Files (*.csv)")
        if file:
            self.metadata_path_entry.setText(file)
            self.metadata_path_set()

    def browse_ssh_key(self):
        file_dialog = QFileDialog()
        ssh_key_file, _ = file_dialog.getOpenFileName(self, "Select SSH Key File", os.path.join(os.path.expanduser('~'), '.ssh'), "SSH Key Files (*.pem *.ppk *.key  *rsa)")
        if ssh_key_file:
            self.remote_key_entry.setText(ssh_key_file)

    def browse_slurm_config(self):
        file_dialog = QFileDialog()
        slurm_config_file, _ = file_dialog.getOpenFileName(self, "Select Slurm Config File", os.getcwd(), 'Slurm Config Files (*.json)')
        if slurm_config_file:
            self.remote_config_entry.setText(slurm_config_file)
    
    def select_metadata_path(self):
        file, _ = QFileDialog.getSaveFileName(self, "Select Metadata File", os.path.join(os.getcwd(), 'metadata'), "CSV Files (*.csv)")
        if file:
            self.query_save_entry.setText(file)
            #self.metadata_path_set()
            
    def metadata_path_set(self):
        metadata_path = self.metadata_path_entry.text()
        self.load_metadata(metadata_path)  # Pass only the metadata_path 

    def browse_target_list(self):
        """Opens a file dialog to select the target list file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Target List", "", "CSV Files (*.csv)")
        if file_path:
            self.target_list_entry.setText(file_path)

    # -------- Query ALMA Database Functions -------
    def get_tap_service(self):
        urls = ["https://almascience.eso.org/tap", "https://almascience.nao.ac.jp/tap",
                "https://almascience.nrao.edu/tap"
        ]
        while True:  # Infinite loop to keep trying until successful
            for url in urls:
                try:
                    service = pyvo.dal.TAPService(url)
                    # Test the connection with a simple query to ensure the service is working
                    result = service.search("SELECT TOP 1 * FROM ivoa.obscore")
                    self.terminal.add_log(f"Connected successfully to {url}")
                    return service
                except Exception as e:
                    self.terminal.add_log(f"Failed to connect to {url}: {e}")
                    self.terminal.add_log("Retrying other servers...")
            self.terminal.add_log("All URLs attempted and failed, retrying...")
    
    def get_science_types(self):
        service = self.get_tap_service()
        query = f"""  
                SELECT science_keyword, scientific_category  
                FROM ivoa.obscore  
                WHERE science_observation = 'T'    
                """
        db = service.search(query).to_table().to_pandas()
        science_keywords = db['science_keyword'].unique()
        scientific_category = db['scientific_category'].unique()
        science_keywords = list(filter(lambda x: x != "", science_keywords))
        scientific_category = list(filter(lambda x: x != "", scientific_category))

        unique_keywords = []
        # Iterazione attraverso ogni stringa nella lista
        for keywords_string in science_keywords:
        # Dividi la stringa in base alla virgola e rimuovi gli spazi bianchi
            keywords_list = [keyword.strip() for keyword in keywords_string.split(',')]
        # Aggiungi le parole alla lista dei valori univoci
            unique_keywords.extend(keywords_list)
        # Utilizza il set per ottenere i valori univoci
        unique_keywords = sorted(set(unique_keywords))
        unique_keywords = [keyword for keyword in unique_keywords if (
                            keyword != 'Evolved stars: Shaping/physical structure' and
                            keyword != 'Exoplanets' and 
                            keyword != 'Galaxy structure &evolution')]

        return  unique_keywords, scientific_category
    
    def query_by_science_type(self, science_keyword=None, scientific_category=None, band=None, fov_range=None, time_resolution_range=None, total_time_range=None, frequency_range=None):
        """Query for all science observations of given member OUS UID and target name, selecting all columns of interest.

        Parameters:
        service (pyvo.dal.TAPService): A TAPService instance for querying the database.

        Returns:
        pandas.DataFrame: A table of query results.
        """
        service = self.get_tap_service()
        # Default values for parameters if they are None
        if science_keyword is None:
            science_keyword = ""
        if scientific_category is None:
            scientific_category = ""
        if band is None:
            band = ""

        # Build query components based on the type and content of each parameter
        science_keyword_query = f"science_keyword like '%{science_keyword}%'"
        if isinstance(science_keyword, list):
            if len(science_keyword) == 1:
                science_keyword_query = f"science_keyword like '%{science_keyword[0]}%'"
            else:
                science_keywords = "', '".join(science_keyword)
                science_keyword_query = f"science_keyword in ('{science_keywords}')"

        scientific_category_query = f"scientific_category like '%{scientific_category}%'"
        if isinstance(scientific_category, list):
            if len(scientific_category) == 1:
                scientific_category_query = f"scientific_category like '%{scientific_category[0]}%'"
            else:
                scientific_categories = "', '".join(scientific_category)
                scientific_category_query = f"scientific_category in ('{scientific_categories}')"

        band_query = f"band_list like '%{band}%'"
        if isinstance(band, list):
            if len(band) == 1:
                band_query = f"band_list like '%{band[0]}%'"
            else:
                bands = [str(x) for x in band]
                bands = "', '".join(bands)
                band_query = f"band_list in ('{bands}')"

        # Additional filtering based on ranges
        if fov_range is None:
            fov_query = ""
        else:
            fov_query = f"s_fov BETWEEN {fov_range[0]} AND {fov_range[1]}"
        if time_resolution_range is None:
            time_resolution_query = ""
        else:
            time_resolution_query = f"t_resolution BETWEEN {time_resolution_range[0]} AND {time_resolution_range[1]}"

        if total_time_range is None:
            total_time_query = ""
        else:    
            total_time_query = f"t_max BETWEEN {total_time_range[0]} AND {total_time_range[1]}"

        if frequency_range is None:
            frequency_query = ""
        else:
            frequency_query = f"frequency BETWEEN {frequency_range[0]} AND {frequency_range[1]}"

        # Combine all conditions into one WHERE clause
        conditions = [science_keyword_query, scientific_category_query, band_query, fov_query, time_resolution_query, total_time_query, frequency_query]
        conditions = [cond for cond in conditions if cond]  # Remove empty conditions
        where_clause = " AND ".join(conditions)
        where_clause += " AND is_mosaic = 'F' AND science_observation = 'T'"  # Add fixed conditions

        query = f"""
                SELECT *
                FROM ivoa.obscore
                WHERE {where_clause}
                """

        result = service.search(query).to_table().to_pandas()

        return result

    def plot_science_keywords_distributions(master_path):
        service = self.get_tap_service()
        plot_dir = os.path.join(master_path, "plots")

        # Check if plot directory exists
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            existing_plots = []  # Initialize as empty list if plot directory doesn't exist
        else:
            # Check if plot files already exist
            existing_plots = [f for f in os.listdir(plot_dir) if f.endswith('.png')]

        expected_plots = ['science_vs_bands.png', 'science_vs_int_time.png', 'science_vs_source_freq.png',
                          'science_vs_FoV.png']

        if all(plot_file in existing_plots for plot_file in expected_plots):
            return
        else:
            self.terminal.add_log(f"Generating helping plots to guide you in the scientific query, check them in {plot_dir}.")
            # Identify missing plots
        missing_plots = [plot for plot in expected_plots if plot not in existing_plots]

        # Query only for variables associated with missing plots
        query_variables = set()
        for missing_plot in missing_plots:
            if missing_plot == 'science_vs_bands.png':
                query_variables.update(['science_keyword', 'band_list'])
            elif missing_plot == 'science_vs_int_time.png':
                query_variables.update(['science_keyword', 't_resolution'])
            elif missing_plot == 'science_vs_source_freq.png':
                query_variables.update(['science_keyword', 'frequency'])
            elif missing_plot == 'science_vs_FoV.png':
                query_variables.update(['science_keyword', 'band_list'])
        query = f"""  
                SELECT {', '.join(query_variables)}, member_ous_uid
                FROM ivoa.obscore  
                WHERE science_observation = 'T'
                AND is_mosaic = 'F'
                """

        custom_palette = sns.color_palette("tab20")
        sns.set_palette(custom_palette)
        db = service.search(query).to_table().to_pandas()
        db = db.drop_duplicates(subset='member_ous_uid')

        # Splitting the science keywords at commas
        db['science_keyword'] = db['science_keyword'].str.split(',')
        db['science_keyword'] = db['science_keyword'].apply(lambda x: [y.strip() for y in x])
        db = db.explode('science_keyword')
        db = db.drop(db[db['science_keyword'] == ''].index)
        db = db.drop(db[db['science_keyword'] == 'Exoplanets'].index)
        db = db.drop(db[db['science_keyword'] == 'Galaxy structure &evolution'].index)
        db = db.drop(db[db['science_keyword'] == 'Evolved stars: Shaping/physical structure'].index)
        short_keyword = {
            'Solar system - Trans-Neptunian Objects (TNOs)' : 'Solar System - TNOs',
            'Photon-Dominated Regions (PDR)/X-Ray Dominated Regions (XDR)': 'Photon/X-Ray Domanited Regions',
            'Luminous and Ultra-Luminous Infra-Red Galaxies (LIRG & ULIRG)': 'LIRG & ULIRG',
            'Cosmic Microwave Background (CMB)/Sunyaev-Zel\'dovich Effect (SZE)': 'CMB/Sunyaev-Zel\'dovich Effect',
            'Active Galactic Nuclei (AGN)/Quasars (QSO)': 'AGN/QSO',
            'Inter-Stellar Medium (ISM)/Molecular clouds': 'ISM & Molecular Clouds',
        }

        db['science_keyword'] = db['science_keyword'].replace(short_keyword)

        for missing_plot in missing_plots:
            if missing_plot == 'science_vs_bands.png':
                db['band_list'] = db['band_list'].str.split(' ')
                db['band_list'] = db['band_list'].apply(lambda x: [y.strip() for y in x])
                db = db.explode('band_list')

                db_sk_b = db.groupby(['science_keyword', 'band_list']).size().unstack(fill_value=0)

                plt.rcParams["figure.figsize"] = (28,20)
                db_sk_b.plot(kind='barh', stacked=True, color=custom_palette)
                plt.title('Science Keywords vs. ALMA Bands')
                plt.xlabel('Counts')
                plt.ylabel('Science Keywords')
                plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='ALMA Bands')
                plt.savefig(os.path.join(plot_dir, 'science_vs_bands.png'))
                plt.close()

            elif missing_plot == 'science_vs_int_time.png':
                db = db[db['t_resolution'] <= 3e4]
                time_bins = np.arange(db['t_resolution'].min(), db['t_resolution'].max(), 1000)  # 1000 second bins
                db['time_bin'] = pd.cut(db['t_resolution'], bins=time_bins)

                db_sk_t = db.groupby(['science_keyword', 'time_bin']).size().unstack(fill_value=0)

                plt.rcParams["figure.figsize"] = (28,20)
                db_sk_t.plot(kind='barh', stacked=True)
                plt.title('Science Keywords vs. Integration Time')
                plt.xlabel('Counts')
                plt.ylabel('Science Keywords')
                plt.legend(title='Integration Time', loc='upper left', bbox_to_anchor=(1.01, 1))
                plt.savefig(os.path.join(plot_dir, 'science_vs_int_time.png'))
                plt.close()

            elif missing_plot == 'science_vs_source_freq.png':
                frequency_bins = np.arange(db['frequency'].min(), db['frequency'].max(), 50)  # 50 GHz bins
                db['frequency_bin'] = pd.cut(db['frequency'], bins=frequency_bins)

                db_sk_f = db.groupby(['science_keyword', 'frequency_bin']).size().unstack(fill_value=0)

                plt.rcParams["figure.figsize"] = (28,20)
                db_sk_f.plot(kind='barh', stacked=True, color=custom_palette)
                plt.title('Science Keywords vs. Source Frequency')
                plt.xlabel('Counts')
                plt.ylabel('Science Keywords')
                plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='Frequency')
                plt.savefig(os.path.join(plot_dir, 'science_vs_source_freq.png')) 
                plt.close()

            elif missing_plot == 'science_vs_FoV.png':
                db['band_list'] = db['band_list'].str.split(' ')
                db['band_list'] = db['band_list'].apply(lambda x: [y.strip() for y in x])
                db = db.explode('band_list')
                db['fov'] = db['band_list'].apply(lambda x: get_fov_from_band(int(x)))
                fov_bins = np.arange(db['fov'].min(), db['fov'].max(), 10)  #  10 arcsec bins
                db['fov_bins'] = pd.cut(db['fov'], bins=fov_bins)

                db_sk_fov = db.groupby(['science_keyword', 'fov_bins']).size().unstack(fill_value=0)

                plt.rcParams["figure.figsize"] = (28,20)
                db_sk_fov.plot(kind='barh', stacked=True, color=custom_palette)
                plt.title('Science Keywords vs. FoV')
                plt.xlabel('Counts')
                plt.ylabel('Science Keywords')
                plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='FoV')
                plt.savefig(os.path.join(plot_dir, 'science_vs_FoV.png'))
                plt.close()

    def update_query_save_label(self, query_type):
        """Shows/hides the target list row and query save row based on query type."""
        if query_type == "science":
            self.show_hide_widgets(self.target_list_row, show=False)  # Hide target list row
            self.show_hide_widgets(self.query_save_row, show=True)   # Show query save row
            self.query_save_label.setText("Save Metadata to:")
        else:  # query_type == "target"
            self.show_hide_widgets(self.target_list_row, show=True)   # Show target list row
            self.show_hide_widgets(self.query_save_row, show=True)  # Hide query save row

    def execute_query(self):
        self.terminal.add_log("Executing query...")
        if self.metadata_mode_combo.currentText() == "query":
            query_type = self.query_type_combo.currentText()
            if query_type == "science":
                if not hasattr(self, 'metadata_query_widgets_added') or not self.metadata_query_widgets_added:
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
                    self.terminal.add_log(f'Loading target list {self.target_list_entry.text()}')
                    target_list = pd.read_csv(self.target_list_entry.text())
                    self.target_list = target_list.values.tolist()
                    self.metadata = self.query_for_metadata_by_targets()
            else:
                # Handle invalid query type (optional)
                pass  
    
    def show_scientific_keywords(self):
        # Implement the logic to query metadata based on science type
        self.terminal.add_log("Querying metadata by science type...")
        #self.plot_window = PlotWindow()
        #self.plot_window.show()
        self.science_keywords, self.scientific_categories = ual.get_science_types()
        self.terminal.add_log('Available science keywords:')
        for i, keyword in enumerate(self.science_keywords):
            self.terminal.add_log(f'{i}: {keyword}')
        self.terminal.add_log('\nAvailable scientific categories:')
        for i, category in enumerate(self.scientific_categories):
            self.terminal.add_log(f'{i}: {category}')
    
    def query_for_metadata_by_science_type(self):
        self.terminal.add_log('Querying by Science Keyword')
        science_keyword_number = self.science_keyword_entry.text()
        scientific_category_number = self.scientific_category_entry.text()
        band = self.band_entry.text()
        fov_input = self.fov_entry.text()
        time_resolution_input = self.time_resolution_entry.text()
        frequency_input = self.frequency_entry.text()
        save_to_input = self.query_save_entry.text()
        
        # Get selected science keywords and categories
        science_keyword = [self.science_keywords[int(i)] for i in science_keyword_number.split()] if science_keyword_number else None
        scientific_category = [self.scientific_categories[int(i)] for i in scientific_category_number.split()] if scientific_category_number else None
        bands = [int(x) for x in band.split()] if band else None
        def to_range(text):
            values = [float(x) for x in text.split()] if text else None
            return tuple(values) if values and len(values) > 1 else (0, values[0]) if values else None
        fov_range = to_range(fov_input)
        time_resolution_range = to_range(time_resolution_input)
        frequency_range = to_range(frequency_input)
        df = ual.query_by_science_type(science_keyword, scientific_category, bands, fov_range, time_resolution_range, frequency_range)
        df = df.drop_duplicates(subset='member_ous_uid').drop(df[df['science_keyword'] == ''].index)
        # Rename columns and select relevant data
        rename_columns = {
            'target_name': 'ALMA_source_name',
            'pwv': 'PWV',
            'schedblock_name': 'SB_name',
            'velocity_resolution': 'Vel.res.',
            'spatial_resolution': 'Ang.res.',
            's_ra': 'RA',
            's_dec': 'Dec',
            's_fov': 'FOV',
            't_resolution': 'Int.Time',
            'cont_sensitivity_bandwidth': 'Cont_sens_mJybeam',
            'sensitivity_10kms': 'Line_sens_10kms_mJybeam',
            'obs_release_date': 'Obs.date',
            'band_list': 'Band',
            'bandwidth': 'Bandwidth',
            'frequency': 'Freq',
            'frequency_support': 'Freq.sup.'
        }
        df.rename(columns=rename_columns, inplace=True)
        database = df[['ALMA_source_name', 'Band', 'PWV', 'SB_name', 'Vel.res.', 'Ang.res.', 'RA', 'Dec', 'FOV', 'Int.Time',
                      'Cont_sens_mJybeam', 'Line_sens_10kms_mJybeam', 'Obs.date', 'Bandwidth', 'Freq',
                       'Freq.sup.', 'antenna_arrays', 'proposal_id', 'member_ous_uid', 'group_ous_uid']]
        database.loc[:, 'Obs.date'] = database['Obs.date'].apply(lambda x: x.split('T')[0])
        database.to_csv(save_to_input, index=False)
        self.metadata = database
        self.terminal.add_log(f"Metadata saved to {save_to_input}")
        del database

    def query_for_metadata_by_targets(self):
        """Query for metadata for all predefined targets and compile the results into a single DataFrame.

        Parameters:
        service (pyvo.dal.TAPService): A TAPService instance for querying the database.
        targets (list of tuples): A list where each tuple contains (target_name, member_ous_uid).
        path (str): The path to save the results to.

        Returns:
        pandas.DataFrame: A DataFrame containing the results for all queried targets.
        """
        # Query all targets and compile the results
        self.terminal.add_log("Querying metadata from target list...")
        df = ual.query_all_targets(self.target_list)
        df = df.drop_duplicates(subset='member_ous_uid')
        save_to_input = self.query_save_entry.text()
        # Define a dictionary to map existing column names to new names with unit initials
        rename_columns = {
            'target_name': 'ALMA_source_name',
            'pwv': 'PWV',
            'schedblock_name': 'SB_name',
            'velocity_resolution': 'Vel.res.',
            'spatial_resolution': 'Ang.res.',
            's_ra': 'RA',
            's_dec': 'Dec',
            's_fov': 'FOV',
            't_resolution': 'Int.Time',
            'cont_sensitivity_bandwidth': 'Cont_sens_mJybeam',
            'sensitivity_10kms': 'Line_sens_10kms_mJybeam',
            'obs_release_date': 'Obs.date',
            'band_list': 'Band',
            'bandwidth': 'Bandwidth',
            'frequency': 'Freq',
            'frequency_support': 'Freq.sup.'
        }
        df.rename(columns=rename_columns, inplace=True)
        database = df[['ALMA_source_name', 'Band', 'PWV', 'SB_name', 'Vel.res.', 'Ang.res.', 'RA', 'Dec', 'FOV', 'Int.Time',
                      'Cont_sens_mJybeam', 'Line_sens_10kms_mJybeam', 'Obs.date', 'Bandwidth', 'Freq',
                       'Freq.sup.', 'antenna_arrays', 'proposal_id', 'member_ous_uid', 'group_ous_uid']]
        database.loc[:, 'Obs.date'] = database['Obs.date'].apply(lambda x: x.split('T')[0])
        database.to_csv(save_to_input, index=False)
        self.metadata = database
        self.terminal.add_log(f"Metadata saved to {save_to_input}")
        
    # ----- Auxiliary Functions -----------------

    def load_metadata(self, metadata_path):
        try:
            self.terminal.add_log(f"Loading metadata from {metadata_path}")
            self.metadata = pd.read_csv(metadata_path)
            self.terminal.add_log('Metadata contains {} samples'.format(len(self.metadata)))

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
        
        path_line_emission_csv = os.path.join(os.getcwd(), 'brightnes', 'calibrated_lines.csv')
        db_line = uas.read_line_emission_csv(path_line_emission_csv, sep=',').sort_values(by='Line')
        line_names = db_line['Line'].values
        rest_frequencies = db_line['freq(GHz)'].values
        self.terminal.add_log('Please choose the lines from the following list\n')
        for i in range(len(line_names)):
            self.terminal.add_log(f'{i}: {line_names[i]} - {rest_frequencies[i]:.2e} GHz\n')
        self.line_displayed = True

    def download_galaxy_zoo(self):
        """
        Downloads a Kaggle dataset to the specified path.
        """
        self.terminal.add_log('\nGalaxy Zoo data not found on disk, downloading from Kaggle...')
        api.authenticate()  # Authenticate with your Kaggle credentials
        dataset_name = 'jaimetrickz/galaxy-zoo-2-images'
        # Download the dataset as a zip file
        api.dataset_download_files(dataset_name, path=self.galaxy_zoo_entry.text(), unzip=True)
        self.terminal.add_log(f"\nDataset {dataset_name} downloaded to {self.galaxy_zoo_entry.text()}")

    def download_galaxy_zoo_on_remote(self):
        """
        Downloads a Kaggle dataset to the specified path.
        """
        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text(), private_key_pass=self.remote_key_pass_entry.text())
                
        else:
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text())
        if sftp.exists(self.galaxy_zoo_entry.text()):
            if not sftp.listdir(self.galaxy_zoo_entry.text()):
                self.terminal.add_log('\nGalaxy Zoo data not found on disk, downloading from Kaggle...')
                if not sftp.exists('/home/{}/.kaggle'.format(self.remote_user_entry.text())):
                    sftp.mkdir('/home/{}/.kaggle'.format(self.remote_user_entry.text()))
                if not sftp.exists('/home/{}/.kaggle/kaggle.json'.format(self.remote_user_entry.text())):
                    sftp.put(os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json'), '/home/{}/.kaggle/kaggle.json'.format(self.remote_user_entry.text()))
                    sftp.chmod('/home/{}/.kaggle/kaggle.json'.format(self.remote_user_entry.text()), 600)
                if self.remote_key_pass_entry.text() != "":
                    key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text(), password=self.remote_key_pass_entry.text())
                else:
                    key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text())
                venv_dir = os.path.join('/home/{}/'.format(self.remote_user_entry.text()), 'almasim_env')
                commands = f"""
                source {venv_dir}/bin/activate
                python -c "from kaggle import api; api.dataset_download_files('jaimetrickz/galaxy-zoo-2-images', path='{self.galaxy_zoo_entry.text()}', unzip=True)"
                """     
                paramiko_client = paramiko.SSHClient()
                paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                paramiko_client.connect(self.remote_address_entry.text(), username=self.remote_user_entry.text(), pkey=key)
                stdin, stdout, stderr = paramiko_client.exec_command(commands)
                self.terminal.add_log(stdout.read().decode())
                self.terminal.add_log(stderr.read().decode())

    def check_tng_dirs(self):
        tng_dir = self.tng_entry.text()
        if not os.path.exists(os.path.join(tng_dir, 'TNG100-1')):
            os.makedirs(os.path.join(tng_dir, 'TNG100-1'))
        if not os.path.exists(os.path.join(tng_dir, 'TNG100-1', 'output')):
            os.makedirs(os.path.join(tng_dir, 'TNG100-1', 'output'))
        if not os.path.exists(os.path.join(tng_dir, 'TNG100-1', 'postprocessing')):
            os.makedirs(os.path.join(tng_dir, 'TNG100-1', 'postprocessing'))
        if not os.path.exists(os.path.join(tng_dir, 'TNG100-1', 'postprocessing', 'offsets')):
            os.makedirs(os.path.join(tng_dir, 'TNG100-1', 'postprocessing', 'offsets'))
        if not isfile(os.path.join(tng_dir, 'TNG100-1', 'simulation.hdf5')):
            self.terminal.add_log('Downloading simulation file')
            url = "http://www.tng-project.org/api/TNG100-1/files/simulation.hdf5"
            cmd = "wget -nv --content-disposition --header=API-Key:{} -O {} {}".format(self.tng_api_key_entry.text(), os.path.join(tng_dir, 'TNG100-1', 'simulation.hdf5'), url)
            subprocess.check_call(cmd, shell=True)
            self.terminal.add_log('Done.')
    
    def create_remote_environment(self):
        self.terminal.add_log('Checking ALMASim environment')
        repo_url = 'https://github.com/MicheleDelliVeneri/ALMASim.git'
        illustris_url = 'https://github.com/illustristng/illustris_python.git'
        if self.remote_dir_line.text() != '':
            work_dir = self.remote_dir_line.text()
            repo_dir = os.path.join(work_dir, 'ALMASim')
            venv_dir = os.path.join(work_dir, 'almasim_env')
            illustris_dir = os.path.join(work_dir, 'illustris_python')
        else:
            venv_dir = os.path.join('/home/{}'.format(self.remote_user_entry.text()), 'almasim_env')
            repo_dir = os.path.join('/home/{}'.format(self.remote_user_entry.text()), 'ALMASim')
            illustris_dir = os.path.join('/home/{}/'.format(self.remote_user_entry.text()), 'illustris_python')
        self.remote_main_dir = repo_dir
        self.remote_venv_dir = venv_dir
        if self.remote_key_pass_entry.text() != "":
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text(), password=self.remote_key_pass_entry.text())
        else:
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text())

        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text(), private_key_pass=self.remote_key_pass_entry.text())
                
        else:
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text())
        if not sftp.exists('/home/{}/.config'.format(self.remote_user_entry.text())):
            sftp.mkdir('/home/{}/.config'.format(self.remote_user_entry.text()))
        
        if not sftp.exists('/home/.config/{}/{}'.format(self.remote_user_entry.text(), self.settings_path.split(os.sep)[-1])):
            sftp.put(self.settings_path, '/home/{}/.config/{}'.format(self.remote_user_entry.text(), self.settings_path.split(os.sep)[-1]))
        commands = f"""
            chmod 600 /home/{self.remote_user_entry.text()}/.config/{self.settings_path.split(os.sep)[-1]}
            if [ ! -d {repo_dir} ]; then
                git clone {repo_url} {repo_dir}
            fi
            cd {repo_dir}
            git pull
            if [ ! -d {venv_dir} ]; then
                /usr/bin/python3.12 -m venv {venv_dir}
                source {venv_dir}/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
            fi
            if [ ! -d {illustris_dir} ]; then
                git clone {illustris_url} {illustris_dir}
                source {venv_dir}/bin/activate
                cd {illustris_dir}
                pip install .
            fi
            """
        
        paramiko_client = paramiko.SSHClient()
        paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        paramiko_client.connect(self.remote_address_entry.text(), username=self.remote_user_entry.text(), pkey=key)
        stdin, stdout, stderr = paramiko_client.exec_command(commands)
        self.terminal.add_log(stdout.read().decode())
        self.terminal.add_log(stderr.read().decode())

    def create_remote_output_dir(self):
        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text(), private_key_pass=self.remote_key_pass_entry.text())
                
        else:
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text())
        output_path = os.path.join(self.output_entry.text(), self.project_name_entry.text())
        plot_path = os.path.join(output_path, 'plots')
        if not sftp.exists(output_path):
            sftp.mkdir(output_path)
        if not sftp.exists(plot_path):
            sftp.mkdir(plot_path)
    
    def remote_check_tng_dirs(self):
        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text(), private_key_pass=self.remote_key_pass_entry.text())
                
        else:
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text())
        tng_dir = self.tng_entry.text()
        if not sftp.exists(os.path.join(tng_dir, 'TNG100-1')):
            sftp.mkdir(os.path.join(tng_dir, 'TNG100-1'))
        if not sftp.exists(os.path.join(tng_dir, 'TNG100-1', 'output')):
            sftp.mkdir(os.path.join(tng_dir, 'TNG100-1', 'output'))
        if not sftp.exists(os.path.join(tng_dir, 'TNG100-1', 'postprocessing')):
            sftp.mkdir(os.path.join(tng_dir, 'TNG100-1', 'postprocessing'))
        if not sftp.exists(os.path.join(tng_dir, 'TNG100-1', 'postprocessing', 'offsets')):
            sftp.mkdir(os.path.join(tng_dir, 'TNG100-1', 'postprocessing', 'offsets'))
        if not sftp.exists(os.path.join(tng_dir, 'TNG100-1', 'simulation.hdf5')):
            self.terminal.add_log('Downloading simulation file')
            url = "http://www.tng-project.org/api/TNG100-1/files/simulation.hdf5"
            cmd = "wget -nv --content-disposition --header=API-Key:{} -O {} {}".format(self.tng_api_key_entry.text(), os.path.join(tng_dir, 'TNG100-1', 'simulation.hdf5'), url)
            if self.remote_key_pass_entry.text() != "":
                key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text(), password=self.remote_key_pass_entry.text())
            else:
                key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text())
            paramiko_client = paramiko.SSHClient()
            paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            paramiko_client.connect(self.remote_address_entry.text(), username=self.remote_user_entry.text(), pkey=key)
            stdin, stdout, stderr = paramiko_client.exec_command(cmd)
            self.terminal.add_log(stdout.read().decode())
            self.terminal.add_log(stderr.read().decode())
            self.terminal.add_log('Done.')

    def copy_metadata_on_remote(self):
        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text(), private_key_pass=self.remote_key_pass_entry.text())
                
        else:
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text())
        
        self.input_params.to_csv('input_params.csv', index=False)
        sftp.put('input_params.csv', self.remote_main_dir + '/input_params.csv')
        os.remove('input_params.csv')

    def copy_settings_on_remote(self):
        if self.remote_key_pass_entry.text() != "":
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text(), private_key_pass=self.remote_key_pass_entry.text())
                
        else:
            sftp = pysftp.Connection(self.remote_address_entry.text(), username=self.remote_user_entry.text(), private_key=self.remote_key_entry.text())

        if not sftp.exists(self.remote_main_dir + '/settings.json'):
            sftp.put(self.settings_path, self.remote_main_dir + '/settings.plist')

    def run_on_slurm_cluster(self):
        slurm_config = self.remote_config_entry.text()
        if self.remote_key_pass_entry.text() != "":
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text(), password=self.remote_key_pass_entry.text())
        else:
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text())
            
        settings_path= os.path.join(self.remote_main_dir, 'settings.plist')
        dask_commands = f"""
        cd {self.remote_main_dir}
        source {self.remote_venv_dir}/bin/activate
        export QT_QPA_PLATFORM=offscreen
        python -c "import sys; import os; import ui; from PyQt6.QtWidgets import QApplication; app = QApplication(sys.argv); ui.ALMASimulatorUI.settings_file = '{settings_path}'; window=ui.ALMASimulatorUI(); window.create_slurm_cluster_and_run(); sys.exit(app.exec())"
        """
        paramiko_client = paramiko.SSHClient()
        paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        paramiko_client.connect(self.remote_address_entry.text(), username=self.remote_user_entry.text(), pkey=key)
        stdin, stdout, stderr = paramiko_client.exec_command(dask_commands)
        self.terminal.add_log(stdout.read().decode())
        self.terminal.add_log(stderr.read().decode())

    @classmethod
    def create_slurm_cluster_and_run(cls):
        input_params = pd.read_csv('input_params.csv')
        with open('slurm_config.json', 'r') as f:
            config = json.load(f)
        cluster = SLURMCluster(
            queue=config['queue'],
            account=config['account'],
            cores=config['cores'],
            memory=config['memory'],
            job_extra_directives=config['job_extra'],
            )
       
        client = Client(cluster)
        # Get information
        print("Dashboard Link: {}".format(client.dashboard_link))
        print("Workers: {}".format(len(client.scheduler_info()['workers'])))
        print("Total threads: {}".format(sum(w['nthreads'] for w in client.scheduler_info()['workers'].values())))
        print("Total memory: {}".format(sum(w['memory_limit'] for w in client.scheduler_info()['workers'].values())))
        cluster.scale(jobs=int(int(cls.ncpu_entry.text())//4))
        ddf = dd.from_pandas(input_params, npartitions=int(int(cls.ncpu_entry.text()) // 4))
        output_type = "object"
        results = ddf.map_partitions(lambda df: df.apply(lambda row: cls.simulator(*row), axis=1), meta=output_type).compute()
        client.close()
        cluster.close()

    def run_on_pbs_cluster(self):
        pbs_config = self.remote_config_entry.text()
        if self.remote_key_pass_entry.text() != "":
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text(), password=self.remote_key_pass_entry.text())
        else:
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text())
            
        
        dask_commands = f"""
        cd {self.remote_main_dir}
        source {self.remote_venv_dir}/bin/activate
        export QT_QPA_PLATFORM=offscreen
        python -c "import sys; import os; import ui; from PyQt6.QtWidgets import QApplication; app = QApplication(sys.argv); window=ui.ALMASimulatorUI(); window.create_pbs_cluster_and_run(); sys.exit(app.exec())"
        """
        paramiko_client = paramiko.SSHClient()
        paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        paramiko_client.connect(self.remote_address_entry.text(), username=self.remote_user_entry.text(), pkey=key)
        stdin, stdout, stderr = paramiko_client.exec_command(dask_commands)
        self.terminal.add_log(stdout.read().decode())
        self.terminal.add_log(stderr.read().decode())

    def run_on_mpi_machine(self):
        slurm_config = self.remote_config_entry.text()
        if self.remote_key_pass_entry.text() != "":
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text(), password=self.remote_key_pass_entry.text())
        else:
            key = paramiko.RSAKey.from_private_key_file(self.remote_key_entry.text())
            
        settings_path= os.path.join(self.remote_main_dir, 'settings.plist')
        dask_commands = f"""
        cd {self.remote_main_dir}
        source {self.remote_venv_dir}/bin/activate
        export QT_QPA_PLATFORM=offscreen
        python -c "import sys; import os; import ui; from PyQt6.QtWidgets import QApplication; app = QApplication(sys.argv); ui.ALMASimulatorUI.settings_file = '{settings_path}'; window=ui.ALMASimulatorUI(); window.create_local_cluster_and_run(); sys.exit(app.exec())"
        """
        paramiko_client = paramiko.SSHClient()
        paramiko_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        paramiko_client.connect(self.remote_address_entry.text(), username=self.remote_user_entry.text(), pkey=key)
        stdin, stdout, stderr = paramiko_client.exec_command(dask_commands)
        self.terminal.add_log(stdout.read().decode())
        self.terminal.add_log(stderr.read().decode())
        
    @classmethod
    def create_local_cluster_and_run(cls):
        input_params = pd.read_csv('input_params.csv')
        output_type = "object"
        cluster = LocalCluster(n_workers=int(int(cls.ncpu_entry.text()) // 4), threads_per_worker=4, dashboard_address=':8787')
        client = Client(cluster)
        ddf = dd.from_pandas(input_params, npartitions=int(int(cls.ncpu_entry.text()) // 4))
        results = ddf.map_partitions(lambda df: df.apply(lambda row: cls.simulator(*row), axis=1), meta=output_type).compute()
        client.close()
        cluster.close()
    
    def transform_source_type_label(self):
        if self.model_combo.currentText() == 'Galaxy Zoo':
           self.source_type = 'galaxy-zoo'
        elif self.model_combo.currentText() == 'Hubble 100':
            self.source_type = 'hubble-100'
        elif self.model_combo.currentText() == 'Molecular':
            self.source_type = 'molecular'
        elif self.model_combo.currentText() == 'Diffuse':
            self.source_type = 'diffuse'
        elif self.model_combo.currentText() == 'Gaussian':
            self.source_type = 'gaussian'
        elif self.model_combo.currentText() == 'Point':
            self.source_type = 'point'
        elif self.model_combo.currentText() == 'Extended':
            self.source_type = 'extended'

    def sample_given_redshift(self, metadata, n, rest_frequency, extended, zmax=None):
        pd.options.mode.chained_assignment = None
        if isinstance(rest_frequency, np.ndarray):
            rest_frequency = np.sort(np.array(rest_frequency))[0]
        self.terminal.add_log(f"Max frequency recorded in metadata: {np.max(metadata['Freq'].values)}")
        self.terminal.add_log(f"Min frequency recorded in metadata: {np.min(metadata['Freq'].values)}")
        self.terminal.add_log('Filtering metadata based on line catalogue...')
        metadata = metadata[metadata['Freq'] <= rest_frequency]
        self.terminal.add_log(f'Remaining metadata: {len(metadata)}')
        freqs = metadata['Freq'].values
        redshifts = [uas.compute_redshift(rest_frequency * U.GHz, source_freq * U.GHz) for source_freq in freqs]
        metadata.loc[:, 'redshift'] = redshifts

        n_metadata = 0
        z_save = zmax
        self.terminal.add_log('Computing redshifts')
        while n_metadata < ceil(n / 10):
            s_metadata = n_metadata
            if zmax != None:
                f_metadata = metadata[(metadata['redshift'] <= zmax) & (metadata['redshift'] >= 0)]
            else:
                f_metadata = metadata[metadata['redshift'] >= 0]
            n_metadata = len(f_metadata)
            if n_metadata == s_metadata:
                zmax += 0.1
        if zmax != None:
                metadata = metadata[(metadata['redshift'] <= zmax) & (metadata['redshift'] >= 0)]
        else:
            metadata = metadata[metadata['redshift'] >= 0]
        if z_save != zmax:
            self.terminal.add_log(f'Max redshift has been adjusted fit metadata, new max redshift: {round(zmax, 3)}')
        self.terminal.add_log(f'Remaining metadata: {len(metadata)}')
        snapshots = [uas.redshift_to_snapshot(redshift) for redshift in metadata['redshift'].values]
        metadata['snapshot'] = snapshots
        if extended == True:
            #metatada = metadata[metadata['redshift'] < 0.05]
            metadata = metadata[(metadata['snapshot'] == 99) | (metadata['snapshot'] == 95)]

        sample = metadata.sample(n, replace=True)
        return sample
    
    @staticmethod
    def remove_non_numeric(text):
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
        print(x)
        op = math.floor if bin(x)[3] != "1" else math.ceil
        return 2 ** op(math.log(x, 2))

    @staticmethod
    def freq_supp_extractor(freq_sup, obs_freq):
        freq_band, n_channels, freq_mins, freq_maxs, freq_ds = [], [], [], [], []
        freq_sup = freq_sup.split('U')
        for i in range(len(freq_sup)):
            sup = freq_sup[i][1:-1].split(',')
            sup = [su.split('..') for su in sup][:2]
            freq_min, freq_max = float(ALMASimulatorUI.remove_non_numeric(sup[0][0])), float(ALMASimulatorUI.remove_non_numeric(sup[0][1]))
            freq_d = float(ALMASimulatorUI.remove_non_numeric(sup[1][0]))
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
        freq_ranges = np.array([[freq_mins[i].value, freq_maxs[i].value] for i in range(len(freq_mins))])
        idx_ = np.argwhere((obs_freq.value >= freq_ranges[:, 0]) & (obs_freq.value <= freq_ranges[:, 1]))[0][0]
        freq_range = freq_ranges[idx_]
        band_range = freq_range[1] - freq_range[0]
        n_channels = n_channels[idx_]
        central_freq = freq_range[0] + band_range / 2
        freq_d = freq_ds[idx_]
        return band_range * U.GHz, central_freq * U.GHz, n_channels, freq_d

    # -------- Simulation Functions ------------------------
    def start_simulation(self):
        # Implement the logic to start the simulation
        if self.local_mode_combo.currentText() == 'local':
            self.terminal.add_log('Starting simulation on your local machine')
        else: 
            self.terminal.add_log(f'Starting simulation on {self.remote_address_entry.text()}')
        n_sims = int(self.n_sims_entry.text())
        n_cpu = int(self.ncpu_entry.text())
        sim_idxs = np.arange(n_sims)
        self.transform_source_type_label()
        source_types = np.array([self.source_type] * n_sims)
        output_path = os.path.join(self.output_entry.text(), self.project_name_entry.text())
        plot_path = os.path.join(output_path, 'plots')
        # Output Directory 
        if self.local_mode_combo.currentText() == 'local':
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
        else:
            self.create_remote_output_dir()

        output_paths = np.array([output_path] * n_sims)
        tng_paths = np.array([self.tng_entry.text()] * n_sims)

        # Galaxy Zoo Directory 
        if self.local_mode_combo.currentText() == 'local':
            if self.galaxy_zoo_entry.text() and not os.path.exists(os.path.join(self.galaxy_zoo_entry.text(), 'images_gz2')):
                self.download_galaxy_zoo()
        else:
            self.create_remote_environment()
            self.download_galaxy_zoo_on_remote()

        galaxy_zoo_paths = np.array([self.galaxy_zoo_entry.text()] * n_sims)
        
        if self.local_mode_combo.currentText() == 'local':
            main_paths = np.array([os.getcwd()] * n_sims)
        else: 
            main_paths = np.array([os.path.join('/home/{}/'.format(self.remote_user_entry.text()), 'ALMASim')] * n_sims)
        ncpus = np.array([int(self.ncpu_entry.text())] * n_sims)
        project_names = np.array([self.project_name_entry.text()] * n_sims)
        save_mode = np.array([self.save_format_combo.currentText()] * n_sims)

        # Checking Line Mode
        if self.line_mode_checkbox.isChecked():
            line_indices = [int(i) for i in self.line_index_entry.text().split()]
            rest_freq, line_names = uas.get_line_info(os.getcwd(), line_indices)
            if len(rest_freq) == 1:
                rest_freq = rest_freq[0]
            rest_freqs = np.array([rest_freq]*n_sims)
            redshifts = np.array([None]*n_sims)
            n_lines = np.array([None]*n_sims)
            line_names = np.array([line_names]*n_sims)
            z1 = None
        else:
            redshifts = [float(z) for z in self.redshift_entry.text().split()]
            if len(redshifts) == 1:
                redshifts = np.array([redshifts[0]] * n_sims)
                z0, z1 = float(redshifts[0]), float(redshifts[0])
            else:
                z0, z1 = float(redshifts[0]), float(redshifts[1])
                redshifts = np.random.uniform(z0, z1, n_sims)
            n_lines = np.array([int(self.num_lines_entry.text())] * n_sims)
            rest_freq, _ = uas.get_line_info(os.getcwd())
            rest_freqs = np.array([None]*n_sims)
            line_names = np.array([None]*n_sims)

        # Checking Infrared Luminosity
        if self.ir_luminosity_checkbox.isChecked():
            lum_infrared = [float(lum) for lum in self.
            
            
            ir_luminosity_entry.text().split()]
            if len(lum_infrared) == 1:
                lum_ir = np.array([lum_infrared[0]] * n_sims)
            else:
                lum_ir = np.random.uniform(lum_infrared[0], lum_infrared[1], n_sims)
        else:
            lum_ir = np.array([None]*n_sims)

        # Checking SNR
        if self.snr_checkbox.isChecked():
            snr = [float(snr) for snr in self.snr_entry.text().split()]
            if len(snr) == 1:
                snr = np.array([snr[0]]*n_sims)
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
        if self.model_combo.currentText() == 'Extended':
            if self.local_mode_combo.currentText() == 'local':
                self.check_tng_dirs()
            else:
                self.remote_check_tng_dirs()
            tng_apis = np.array([self.tng_api_key_entry.text()] * n_sims)
            self.metadata = self.sample_given_redshift(self.metadata, n_sims, rest_freq, True, z1)
        else:
            tng_apis = np.array([None] * n_sims)
            self.metadata = self.sample_given_redshift(self.metadata, n_sims, rest_freq, False, z1)
        source_names = self.metadata['ALMA_source_name'].values
        ras = self.metadata['RA'].values
        decs = self.metadata['Dec'].values
        bands = self.metadata['Band'].values
        ang_ress = self.metadata['Ang.res.'].values
        vel_ress = self.metadata['Vel.res.'].values
        fovs = self.metadata['FOV'].values
        obs_dates = self.metadata['Obs.date'].values
        pwvs = self.metadata['PWV'].values
        int_times = self.metadata['Int.Time'].values
        bandwidths = self.metadata['Bandwidth'].values
        freqs = self.metadata['Freq'].values
        freq_supports = self.metadata['Freq.sup.'].values
        antenna_arrays = self.metadata['antenna_arrays'].values
        cont_sens = self.metadata['Cont_sens_mJybeam'].values
        self.terminal.add_log('Metadata retrived successfully\n')
        if self.serendipitous_checkbox.isChecked():
            inject_serendipitous = np.array([True] * n_sims)
        else:
            inject_serendipitous = np.array([False] * n_sims)
        if self.local_mode_combo.currentText() == 'local':
            remote = np.array([False] * n_sims)
        else:
            remote = np.array([True] * n_sims)
        self.input_params = pd.DataFrame(zip(
        sim_idxs, source_names, main_paths, output_paths, tng_paths, galaxy_zoo_paths, project_names, ras, decs, bands, ang_ress, vel_ress, fovs, 
        obs_dates, pwvs, int_times, bandwidths, freqs, freq_supports, cont_sens,
        antenna_arrays, n_pixs, n_channels, source_types,
        tng_apis, ncpus, rest_freqs, redshifts, lum_ir, snr,
        n_lines, line_names, save_mode, inject_serendipitous, remote), 
        columns = ['idx', 'source_name', 'main_path', 'output_dir', 'tng_dir', 'galaxy_zoo_dir', 'project_name', 'ra', 'dec', 'band', 
        'ang_res', 'vel_res', 'fov', 'obs_date', 'pwv', 'int_time', 'bandwidth', 
        'freq', 'freq_support', 'cont_sens', 'antenna_array', 'n_pix', 'n_channels', 'source_type',
        'tng_api_key', 'ncpu', 'rest_frequency', 'redshift', 'lum_infrared', 'snr',
        'n_lines', 'line_names', 'save_mode', 'inject_serendipitous', 'remote'])
        if self.local_mode_combo.currentText() == 'remote':
            self.copy_metadata_on_remote()
            self.copy_settings_on_remote()
        if self.comp_mode_combo.currentText() == 'parallel':
            if self.local_mode_combo.currentText() == 'local':
                dask.config.set({'temporary_directory': output_path})
                total_memory = psutil.virtual_memory().total
                num_processes = int(self.ncpu_entry.text()) // 4
                memory_limit = int(0.9 * total_memory / num_processes)
                ddf = dd.from_pandas(self.input_params, npartitions=num_processes)
                cluster = LocalCluster(n_workers=num_processes, threads_per_worker=4, dashboard_address=':8787')
                output_type = "object"
                client = Client(cluster)
                client.register_worker_plugin(MemoryLimitPlugin(memory_limit))
                results =  ddf.map_partitions(lambda df: df.apply(lambda row: ALMASimulatorUI.simulator(*row), axis=1), meta=output_type).compute()
                client.close()
                cluster.close()
            #elif self.local_mode_combo.currentText() == 'remote':
            else:
                if self.remote_mode_combo.currentText() == 'SLURM':
                    self.run_on_slurm_cluster()
                elif self.remote_mode_combo.currentText() == 'PBS':
                    self.run_on_pbs_cluster()
                elif self.remote_mode_combo.currentText() == 'MPI':
                    self.run_on_mpi_machine()   
                else:
                    self.terminal.add_log('Please select a valid remote mode')
        else:
            if self.local_mode_combo.currentText() == 'local':
                for i in range(n_sims):
                    ALMASimulatorUI.simulator(*self.input_params.iloc[i])
            else:
                self.terminal.add_log('Cannot run on remote in sequential mode, changing it to parallel')
                self.comp_mode_combo.setCurrentText('parallel')

    @staticmethod
    def simulator(inx, source_name, main_dir, output_dir, tng_dir, galaxy_zoo_dir, project_name, ra, dec, band, ang_res, vel_res, fov, obs_date, 
                pwv, int_time,  bandwidth, freq, freq_support, cont_sens, antenna_array, n_pix, 
                n_channels, source_type, tng_api_key, ncpu, rest_frequency, redshift, lum_infrared, snr,
                n_lines, line_names, save_mode, inject_serendipitous=False, remote=False):
        """
        Simulates the ALMA observations for the given input parameters.

        Parameters:
        idx (int): Index of the simulation.
        main_path (str): Path to the directory where the file.csv is stored.
        output_dir (str): Path to the output directory.
        tng_dir (str): Path to the TNG directory.
        galaxy_zoo_dir (str): Path to the Galaxy Zoo directory.
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
        print(inx)
        print(source_name)
        print(main_dir)
        print(output_dir)
        print(tng_dir)
        print(galaxy_zoo_dir)
        print(project_name)
        print(ra)
        print(dec)
        print(band)
        print(ang_res)
        print(vel_res)
        print(fov)
        print(obs_date)
        print(pwv)
        print(int_time)
        print(bandwidth)
        print(freq_support)
        print(cont_sens)
        print(antenna_array)
        print(n_pix)
        print(n_channels)
        print(source_type)
        print(tng_api_key)
        print(ncpu)
        print(rest_frequency)
        print(redshift)
        print(lum_infrared)
        print(snr)
        print(n_lines)
        print(line_names)
        print(save_mode)
        print(inject_serendipitous)
        print(remote)
        print("\n\n")
        if remote == True:
            print('\nRunning simulation {}'.format(inx))
            print('Source Name: {}'.format(source_name))
        else:
            ALMASimulatorUI.terminal.add_log('\nRunning simulation {}'.format(inx))
            ALMASimulatorUI.terminal.add_log('Source Name: {}'.format(source_name))
        start = time.time()
        second2hour = 1 / 3600
        ra = ra * U.deg
        dec = dec * U.deg
        fov = fov * 3600 * U.arcsec
        ang_res = ang_res * U.arcsec
        vel_res = vel_res * U.km / U.s
        int_time = int_time * U.s
        source_freq = freq * U.GHz
        band_range, central_freq, t_channels, delta_freq = ALMASimulatorUI.freq_supp_extractor(freq_support, source_freq)
        sim_output_dir = os.path.join(output_dir, project_name + '_{}'.format(inx))
        if not os.path.exists(sim_output_dir):
            os.makedirs(sim_output_dir)
        os.chdir(output_dir)
        if remote == True:
            print('RA: {}'.format(ra))
            print('DEC: {}'.format(dec))
            print('Integration Time: {}'.format(int_time))
        else:
            ALMASimulatorUI.terminal.add_log('RA: {}'.format(ra))
            ALMASimulatorUI.terminal.add_log('DEC: {}'.format(dec))
            ALMASimulatorUI.terminal.add_log('Integration Time: {}'.format(int_time))
        ual.generate_antenna_config_file_from_antenna_array(antenna_array, main_dir, sim_output_dir)
        antennalist = os.path.join(sim_output_dir, "antenna.cfg")
        antenna_name = 'antenna'
        max_baseline = ual.get_max_baseline_from_antenna_config(antennalist) * U.km
        if remote == True:
            print('Field of view: {} arcsec'.format(round(fov.value, 3)))
        else:
            ALMASimulatorUI.terminal.add_log('Field of view: {} arcsec'.format(round(fov.value, 3)) )
        beam_size = ual.estimate_alma_beam_size(central_freq, max_baseline, return_value=False)
        beam_solid_angle = np.pi * (beam_size / 2) ** 2
        cont_sens = cont_sens * U.mJy / (U.arcsec ** 2)
        cont_sens_jy = (cont_sens * beam_solid_angle).to(U.Jy)
        cont_sens  = cont_sens_jy  * snr
        if remote == True:
            print('Minimum detectable continum: {}'.format(cont_sens_jy))
        else:
            ALMASimulatorUI.terminal.add_log("Minimum detectable continum: {}".format(cont_sens_jy))
        cell_size = beam_size / 5
        if n_pix is None: 
            #cell_size = beam_size / 5
            print(n_pix)
            n_pix = ALMASimulatorUI.closest_power_of_2(int(1.5 * fov / cell_size))
        else:
            n_pix = ALMASimulatorUI.closest_power_of_2(n_pix)
            cell_size = fov / n_pix
            # just added
            #beam_size = cell_size * 5
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
            rest_frequency = uas.compute_rest_frequency_from_redshift(main_dir, source_freq.value, redshift) * U.GHz
        continum, line_fluxes, line_names, redshift, line_frequency, source_channel_index, n_channels_nw, bandwidth, freq_sup_nw, cont_frequencies, fwhm_z, lum_infrared  = uas.process_spectral_data(
                                                                            source_type,
                                                                            main_dir,
                                                                            redshift, 
                                                                            central_freq.value,
                                                                            band_range.value,
                                                                            source_freq.value,
                                                                            n_channels,
                                                                            lum_infrared,
                                                                            cont_sens.value,
                                                                            line_names,
                                                                            n_lines,
                                                                            )
        print('OK process spectral data')
        if n_channels_nw != n_channels:
            freq_sup = freq_sup_nw * U.MHz
            n_channels = n_channels_nw
            band_range  = n_channels * freq_sup
        central_channel_index = n_channels // 2
        if remote == True:
            print('Beam size: {} arcsec'.format(round(beam_size.value, 4)))
            print('Central Frequency: {}'.format(central_freq))
            print('Spectral Window: {}'.format(band_range))
            print('Freq Support: {}'.format(delta_freq))
            print('Cube Dimensions: {} x {} x {}'.format(n_pix, n_pix, n_channels))
            print('Redshift: {}'.format(round(redshift, 3)))
            print('Source frequency: {} GHz'.format(round(source_freq.value, 2)))
            print('Band: {}'.format(band))
            print('Velocity resolution: {} Km/s'.format(round(vel_res.value, 2)))
            print('Angular resolution: {} arcsec'.format(round(ang_res.value, 3)))
            print('Infrared Luminosity: {:.2e}'.format(lum_infrared))
        else:
            ALMASimulatorUI.terminal.add_log('Central Frequency: {}'.format(central_freq))
            ALMASimulatorUI.terminal.add_log('Beam size: {} arcsec'.format(round(beam_size.value, 4)))
            ALMASimulatorUI.terminal.add_log('Spectral Window: {}'.format(band_range))
            ALMASimulatorUI.terminal.add_log('Freq Support: {}'.format(delta_freq))
            ALMASimulatorUI.terminal.add_log('Cube Dimensions: {} x {} x {}'.format(n_pix, n_pix, n_channels))
            ALMASimulatorUI.terminal.add_log('Redshift: {}'.format(round(redshift, 3)))
            ALMASimulatorUI.terminal.add_log('Source frequency: {} GHz'.format(round(source_freq.value, 2)))
            ALMASimulatorUI.terminal.add_log('Band: {}'.format(band))
            ALMASimulatorUI.terminal.add_log('Velocity resolution: {} Km/s'.format(round(vel_res.value, 2)))
            ALMASimulatorUI.terminal.add_log('Angular resolution: {} arcsec'.format(round(ang_res.value, 3)))
            ALMASimulatorUI.terminal.add_log('Infrared Luminosity: {:.2e}'.format(lum_infrared))
        if source_type == 'extended':
            snapshot = uas.redshift_to_snapshot(redshift)
            tng_subhaloid = uas.get_subhaloids_from_db(1, main_dir, snapshot)
            outpath = os.path.join(tng_dir, 'TNG100-1', 'output', 'snapdir_0{}'.format(snapshot))
            part_num = uas.get_particles_num(tng_dir, outpath, snapshot, int(tng_subhaloid), tng_api_key)
            if remote == True:
                print('Snapshot: {}'.format(snapshot))
                print('Subhaloid ID: {}'.format(tng_subhaloid))
                print('Number of particles: {}'.format(part_num))
            else:
                ALMASimulatorUI.terminal.add_log('Snapshot: {}'.format(snapshot))
                ALMASimulatorUI.terminal.add_log('Subhaloid ID: {}'.format(tng_subhaloid))
                ALMASimulatorUI.terminal.add_log('Number of particles: {}'.format(part_num))
            while part_num == 0:
                if remote == True:
                    print('No particles found. Checking another subhalo.')
                else:
                    ALMASimulatorUI.terminal.add_log('No particles found. Checking another subhalo.')
                tng_subhaloid = uas.get_subhaloids_from_db(1, main_dir, snapshot)
                outpath = os.path.join(tng_dir, 'TNG100-1', 'output', 'snapdir_0{}'.format(snapshot))
                part_num = uas.get_particles_num(tng_dir, outpath, snapshot, int(tng_subhaloid), tng_api_key)
                if remote == True:
                    print('Number of particles: {}'.format(part_num))
                else:
                    ALMASimulatorUI.terminal.add_log('Number of particles: {}'.format(part_num))
        else:
            snapshot = None
            tng_subhaloid = None
        if type(line_names) == list or isinstance(line_names, np.ndarray):
            for line_name, line_flux in zip(line_names, line_fluxes): 
                if remote == True:
                    print('Simulating Line {} Flux: {:.3e} at z {}'.format(line_name, line_flux, redshift))
                else:     
                    ALMASimulatorUI.terminal.add_log('Simulating Line {} Flux: {:.3e} at z {}'.format(line_name, line_flux, redshift))
        else:
            if remote == True: 
                print('Simulating Line {} Flux: {} at z {}'.format(line_names[0], line_fluxes[0], redshift))
            else:
                ALMASimulatorUI.terminal.add_log('Simulating Line {} Flux: {} at z {}'.format(line_names[0], line_fluxes[0], redshift))
        if remote == True: 
            print('Simulating Continum Flux: {:.2e}'.format(np.mean(continum)))
            print('Continuum Sensitity: {:.2e}'.format(cont_sens))
            print('Generating skymodel cube ...')
        else:
            ALMASimulatorUI.terminal.add_log('Simulating Continum Flux: {:.2e}'.format(np.mean(continum)))
            ALMASimulatorUI.terminal.add_log('Continuum Sensitity: {:.2e}'.format(cont_sens))
            ALMASimulatorUI.terminal.add_log('Generating skymodel cube ...')
        datacube = usm.DataCube(
            n_px_x=n_pix, 
            n_px_y=n_pix,
            n_channels=n_channels, 
            px_size=cell_size, 
            channel_width=delta_freq, 
            velocity_centre=central_freq, 
            ra=ra, 
            dec=dec)
        wcs = datacube.wcs
        fwhm_x, fwhm_y, angle = None, None, None
        if source_type == 'point':
            pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
            pos_z = [int(index) for index in source_channel_index]
            datacube = usm.insert_pointlike(datacube, continum, line_fluxes, int(pos_x), int(pos_y), pos_z, fwhm_z, n_channels)
        elif source_type == 'gaussian':
            pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
            pos_z = [int(index) for index in source_channel_index]
            fwhm_x = np.random.randint(3, 10) 
            fwhm_y = np.random.randint(3, 10)    
            angle = np.random.randint(0, 180)
            datacube = usm.insert_gaussian(datacube, continum, line_fluxes, int(pos_x), int(pos_y), pos_z, fwhm_x, fwhm_y, fwhm_z,
                                             angle, n_pix, n_channels)
        elif source_type == 'extended':
            datacube = usm.insert_extended(datacube, tng_dir, snapshot, int(tng_subhaloid), redshift, ra, dec, tng_api_key, ncpu)
        elif source_type == 'diffuse':
            ALMASimulatorUI.terminal.add_log('To be implemented')
        elif source_type == 'galaxy-zoo':
            galaxy_path = os.path.join(galaxy_zoo_dir, 'images_gz2',  'images')
            pos_z = [int(index) for index in source_channel_index]
            datacube = usm.insert_galaxy_zoo(datacube, continum, line_fluxes, pos_z, fwhm_z, n_pix, n_channels, galaxy_path)

        uas.write_sim_parameters(os.path.join(output_dir, 'sim_params_{}.txt'.format(inx)),
                                ra, dec, ang_res, vel_res, int_time, band, band_range, central_freq,
                                redshift, line_fluxes, line_names, line_frequency, 
                                continum, fov, beam_size, cell_size, n_pix, 
                                n_channels, snapshot, tng_subhaloid, lum_infrared, fwhm_z, source_type, fwhm_x, fwhm_y, angle)

        if inject_serendipitous == True:
            if source_type != 'gaussian':
                fwhm_x = np.random.randint(3, 10)
                fwhm_y = np.random.randint(3, 10)
            datacube = usm.insert_serendipitous(datacube, continum, cont_sens.value, line_fluxes, line_names, line_frequency, 
                                                delta_freq.value, pos_z, fwhm_x, fwhm_y, fwhm_z, n_pix, n_channels, 
                                                os.path.join(output_dir, 'sim_params_{}.txt'.format(inx)))
        #filename = os.path.join(sim_output_dir, 'skymodel_{}.fits'.format(inx))
        #self.terminal.add_log('\nWriting datacube to {}'.format(filename))
        #usm.write_datacube_to_fits(datacube, filename, obs_date)
        header = usm.get_datacube_header(datacube, obs_date)
        model = datacube._array.to_value(datacube._array.unit).T
        totflux = np.sum(model) 
        if remote == True: 
            print('Total Flux injected in model cube: {:.3f} Jy'.format(totflux))
            print('Done')
        else:
            ALMASimulatorUI.terminal.add_log(f'Total Flux injected in model cube: {round(totflux, 3)} Jy')
            ALMASimulatorUI.terminal.add_log('Done')
        del datacube
        if remote == True: 
            print('Observing with ALMA')
        else:
            ALMASimulatorUI.terminal.add_log('Observing with ALMA')
        uin.Interferometer(inx, model, main_dir, output_dir, ra, dec, central_freq, band_range, fov, antenna_array, cont_sens.value, 
                            int_time.value * second2hour, obs_date, header, save_mode)
        if remote == True:
            print('Finished')
        else: 
            ALMASimulatorUI.terminal.add_log('Finished')
        stop = time.time()
        if remote == True:
            print('Execution took {} seconds'.format(strftime("%H:%M:%S", gmtime(stop - start))))
        else:
            ALMASimulatorUI.terminal.add_log('Execution took {} seconds'.format(strftime("%H:%M:%S", gmtime(stop - start))))
        shutil.rmtree(sim_output_dir)
       


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ALMASimulatorUI()
    window.show()
    sys.exit(app.exec())