import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QMessageBox
)

class ALMASimulatorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initialize_ui()
        
    def initialize_ui(self):
        self.setWindowTitle("ALMASim: set up your simulation parameters")

        # --- Create Widgets ---
        self.output_label = QLabel("Output Directory:")
        self.output_entry = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.browse_output_directory)

        self.tng_label = QLabel("TNG Directory:")
        self.tng_entry = QLineEdit()
        self.tng_button = QPushButton("Browse")
        self.tng_button.clicked.connect(self.browse_tng_directory)

        # --- Galaxy Zoo Directory ---
        self.galaxy_zoo_label = QLabel("Galaxy Zoo Directory:")
        self.galaxy_zoo_entry = QLineEdit()
        self.galaxy_zoo_button = QPushButton("Browse")
        self.galaxy_zoo_button.clicked.connect(self.browse_galaxy_zoo_directory)

        # --- Project Name ---
        self.project_name_label = QLabel("Project Name:")
        self.project_name_entry = QLineEdit()

        # --- Number of Simulations ---
        self.n_sims_label = QLabel("Number of Simulations:")
        self.n_sims_entry = QLineEdit()

        # --- Number of CPUs ---
        self.ncpu_label = QLabel("Total Number of CPUs:")
        self.ncpu_entry = QLineEdit()

        # --- Computation Mode ---
        self.comp_mode_label = QLabel("Computation Mode:")
        self.comp_mode_combo = QComboBox()
        self.comp_mode_combo.addItems(["sequential", "parallel"])

        # --- Metadata Retrieval Mode ---
        self.metadata_mode_label = QLabel("Metadata Retrieval Mode:")
        self.metadata_mode_combo = QComboBox()
        self.metadata_mode_combo.addItems(["query", "get"])
        
        # ... (add more labels, line edits, and buttons for other input parameters)

        self.start_button = QPushButton("Start Simulation")
        #self.start_button.clicked.connect(self.start_simulation)

        # --- Layout ---
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Output Directory Row
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_label)
        output_row.addWidget(self.output_entry)
        output_row.addWidget(self.output_button)
        layout.addLayout(output_row)

        # TNG Directory Row
        tng_row = QHBoxLayout()
        tng_row.addWidget(self.tng_label)
        tng_row.addWidget(self.tng_entry)
        tng_row.addWidget(self.tng_button)
        layout.addLayout(tng_row)

        # Galaxy Zoo Directory Row
        galaxy_row = QHBoxLayout()
        galaxy_row.addWidget(self.galaxy_zoo_label)
        galaxy_row.addWidget(self.galaxy_zoo_entry)
        galaxy_row.addWidget(self.galaxy_zoo_button)
        layout.addLayout(galaxy_row)

        # Project Name Row
        project_name_row = QHBoxLayout()
        project_name_row.addWidget(self.project_name_label)
        project_name_row.addWidget(self.project_name_entry)
        layout.addLayout(project_name_row)

        # Number of Simulations Row
        n_sims_row = QHBoxLayout()
        n_sims_row.addWidget(self.n_sims_label)
        n_sims_row.addWidget(self.n_sims_entry)
        layout.addLayout(n_sims_row)

        # Number of CPUs Row
        ncpu_row = QHBoxLayout()
        ncpu_row.addWidget(self.ncpu_label)
        ncpu_row.addWidget(self.ncpu_entry)
        layout.addLayout(ncpu_row)

        # Computation Mode Row
        comp_mode_row = QHBoxLayout()
        comp_mode_row.addWidget(self.comp_mode_label)
        comp_mode_row.addWidget(self.comp_mode_combo)
        layout.addLayout(comp_mode_row)

        # Metadata Retrieval Mode Row
        metadata_mode_row = QHBoxLayout()
        metadata_mode_row.addWidget(self.metadata_mode_label)
        metadata_mode_row.addWidget(self.metadata_mode_combo)
        layout.addLayout(metadata_mode_row)


        # ... (add more rows for other parameters)

        layout.addWidget(self.start_button)
        self.setCentralWidget(central_widget)

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_entry.setText(directory)

    def browse_tng_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select TNG Directory")
        if directory:
            self.tng_entry.setText(directory)

    def browse_galaxy_zoo_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Galaxy Zoo Directory")
        if directory:
            self.galaxy_zoo_entry.setText(directory)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ALMASimulatorUI()
    window.show()
    sys.exit(app.exec())