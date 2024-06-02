import sys
import numpy as np
import pandas as pd
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QScrollArea, QGridLayout, 
    QGroupBox, QCheckBox, QRadioButton, QButtonGroup, QSizePolicy,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QMessageBox, QPlainTextEdit  
)
from PyQt6.QtCore import QSettings, QIODevice, QTextStream, QProcess, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
from kaggle import api
import utility.alma as ual

class EmittingStream(QIODevice):
    def __init__(self, parent=None):
        super().__init__(parent)

    def writeData(self, data):
        if isinstance(data, str):  # Check if data is a string
            data = data.encode('utf-8')  # Encode string into bytes if needed

        self.parent().appendPlainText(data.decode("utf-8"))  # Decode and write to QPlainTextEdit
        return len(data)  # Indicate successful write

    def flush(self):  # Optional: Implement flush if needed for buffering
        pass

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
                ual.plot_science_keywords_distributions(os.getcwd())  # Generate plots if not found

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
    def __init__(self):
        super().__init__()
        self.settings = QSettings("INFN Section of Naples", "ALMASim")
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

        self.galaxy_zoo_label = QLabel("Galaxy Zoo Directory:")
        self.galaxy_zoo_entry = QLineEdit()
        self.galaxy_zoo_button = QPushButton("Browse")
        self.galaxy_zoo_button.clicked.connect(self.browse_galaxy_zoo_directory)

        self.project_name_label = QLabel("Project Name:")
        self.project_name_entry = QLineEdit()

        self.n_sims_label = QLabel("Number of Simulations:")
        self.n_sims_entry = QLineEdit()

        self.ncpu_label = QLabel("Total Number of CPUs:")
        self.ncpu_entry = QLineEdit()

        self.comp_mode_label = QLabel("Computation Mode:")
        self.comp_mode_combo = QComboBox()
        self.comp_mode_combo.addItems(["sequential", "parallel"])

        self.metadata_mode_label = QLabel("Metadata Retrieval Mode:")
        self.metadata_mode_combo = QComboBox()
        self.metadata_mode_combo.addItems(["query", "get"])
        self.metadata_mode_combo.currentTextChanged.connect(self.toggle_metadata_browse)

        self.metadata_path_label = QLabel("Metadata Path:")
        self.metadata_path_entry = QLineEdit()
        self.metadata_path_button = QPushButton("Browse")
        self.metadata_path_button.clicked.connect(self.browse_metadata_path)
        #self.metadata_path_entry.editingFinished.connect(self.metadata_path_set)
        self.metadata_path_row = QHBoxLayout()
        self.metadata_path_row.addWidget(self.metadata_path_label)
        self.metadata_path_row.addWidget(self.metadata_path_entry)
        self.metadata_path_row.addWidget(self.metadata_path_button)

        self.start_button = QPushButton("Start Simulation")
        # self.start_button.clicked.connect(self.start_simulation)

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

        # Output Directory Row
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_label)
        output_row.addWidget(self.output_entry)
        output_row.addWidget(self.output_button)
        self.left_layout.addLayout(output_row)

        # TNG Directory Row
        tng_row = QHBoxLayout()
        tng_row.addWidget(self.tng_label)
        tng_row.addWidget(self.tng_entry)
        tng_row.addWidget(self.tng_button)
        self.left_layout.addLayout(tng_row)

        # Galaxy Zoo Directory Row
        galaxy_row = QHBoxLayout()
        galaxy_row.addWidget(self.galaxy_zoo_label)
        galaxy_row.addWidget(self.galaxy_zoo_entry)
        galaxy_row.addWidget(self.galaxy_zoo_button)
        self.left_layout.addLayout(galaxy_row)

        # Project Name Row
        project_name_row = QHBoxLayout()
        project_name_row.addWidget(self.project_name_label)
        project_name_row.addWidget(self.project_name_entry)
        self.left_layout.addLayout(project_name_row)

        # Number of Simulations Row
        n_sims_row = QHBoxLayout()
        n_sims_row.addWidget(self.n_sims_label)
        n_sims_row.addWidget(self.n_sims_entry)
        self.left_layout.addLayout(n_sims_row)

        # Number of CPUs Row
        ncpu_row = QHBoxLayout()
        ncpu_row.addWidget(self.ncpu_label)
        ncpu_row.addWidget(self.ncpu_entry)
        self.left_layout.addLayout(ncpu_row)

        # Computation Mode Row
        comp_mode_row = QHBoxLayout()
        comp_mode_row.addWidget(self.comp_mode_label)
        comp_mode_row.addWidget(self.comp_mode_combo)
        self.left_layout.addLayout(comp_mode_row)

        # Metadata Retrieval Mode Row
        metadata_mode_row = QHBoxLayout()
        metadata_mode_row.addWidget(self.metadata_mode_label)
        metadata_mode_row.addWidget(self.metadata_mode_combo)
        self.left_layout.addLayout(metadata_mode_row)

        # Button Row
        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.start_button)
        self.left_layout.addLayout(button_row)

        right_layout.addWidget(self.terminal)

        main_layout.addLayout(self.left_layout)
        main_layout.addLayout(right_layout)

        # Load saved settings
        self.load_settings()

        # Redirect stdout and stderr
        #sys.stdout = EmittingStream(self.terminal)
        #sys.stderr = EmittingStream(self.terminal)
        self.terminal.start_log("")
        # Check metadata mode on initialization
        self.metadata_mode_combo.currentTextChanged.connect(self.toggle_metadata_browse)
        #self.load_metadata(self.metadata_path_entry.text())
        current_mode = self.metadata_mode_combo.currentText()
        if current_mode == 'query':
            self.add_query_widgets()
        self.toggle_metadata_browse(current_mode)  # Call here


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

    def browse_metadata_path(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Metadata File", os.path.join(os.getcwd(), 'metadata'), "CSV Files (*.csv)")
        if file:
            self.metadata_path_entry.setText(file)
            self.metadata_path_set()

    def select_metadata_path(self):
        file, _ = QFileDialog.getSaveFileName(self, "Select Metadata File", os.path.join(os.getcwd(), 'metadata'), "CSV Files (*.csv)")
        if file:
            self.query_save_entry.setText(file)
            #self.metadata_path_set()
            
    def metadata_path_set(self):
        metadata_path = self.metadata_path_entry.text()
        self.load_metadata(metadata_path)  # Pass only the metadata_path 

    def reset_fields(self):
        self.output_entry.clear()
        self.tng_entry.clear()
        self.galaxy_zoo_entry.clear()
        self.ncpu_entry.clear()
        self.n_sims_entry.clear()
        self.metadata_path_entry.clear()
        self.metadata_mode_combo.setCurrentText("get")
        self.comp_mode_combo.setCurrentText("sequential")
        self.query_save_entry.clear()

    def load_settings(self):
        self.output_entry.setText(self.settings.value("output_directory", ""))
        self.tng_entry.setText(self.settings.value("tng_directory", ""))
        self.galaxy_zoo_entry.setText(self.settings.value("galaxy_zoo_directory", ""))
        self.n_sims_entry.setText(self.settings.value("n_sims", ""))
        self.ncpu_entry.setText(self.settings.value("ncpu", ""))
        self.metadata_mode_combo.setCurrentText(self.settings.value("metadata_mode", "get"))
        self.comp_mode_combo.setCurrentText(self.settings.value("comp_mode", "sequential"))
        self.metadata_path_entry.setText(self.settings.value("metadata_path", ""))
        if self.metadata_mode_combo.currentText() == "get" and self.metadata_path_entry.text():
            self.load_metadata(self.metadata_path_entry.text())
        elif self.metadata_mode_combo.currentText() == "query" and self.settings.value("query_save_entry", ""):
            self.query_save_entry.setText(self.settings.value("query_save_entry", ""))
        if self.galaxy_zoo_entry.text() and not os.listdir(self.galaxy_zoo_entry.text()):
            self.download_galaxy_zoo()
        
    
    def closeEvent(self, event):
        self.settings.setValue("output_directory", self.output_entry.text())
        self.settings.setValue("tng_directory", self.tng_entry.text())
        self.settings.setValue("galaxy_zoo_directory", self.galaxy_zoo_entry.text())
        self.settings.setValue('n_sims', self.n_sims_entry.text())
        self.settings.setValue("ncpu", self.ncpu_entry.text())
        self.settings.setValue("metadata_path", self.metadata_path_entry.text())
        super().closeEvent(event)

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

    def add_metadata_widgets(self):
        self.metadata_path_label = QLabel("Metadata Path:")
        self.metadata_path_entry = QLineEdit()
        self.metadata_path_button = QPushButton("Browse")
        self.metadata_path_button.clicked.connect(self.browse_metadata_path)
        self.metadata_path_row = QHBoxLayout()
        self.metadata_path_row.addWidget(self.metadata_path_label)
        self.metadata_path_row.addWidget(self.metadata_path_entry)
        self.metadata_path_row.addWidget(self.metadata_path_button)
        self.left_layout.insertLayout(8, self.metadata_path_row)
        self.left_layout.update() 

    def update_query_save_label(self, query_type):
        if query_type == "science":
            self.query_save_label.setText("Save Query To:")
        else:  # query_type == "target"
            self.query_save_label.setText("Load Target List:")

    def execute_query(self):
        if self.metadata_mode_combo.currentText() == "query":
            query_type = self.query_type_combo.currentText()
            if query_type == "science":
                self.query_for_metadata_by_science_type()
            elif query_type == "target":
                self.query_metadata_from_target_list()
            else:
                # Handle invalid query type (optional)
                pass  
    
    def query_for_metadata_by_science_type(self):
        # Implement the logic to query metadata based on science type
        self.terminal.add_log("Querying metadata by science type...")
        self.plot_window = PlotWindow()
        self.plot_window.show()
        self.science_keywords, self.scientific_categories = ual.get_science_types()
        self.terminal.add_log('Available science keywords:')
        for i, keyword in enumerate(self.science_keywords):
            self.terminal.add_log(f'{i}: {keyword}')
        self.terminal.add_log('\nAvailable scientific categories:')
        for i, category in enumerate(self.scientific_categories):
            self.terminal.add_log(f'{i}: {category}')

        # Check if widgets already exist, remove them if so
        if hasattr(self, 'metadata_query_widgets_added') and self.metadata_query_widgets_added:
            self.remove_metadata_query_widgets()
        self.add_metadata_query_widgets()
        self.metadata_query_widgets_added = True
       

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
        
        execute_query_button = QPushButton("Continue Query")
        execute_query_button.clicked.connect(self.execute_query)

        # Create layouts and add widgets
        science_keyword_row = QHBoxLayout()
        science_keyword_row.addWidget(science_keyword_label)
        science_keyword_row.addWidget(self.science_keyword_entry)

        scientific_category_row = QHBoxLayout()
        scientific_category_row.addWidget(scientific_category_label)
        scientific_category_row.addWidget(self.scientific_category_entry)

        band_row = QHBoxLayout()
        band_row.addWidget(band_label)
        band_row.addWidget(self.band_entry)

        fov_row = QHBoxLayout()
        fov_row.addWidget(fov_label)
        fov_row.addWidget(self.fov_entry)

        time_resolution_row  = QHBoxLayout()
        time_resolution_row.addWidget(time_resolution_label)
        time_resolution_row.addWidget(self.time_resolution_entry)

        frequency_row = QHBoxLayout()
        frequency_row.addWidget(frequency_label)
        frequency_row.addWidget(self.frequency_entry)

        execute_query_row = QHBoxLayout()
        execute_query_row.addWidget(execute_query_button)

        # Insert rows into left_layout (adjust index if needed)
        self.left_layout.insertLayout(11, science_keyword_row)
        self.left_layout.insertLayout(12, scientific_category_row)
        self.left_layout.insertLayout(13, band_row)
        self.left_layout.insertLayout(14, fov_row)
        self.left_layout.insertLayout(15, time_resolution_row)
        self.left_layout.insertLayout(16, frequency_row)
        self.left_layout.insertWidget(17, execute_query_button)
        

    def remove_metadata_query_widgets(self):
        # Similar to remove_query_widgets from the previous response, but remove
        # all the rows and widgets added in add_metadata_query_widgets.
        widgets_to_remove = [
            self.science_keyword_row, self.scientific_category_row, self.band_row,
            self.fov_row, self.time_resolution_row, self.total_time_row, self.frequency_row,
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
    
    def query_metadata_from_target_list(self):
        # Implement the logic to query metadata based on a target list
        self.terminal.add_log("Querying metadata from target list...")

    def add_query_widgets(self):
        # Create widgets for querying
        self.query_type_label = QLabel("Query Type:")
        self.query_type_combo = QComboBox()
        self.query_type_combo.addItems(["science", "target"])
        self.query_type_row = QHBoxLayout()
        self.query_type_row.addWidget(self.query_type_label)
        self.query_type_row.addWidget(self.query_type_combo)
        self.query_save_label = QLabel("Save Metadata to:")
        self.query_type_combo.currentTextChanged.connect(self.update_query_save_label)
        # Set the initial label text
        self.update_query_save_label(self.query_type_combo.currentText())
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
        # Insert layouts at the correct positions
        self.left_layout.insertLayout(8, self.query_type_row)
        self.left_layout.insertLayout(9, self.query_save_row)
        self.left_layout.insertWidget(10, self.query_execute_button)

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

        self.metadata_query_widgets_added = False
        
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
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ALMASimulatorUI()
    window.show()
    sys.exit(app.exec())