import pytest
from pytestqt.qtbot import QtBot
from PyQt6.QtWidgets import QApplication
import almasim.ui as ui
import sys
import faulthandler

faulthandler.enable()


@pytest.fixture(scope="session")
def app():
    """Fixture for setting up the QApplication instance."""
    app = QApplication(sys.argv)
    yield app
    app.quit()


@pytest.fixture
def main_window(qtbot: QtBot):
    """Fixture for setting up the main window."""
    window = ui.ALMASimulator()
    qtbot.addWidget(window)
    # window.show()
    yield window
    # window.close()


def test_ALMASimulator_creation(main_window):
    """Test the creation of the ALMASimulator window."""
    # Check if the main window is visible
    assert main_window


if __name__ == "__main__":
    pytest.main(["-v", __file__])
