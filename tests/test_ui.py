import pytest
from pytestqt.qtbot import QtBot
import almasim.ui as ui
import faulthandler
import os

faulthandler.enable()
os.environ["LC_ALL"] = "C"


@pytest.fixture
def main_window(qtbot: QtBot):
    """Fixture for setting up the main window."""
    window = ui.ALMASimulator()
    qtbot.addWidget(window)
    window.show()
    yield window
    # window.close()


def test_ALMASimulator_creation(main_window):
    """Test the creation of the ALMASimulator window."""
    # Check if the main window is visible
    assert main_window.isVisible()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
