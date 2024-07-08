from .ui import ALMASimulator, QApplication
import sys


def run():
    app = QApplication(sys.argv)
    window = ALMASimulator()
    window.show()
    sys.exit(app.exec())
