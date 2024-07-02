import almasim.ui as ui
import sys

app = ui.QApplication(sys.argv)
window = ui.ALMASimulator()
window.show()
sys.exit(app.exec())