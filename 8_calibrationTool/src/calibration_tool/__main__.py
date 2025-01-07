# src/calibration_tool/__main__.py

import sys
from PyQt5.QtWidgets import QApplication
from calibration_tool.gui.main_window import CalibrationGUI

def main():
    """主程序入口"""
    app = QApplication(sys.argv)
    gui = CalibrationGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()