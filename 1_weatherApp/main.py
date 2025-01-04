#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/29 20:01
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : main.py
# @Description   : 

import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.ui.styles import MAIN_STYLE

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(MAIN_STYLE)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()