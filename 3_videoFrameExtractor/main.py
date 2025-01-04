#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/01/02 12:34
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : main.py
# @Description   : 

import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()