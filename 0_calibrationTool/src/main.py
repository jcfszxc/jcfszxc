#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/12 19:22
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : main.py
# @Description   : 程序主入口文件，负责初始化应用程序和启动用户界面

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import CalibrationGUI

def main():
    """主程序入口"""
    app = QApplication(sys.argv)
    gui = CalibrationGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


