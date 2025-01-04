#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:56
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : main.py
# @Description   : 

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.views.main_window import MainWindow
from src.config import Config
from src.utils.logger import get_logger

# 初始化配置
Config.init_app()

# 获取logger
logger = get_logger(__name__)

def main():
    """程序入口函数"""
    try:
        # 创建QApplication实例
        app = QApplication(sys.argv)
        
        # 设置应用程序样式
        app.setStyle('Fusion')
        
        # 创建并显示主窗口
        window = MainWindow()
        window.show()
        
        logger.info(f"Application {Config.APP_NAME} started")
        
        # 进入事件循环
        sys.exit(app.exec())
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == '__main__':
    main()