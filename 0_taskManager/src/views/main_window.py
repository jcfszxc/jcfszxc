#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:54
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : main_window.py
# @Description   : 

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QMenuBar, QStatusBar)
from PyQt6.QtCore import Qt
from src.constants import UIConstants
from src.views.task_widget import TaskListWidget
from src.controllers.task_controller import TaskController
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.task_controller = TaskController()
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle(UIConstants.WINDOW_TITLE)
        self.setGeometry(100, 100, UIConstants.WINDOW_WIDTH, UIConstants.WINDOW_HEIGHT)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 创建任务列表部件
        self.task_list = TaskListWidget(self.task_controller)
        layout.addWidget(self.task_list)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        logger.info("Main window UI initialized")
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        file_menu.addAction('导出任务', self.export_tasks)
        file_menu.addAction('导入任务', self.import_tasks)
        file_menu.addSeparator()
        file_menu.addAction('退出', self.close)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图')
        view_menu.addAction('刷新', self.task_list.refresh_tasks)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        help_menu.addAction('关于', self.show_about)
    
    def export_tasks(self):
        """导出任务列表"""
        # TODO: 实现任务导出功能
        self.status_bar.showMessage('导出功能开发中...')
    
    def import_tasks(self):
        """导入任务列表"""
        # TODO: 实现任务导入功能
        self.status_bar.showMessage('导入功能开发中...')
    
    def show_about(self):
        """显示关于对话框"""
        # TODO: 实现关于对话框
        self.status_bar.showMessage('关于对话框开发中...')