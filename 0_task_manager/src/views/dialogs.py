#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:55
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : dialogs.py
# @Description   : 

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from src.config import Config

class AboutDialog(QDialog):
    """关于对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("关于")
        layout = QVBoxLayout(self)
        
        app_info = (
            f"{Config.APP_NAME}\n"
            f"版本: {Config.APP_VERSION}\n\n"
            "个人任务管理工具\n"
            "帮助您更好地管理和完成任务"
        )
        
        info_label = QLabel(app_info)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)