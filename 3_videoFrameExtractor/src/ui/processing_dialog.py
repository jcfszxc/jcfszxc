#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/01/02 12:33
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : processing_dialog.py
# @Description   : 

from PyQt6.QtWidgets import (QDialog, QProgressBar, QLabel, 
                          QPushButton, QVBoxLayout)
from PyQt6.QtCore import Qt

class ProcessingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("处理进度")
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.current_file_label = QLabel("正在处理: ")
        layout.addWidget(self.current_file_label)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        
        self.cancel_button = QPushButton("取消")
        layout.addWidget(self.cancel_button)
        
        self.setFixedSize(400, 150)
    
    def update_progress(self, filename, current, total):
        self.current_file_label.setText(f"正在处理: {filename}")
        percentage = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(percentage)
        self.status_label.setText(f"进度: {current}/{total} 帧")
    
    def set_finished(self, total_saved, total_errors):
        self.current_file_label.setText("处理完成!")
        self.status_label.setText(
            f"共保存 {total_saved} 帧图像，遇到 {total_errors} 个错误"
        )
        self.progress_bar.setValue(100)
        self.cancel_button.setText("关闭")