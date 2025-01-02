#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/01/02 12:33
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : main_window.py
# @Description   : 

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                          QPushButton, QLabel, QFileDialog, QLineEdit,
                          QMessageBox, QSpinBox, QHBoxLayout)
from PyQt6.QtCore import QThread
from ..core.video_processor import VideoProcessor
from .processing_dialog import ProcessingDialog

class ProcessingThread(QThread):
    def __init__(self, processor, input_dir, output_dir):
        super().__init__()
        self.processor = processor
        self.input_dir = input_dir
        self.output_dir = output_dir
    
    def run(self):
        self.processor.process_directory(self.input_dir, self.output_dir)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频帧提取器")
        self.setup_ui()
        self.setup_processor()
    
    def setup_processor(self):
        self.processor = VideoProcessor()
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.processing_finished.connect(self.processing_finished)
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 输入目录选择
        input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("选择输入视频目录")
        input_layout.addWidget(self.input_path)
        
        input_button = QPushButton("浏览")
        input_button.clicked.connect(self.select_input_directory)
        input_layout.addWidget(input_button)
        layout.addLayout(input_layout)
        
        # 输出目录选择
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("选择输出图片目录")
        output_layout.addWidget(self.output_path)
        
        output_button = QPushButton("浏览")
        output_button.clicked.connect(self.select_output_directory)
        output_layout.addWidget(output_button)
        layout.addLayout(output_layout)
        
        # 开始处理按钮
        self.start_button = QPushButton("开始处理")
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)
        
        self.setFixedSize(500, 200)
    
    def select_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输入视频目录")
        if directory:
            self.input_path.setText(directory)
    
    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出图片目录")
        if directory:
            self.output_path.setText(directory)
    
    def start_processing(self):
        input_dir = self.input_path.text()
        output_dir = self.output_path.text()
        
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "错误", "请选择输入和输出目录")
            return
        
        self.processing_dialog = ProcessingDialog(self)
        self.processing_dialog.cancel_button.clicked.connect(self.cancel_processing)
        
        self.processing_thread = ProcessingThread(
            self.processor, input_dir, output_dir
        )
        self.processing_thread.start()
        
        self.processing_dialog.exec()
    
    def cancel_processing(self):
        if self.processing_dialog.cancel_button.text() == "关闭":
            self.processing_dialog.close()
        else:
            self.processor.stop_processing()
            self.processing_thread.quit()
            self.processing_thread.wait()
            self.processing_dialog.close()
    
    def update_progress(self, filename, current, total):
        if hasattr(self, 'processing_dialog'):
            self.processing_dialog.update_progress(filename, current, total)
    
    def processing_finished(self, total_saved, total_errors):
        if hasattr(self, 'processing_dialog'):
            self.processing_dialog.set_finished(total_saved, total_errors)