#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/29 20:01
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : main_window.py
# @Description   : 

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                          QPushButton, QComboBox)
from PyQt6.QtCore import QTimer
from .weather_widget import WeatherWidget
from ..core.api import WeatherAPI
from config import DEFAULT_CITIES

class MainWindow(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("天气预报")
        self.setMinimumSize(800, 400)
        self.setup_ui()
        
    def setup_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # 顶部控制栏
        control_bar = QWidget()
        control_layout = QHBoxLayout()
        
        self.city_combo = QComboBox()
        for city in DEFAULT_CITIES:
            self.city_combo.addItem(city['name'], city['code'])
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.update_weather)
        
        control_layout.addWidget(self.city_combo)
        control_layout.addWidget(refresh_btn)
        control_layout.addStretch()
        control_bar.setLayout(control_layout)
        
        # 天气显示组件
        self.weather_widget = WeatherWidget()
        
        layout.addWidget(control_bar)
        layout.addWidget(self.weather_widget)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # 设置自动更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_weather)
        self.update_timer.start(300000)  # 5分钟更新一次
        
        # 初始更新
        self.update_weather()
        
    def update_weather(self):
        """更新天气数据"""
        city_code = self.city_combo.currentData()
        
        # 获取当前天气
        current_weather = WeatherAPI.get_current_weather(city_code)
        if current_weather:
            self.weather_widget.update_current_weather(current_weather)
        
        # 获取天气预报
        forecast = WeatherAPI.get_forecast(city_code)
        if forecast:
            self.weather_widget.update_forecast(forecast)