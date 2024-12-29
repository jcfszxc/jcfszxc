#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/29 20:00
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : weather_widget.py
# @Description   : 

from typing import List
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                          QPushButton, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from ..core.data_models import WeatherData, ForecastData
from ..core.utils import get_icon_path, format_temperature, format_date

class WeatherWidget(QWidget):
    """天气显示组件"""
    refresh_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 当前天气部分
        current_weather = QWidget()
        current_layout = QHBoxLayout()
        
        self.temp_label = QLabel("N/A")
        self.temp_label.setStyleSheet("font-size: 36px; font-weight: bold;")
        
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(64, 64)
        
        self.desc_label = QLabel("N/A")
        self.humidity_label = QLabel("湿度: N/A")
        self.wind_label = QLabel("风速: N/A")
        
        current_layout.addWidget(self.temp_label)
        current_layout.addWidget(self.icon_label)
        current_layout.addStretch()
        
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.desc_label)
        info_layout.addWidget(self.humidity_label)
        info_layout.addWidget(self.wind_label)
        
        current_layout.addLayout(info_layout)
        current_weather.setLayout(current_layout)
        
        # 预报部分
        forecast_widget = QWidget()
        self.forecast_layout = QHBoxLayout()
        forecast_widget.setLayout(self.forecast_layout)
        
        layout.addWidget(current_weather)
        layout.addWidget(forecast_widget)
        self.setLayout(layout)
        
    def update_current_weather(self, weather: WeatherData):
        """更新当前天气显示"""
        self.temp_label.setText(format_temperature(weather.temperature))
        
        icon_path = get_icon_path(weather.icon)
        self.icon_label.setPixmap(QPixmap(icon_path).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio))
        
        self.desc_label.setText(weather.description)
        self.humidity_label.setText(f"湿度: {weather.humidity}%")
        self.wind_label.setText(f"风速: {weather.wind_speed}m/s")
        
    def update_forecast(self, forecast_data: List[ForecastData]):
        """更新天气预报显示"""
        # 清除现有预报
        for i in reversed(range(self.forecast_layout.count())): 
            self.forecast_layout.itemAt(i).widget().setParent(None)
        
        # 添加新预报
        for forecast in forecast_data:
            forecast_day = QWidget()
            layout = QVBoxLayout()
            
            date_label = QLabel(format_date(forecast.date))
            icon_label = QLabel()
            icon_label.setPixmap(QPixmap(get_icon_path(forecast.icon)).scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio))
            temp_label = QLabel(f"{format_temperature(forecast.temp_min)} - {format_temperature(forecast.temp_max)}")
            
            layout.addWidget(date_label, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(temp_label, alignment=Qt.AlignmentFlag.AlignCenter)
            
            forecast_day.setLayout(layout)
            self.forecast_layout.addWidget(forecast_day)