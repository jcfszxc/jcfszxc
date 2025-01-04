#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/29 19:59
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : data_models.py
# @Description   : 

from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class WeatherData:
    """天气数据模型"""
    temperature: float
    humidity: int
    wind_speed: float
    description: str
    icon: str
    timestamp: datetime

@dataclass
class ForecastData:
    """天气预报数据模型"""
    date: datetime
    temp_min: float
    temp_max: float
    description: str
    icon: str