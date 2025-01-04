#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/29 20:00
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : utils.py
# @Description   : 

from datetime import datetime
from typing import List
import os

def get_icon_path(icon_code: str) -> str:
    """获取天气图标路径"""
    icon_folder = os.path.join(os.path.dirname(__file__), '..', 'resources', 'icons')
    return os.path.join(icon_folder, f"{icon_code}.png")

def format_temperature(temp: float) -> str:
    """格式化温度显示"""
    return f"{temp:.1f}°C"

def format_date(date: datetime) -> str:
    """格式化日期显示"""
    return date.strftime("%m-%d")
