#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/13 00:05
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : settings.py
# @Description   : 该模块定义了可见光-红外图像标定工具的配置类，包括GUI界面设置和图像处理参数



from dataclasses import dataclass
from typing import Tuple

@dataclass
class GUISettings:
    """GUI配置类"""
    WINDOW_TITLE: str = "可见光-红外图像标定工具"
    MIN_IMAGE_SIZE: Tuple[int, int] = (480, 320)
    POINTS_LIST_HEIGHT: Tuple[int, int] = (100, 200)  # (min, max)

@dataclass
class ImageSettings:
    """图像处理配置类"""
    SUPPORTED_FORMATS: Tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")
    NORMAL_POINT_COLOR: Tuple[int, int, int] = (0, 255, 0)  # 绿色
    HIGHLIGHT_POINT_COLOR: Tuple[int, int, int] = (255, 0, 0)  # 红色
    NORMAL_POINT_RADIUS: int = 5
    HIGHLIGHT_POINT_RADIUS: int = 8