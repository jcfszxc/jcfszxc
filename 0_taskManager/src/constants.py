#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:51
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : constants.py
# @Description   : 

from enum import Enum

class Priority(Enum):
    """任务优先级枚举"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "未完成"
    COMPLETED = "已完成"
    ARCHIVED = "已归档"

# UI常量
class UIConstants:
    """UI相关常量"""
    WINDOW_TITLE = "个人任务管理器"
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    
    # 字体设置
    FONT_FAMILY = "Microsoft YaHei"
    FONT_SIZE_NORMAL = 10
    FONT_SIZE_LARGE = 12
    
    # 颜色设置
    COLOR_PRIMARY = "#2196F3"
    COLOR_SUCCESS = "#4CAF50"
    COLOR_WARNING = "#FFC107"
    COLOR_DANGER = "#F44336"
    COLOR_INFO = "#2196F3"
    
    # 边距设置
    MARGIN_SMALL = 5
    MARGIN_NORMAL = 10
    MARGIN_LARGE = 15