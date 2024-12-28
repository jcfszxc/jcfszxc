#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:53
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : validators.py
# @Description   : 

from datetime import datetime
from typing import Optional
from src.constants import Priority, TaskStatus

class TaskValidator:
    """任务数据验证器"""
    
    @staticmethod
    def validate_name(name: str) -> bool:
        """验证任务名称"""
        return bool(name and len(name.strip()) > 0)
    
    @staticmethod
    def validate_priority(priority: str) -> bool:
        """验证优先级"""
        try:
            Priority(priority)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_status(status: str) -> bool:
        """验证任务状态"""
        try:
            TaskStatus(status)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_date(date_str: Optional[str]) -> bool:
        """验证日期格式"""
        if not date_str:
            return True
        try:
            datetime.fromisoformat(date_str)
            return True
        except ValueError:
            return False