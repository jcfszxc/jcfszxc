import pytest
from datetime import datetime
from src.utils.validators import TaskValidator
from src.constants import Priority, TaskStatus

def test_name_validation():
    """测试任务名称验证"""
    assert TaskValidator.validate_name("有效名称") is True
    assert TaskValidator.validate_name("") is False
    assert TaskValidator.validate_name("  ") is False

def test_priority_validation():
    """测试优先级验证"""
    assert TaskValidator.validate_priority("高") is True
    assert TaskValidator.validate_priority("中") is True
    assert TaskValidator.validate_priority("低") is True
    assert TaskValidator.validate_priority("无效") is False

def test_status_validation():
    """测试状态验证"""
    assert TaskValidator.validate_status("未完成") is True
    assert TaskValidator.validate_status("已完成") is True
    assert TaskValidator.validate_status("无效状态") is False

def test_date_validation():
    """测试日期验证"""
    assert TaskValidator.validate_date("2024-01-01 12:00:00") is True
    assert TaskValidator.validate_date("invalid-date") is False
    assert TaskValidator.validate_date(None) is True