#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:58
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : test_models.py
# @Description   : 

import pytest
from datetime import datetime
from src.models.task import Task
from src.constants import Priority, TaskStatus

def test_task_creation():
    """测试任务创建"""
    task = Task.create(
        name="测试任务",
        priority=Priority.MEDIUM,
        description="这是一个测试任务"
    )
    
    assert task.name == "测试任务"
    assert task.priority == Priority.MEDIUM
    assert task.description == "这是一个测试任务"
    assert task.status == TaskStatus.PENDING
    assert task.created_at <= datetime.now()
    assert task.completed_at is None

def test_task_completion():
    """测试任务完成"""
    task = Task.create("测试任务", Priority.LOW)
    task.complete()
    
    assert task.status == TaskStatus.COMPLETED
    assert task.completed_at <= datetime.now()

def test_task_reopening():
    """测试重新打开任务"""
    task = Task.create("测试任务", Priority.HIGH)
    task.complete()
    task.reopen()
    
    assert task.status == TaskStatus.PENDING
    assert task.completed_at is None