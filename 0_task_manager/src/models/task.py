#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:52
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : task.py
# @Description   : 

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from src.constants import Priority, TaskStatus

@dataclass
class Task:
    """任务数据类"""
    id: Optional[int]
    name: str
    description: str
    priority: Priority
    status: TaskStatus
    created_at: datetime
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @classmethod
    def create(cls, name: str, priority: Priority, description: str = "",
              due_date: Optional[datetime] = None) -> 'Task':
        """创建新任务"""
        return cls(
            id=None,
            name=name,
            description=description,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            due_date=due_date,
            completed_at=None
        )
    
    def complete(self):
        """完成任务"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def reopen(self):
        """重新打开任务"""
        self.status = TaskStatus.PENDING
        self.completed_at = None