#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:52
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : task_controller.py
# @Description   : 

from typing import List, Optional
from datetime import datetime
from src.models.task import Task
from src.models.database import Database
from src.constants import Priority, TaskStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TaskController:
    """任务控制器"""
    
    def __init__(self):
        self.db = Database()
    
    def create_task(self, name: str, priority: Priority,
                   description: str = "", due_date: Optional[datetime] = None) -> Task:
        """创建新任务"""
        task = Task.create(name, priority, description, due_date)
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tasks (name, description, priority, status, created_at, due_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (task.name, task.description, task.priority.value,
                 task.status.value, task.created_at.isoformat(),
                 task.due_date.isoformat() if task.due_date else None))
            conn.commit()
            task.id = cursor.lastrowid
        
        logger.info(f"Created task: {task.name}")
        return task
    
    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM tasks ORDER BY created_at DESC')
            rows = cursor.fetchall()
            
            return [self._row_to_task(row) for row in rows]
    
    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        """根据ID获取任务"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM tasks WHERE id = ?', (task_id,))
            row = cursor.fetchone()
            
            return self._row_to_task(row) if row else None
    
    def update_task(self, task: Task):
        """更新任务"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE tasks
                SET name = ?, description = ?, priority = ?, status = ?,
                    due_date = ?, completed_at = ?
                WHERE id = ?
            ''', (task.name, task.description, task.priority.value,
                 task.status.value,
                 task.due_date.isoformat() if task.due_date else None,
                 task.completed_at.isoformat() if task.completed_at else None,
                 task.id))
            conn.commit()
        
        logger.info(f"Updated task: {task.name}")
    
    def delete_task(self, task_id: int):
        """删除任务"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
            conn.commit()
        
        logger.info(f"Deleted task with ID: {task_id}")
    
    def _row_to_task(self, row) -> Task:
        """将数据库行转换为Task对象"""
        return Task(
            id=row[0],
            name=row[1],
            description=row[2],
            priority=Priority(row[3]),
            status=TaskStatus(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            due_date=datetime.fromisoformat(row[6]) if row[6] else None,
            completed_at=datetime.fromisoformat(row[7]) if row[7] else None
        )