#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:55
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : task_widget.py
# @Description   : 

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                           QLineEdit, QComboBox, QTableWidget, QTableWidgetItem,
                           QMessageBox, QHeaderView, QDialog, QLabel, QDateTimeEdit)
from PyQt6.QtCore import Qt, QDateTime
from src.constants import Priority, TaskStatus, UIConstants
from src.controllers.task_controller import TaskController
from src.utils.validators import TaskValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TaskListWidget(QWidget):
    """任务列表部件"""
    
    def __init__(self, task_controller: TaskController):
        super().__init__()
        self.task_controller = task_controller
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建任务输入区域
        input_layout = QHBoxLayout()
        
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("输入任务名称")
        input_layout.addWidget(self.task_input)
        
        self.priority_combo = QComboBox()
        self.priority_combo.addItems([p.value for p in Priority])
        input_layout.addWidget(self.priority_combo)
        
        self.due_date_edit = QDateTimeEdit(QDateTime.currentDateTime())
        self.due_date_edit.setCalendarPopup(True)
        input_layout.addWidget(self.due_date_edit)
        
        add_button = QPushButton("添加任务")
        add_button.clicked.connect(self.add_task)
        input_layout.addWidget(add_button)
        
        layout.addLayout(input_layout)
        
        # 创建任务表格
        self.task_table = QTableWidget()
        self.task_table.setColumnCount(6)
        self.task_table.setHorizontalHeaderLabels(
            ["任务名称", "优先级", "截止日期", "状态", "创建时间", "操作"]
        )
        self.task_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.task_table)
        
        # 加载任务列表
        self.refresh_tasks()
    
    def delete_task(self, task):
        """删除任务"""
        reply = QMessageBox.question(
            self,
            "确认删除",
            "确定要删除任务 '{}' 吗？".format(task.name),  # 修改了字符串格式化方式
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.task_controller.delete_task(task.id)
                self.refresh_tasks()
                logger.info("Task deleted: {}".format(task.name))
            except Exception as e:
                QMessageBox.critical(self, "错误", "删除任务失败：{}".format(str(e)))
                logger.error("Failed to delete task: {}".format(str(e)))

    def add_task(self):
        """添加新任务"""
        name = self.task_input.text().strip()
        if not TaskValidator.validate_name(name):
            QMessageBox.warning(self, "警告", "任务名称不能为空！")
            return
        
        priority = Priority(self.priority_combo.currentText())
        due_date = self.due_date_edit.dateTime().toPyDateTime()
        
        try:
            self.task_controller.create_task(name, priority, due_date=due_date)
            self.task_input.clear()
            self.refresh_tasks()
            logger.info("Task added: {}".format(name))
        except Exception as e:
            QMessageBox.critical(self, "错误", "添加任务失败：{}".format(str(e)))
            logger.error("Failed to add task: {}".format(str(e)))
    
    def refresh_tasks(self):
        """刷新任务列表"""
        tasks = self.task_controller.get_all_tasks()
        self.task_table.setRowCount(len(tasks))
        
        for row, task in enumerate(tasks):
            # 任务名称
            self.task_table.setItem(row, 0, QTableWidgetItem(task.name))
            # 优先级
            self.task_table.setItem(row, 1, QTableWidgetItem(task.priority.value))
            # 截止日期
            due_date = task.due_date.strftime("%Y-%m-%d %H:%M") if task.due_date else "无"
            self.task_table.setItem(row, 2, QTableWidgetItem(due_date))
            # 状态
            self.task_table.setItem(row, 3, QTableWidgetItem(task.status.value))
            # 创建时间
            self.task_table.setItem(row, 4, QTableWidgetItem(
                task.created_at.strftime("%Y-%m-%d %H:%M")
            ))
            
            # 操作按钮
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            
            # 编辑按钮
            edit_btn = QPushButton("编辑")
            edit_btn.clicked.connect(lambda checked, t=task: self.edit_task(t))
            btn_layout.addWidget(edit_btn)
            
            # 完成/重开按钮
            toggle_btn = QPushButton(
                "重开" if task.status == TaskStatus.COMPLETED else "完成"
            )
            toggle_btn.clicked.connect(lambda checked, t=task: self.toggle_task(t))
            btn_layout.addWidget(toggle_btn)
            
            # 删除按钮
            delete_btn = QPushButton("删除")
            delete_btn.clicked.connect(lambda checked, t=task: self.delete_task(t))
            btn_layout.addWidget(delete_btn)
            
            self.task_table.setCellWidget(row, 5, btn_widget)
    
    def edit_task(self, task):
        """编辑任务"""
        dialog = TaskEditDialog(task, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_tasks()
    
    def toggle_task(self, task):
        """切换任务状态"""
        if task.status == TaskStatus.COMPLETED:
            task.reopen()
        else:
            task.complete()
        
        try:
            self.task_controller.update_task(task)
            self.refresh_tasks()
            logger.info(f"Task status toggled: {task.name}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新任务状态失败：{str(e)}")
            logger.error(f"Failed to toggle task status: {str(e)}")
    
    def delete_task(self, task):
        """删除任务"""
        reply = QMessageBox.question(
            self,
            "确认删除",
            f'确定要删除任务"{task.name}"吗？',  # 修改引号的使用方式
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.task_controller.delete_task(task.id)
                self.refresh_tasks()
                logger.info(f"Task deleted: {task.name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除任务失败：{str(e)}")
                logger.error(f"Failed to delete task: {str(e)}")

class TaskEditDialog(QDialog):
    """任务编辑对话框"""
    
    def __init__(self, task, parent=None):
        super().__init__(parent)
        self.task = task
        self.task_controller = parent.task_controller
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("编辑任务")
        layout = QVBoxLayout(self)
        
        # 任务名称
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("任务名称:"))
        self.name_input = QLineEdit(self.task.name)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # 优先级
        priority_layout = QHBoxLayout()
        priority_layout.addWidget(QLabel("优先级:"))
        self.priority_combo = QComboBox()
        self.priority_combo.addItems([p.value for p in Priority])
        self.priority_combo.setCurrentText(self.task.priority.value)
        priority_layout.addWidget(self.priority_combo)
        layout.addLayout(priority_layout)
        
        # 截止日期
        due_date_layout = QHBoxLayout()
        due_date_layout.addWidget(QLabel("截止日期:"))
        self.due_date_edit = QDateTimeEdit(
            self.task.due_date if self.task.due_date else QDateTime.currentDateTime()
        )
        self.due_date_edit.setCalendarPopup(True)
        due_date_layout.addWidget(self.due_date_edit)
        layout.addLayout(due_date_layout)
        
        # 按钮
        button_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_task)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def save_task(self):
        """保存任务"""
        name = self.name_input.text().strip()
        if not TaskValidator.validate_name(name):
            QMessageBox.warning(self, "警告", "任务名称不能为空！")
            return
        
        self.task.name = name
        self.task.priority = Priority(self.priority_combo.currentText())
        self.task.due_date = self.due_date_edit.dateTime().toPyDateTime()
        
        try:
            self.task_controller.update_task(self.task)
            self.accept()
            logger.info(f"Task updated: {self.task.name}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新任务失败：{str(e)}")
            logger.error(f"Failed to update task: {str(e)}")