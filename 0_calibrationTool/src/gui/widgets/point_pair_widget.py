#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/15 20:18
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : point_pair_widget.py
# @Description   : 



from typing import Tuple, Callable
from PyQt5.QtWidgets import QPushButton, QMenu
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent

class PointPairWidget(QPushButton):
    """特征点对显示组件"""
    
    STYLE_TEMPLATE = """
        QPushButton {
            background-color: #f0f0f0;
            border: 2px solid #c0c0c0;
            border-radius: 5px;
            padding: 5px;
            text-align: center;  /* 添加文本居中对齐 */
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QPushButton[selected="true"] {
            background-color: #90EE90;
            border: 2px solid #008000;
        }
    """
    
    def __init__(
        self,
        index: int,
        visible_point: Tuple[float, float],
        ir_point: Tuple[float, float],
        click_handler: Callable[[int], None],
        delete_handler: Callable[[int], None]
    ):
        """
        Args:
            index: 点对索引
            visible_point: 可见光图像上的点坐标
            ir_point: 红外图像上的点坐标
            click_handler: 点击事件处理函数
            delete_handler: 删除点对的处理函数
        """
        # 创建按钮文本,显示索引和坐标
        button_text = (
            f"点对 {index + 1}\n"
            f"可见光: ({int(visible_point[0])}, {int(visible_point[1])})\n"
            f"红外: ({int(ir_point[0])}, {int(ir_point[1])})"
        )
        super().__init__(button_text)
        
        self.setMinimumSize(120, 80)  # 增加按钮大小以容纳更多文本
        self.setStyleSheet(self.STYLE_TEMPLATE)
        
        tooltip = (
            f"可见光: ({int(visible_point[0])}, {int(visible_point[1])})\n"
            f"红外: ({int(ir_point[0])}, {int(ir_point[1])})"
        )
        self.setToolTip(tooltip)
        
        self.index = index
        self.click_handler = click_handler
        self.delete_handler = delete_handler
        
        # 不再直接连接clicked信号，而是通过mousePressEvent处理
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def mousePressEvent(self, event: QMouseEvent):
        """重写鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            # 左键点击执行选择操作
            self.click_handler(self.index)
        elif event.button() == Qt.RightButton:
            # 右键点击显示上下文菜单
            # 注意：如果设置了CustomContextMenu策略，这里不需要额外处理
            pass
        super().mousePressEvent(event)
    
    def show_context_menu(self, position):
        """显示右键菜单"""
        context_menu = QMenu(self)
        delete_action = context_menu.addAction("删除点对")
        
        # 连接删除动作
        action = context_menu.exec_(self.mapToGlobal(position))
        if action == delete_action:
            self.delete_handler(self.index)

