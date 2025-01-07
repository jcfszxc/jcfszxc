from typing import Optional, Tuple, Callable
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import numpy as np

class ImageDisplayWidget(QLabel):
    """图像显示组件"""
    magnifier_moved = pyqtSignal(QLabel, QPoint, np.ndarray)  # 添加信号用于发送鼠标移动事件
    
    def __init__(self, title: str, min_size: tuple, click_handler=None):
        super().__init__()
        self.title = title
        self.click_handler = click_handler
        
        # 设置最小尺寸
        self.setMinimumSize(*min_size)
        
        # 设置对齐方式
        self.setAlignment(Qt.AlignCenter)
        
        # 启用鼠标追踪
        self.setMouseTracking(True)
        
        # 存储原始图像
        self.original_image = None
        
        # 设置边框和标题
        self.setStyleSheet("""
            QLabel {
                border: 1px solid #CCCCCC;
                background-color: #F5F5F5;
                padding: 5px;
            }
        """)
        
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if self.click_handler:
            self.click_handler(event)
            
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        super().mouseMoveEvent(event)
        if self.original_image is not None:
            self.magnifier_moved.emit(self, event.pos(), self.original_image)
            
    def display_image(self, image: np.ndarray):
        """显示图像
        
        Args:
            image: 要显示的图像数组
        """
        if image is None:
            return
            
        # 保存原始numpy数组用于放大镜功能
        self.original_image = image.copy()
            
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        
        # 将NumPy数组转换为QImage
        q_image = QImage(
            image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        # 将QImage转换为QPixmap并显示
        pixmap = QPixmap.fromImage(q_image)
        self.setPixmap(pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))