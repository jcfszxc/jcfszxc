# zoomable_graphics_view.py
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建场景
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # 设置渲染属性
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 设置拖拽模式
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # 缩放系数
        self.zoom_factor = 1.15
        
        # 适应窗口大小
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            old_pos = self.mapToScene(event.pos())
            
            if event.angleDelta().y() > 0:
                scale_factor = self.zoom_factor
            else:
                scale_factor = 1 / self.zoom_factor
            
            self.scale(scale_factor, scale_factor)
            
            new_pos = self.mapToScene(event.pos())
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
            
            event.accept()
        else:
            super().wheelEvent(event)