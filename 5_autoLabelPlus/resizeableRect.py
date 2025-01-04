from PyQt5.QtGui import QPen, QColor, QBrush
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsRectItem
from PyQt5.QtCore import Qt, QRectF, QTimer  # 添加QTimer的导入

class ResizableRectItem(QGraphicsRectItem):
    """可调整大小的矩形框，带有边界限制"""
    
    # 定义控制柄位置的常量
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 2
    BOTTOM_LEFT = 3


    def __init__(self, rect, parent=None):
        super().__init__(rect, parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setPen(QPen(QColor(255, 0, 0), 2))
        
        # 创建四个角点控制柄
        self.handles = []
        self.handle_size = 8
        
        # 控制柄状态相关
        self.current_handle = None  # 当前拖动的控制柄
        self.hovered_handle = None  # 当前鼠标悬停的控制柄
        self.original_rect = None   # 开始拖动时的矩形
        self.start_pos = None       # 开始拖动时的位置
        self.is_resizing = False    # 是否正在调整大小
        
        # 控制柄颜色配置
        self.handle_colors = {
            'normal': {
                'pen': QColor(128, 128, 255),
                'brush': QColor(128, 128, 255)
            },
            'selected': {
                'pen': QColor(0, 0, 255),
                'brush': QColor(0, 0, 255)
            },
            'hover': {
                'pen': QColor(0, 160, 255),
                'brush': QColor(0, 160, 255)
            },
            'active': {
                'pen': QColor(255, 165, 0),  # 橙色
                'brush': QColor(255, 165, 0)
            }
        }
        
        # 选中状态的样式
        self.selected_style = {
            'pen': QPen(Qt.blue, 2, Qt.DashLine),
            'brush': QBrush(QColor(0, 0, 255, 50))  # 半透明蓝色，alpha=50
        }
        
        # 设置接受悬停事件
        self.setAcceptHoverEvents(True)
        
        # 添加最小尺寸限制
        self.min_size = 10
        
        # 初始化控制柄  
        self.updateHandles()
        
        # 添加对主窗口的引用
        self.main_window = None
        
        # 添加一个计时器用于控制更新频率
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(100)  # 100ms的防抖间隔
        self.update_timer.timeout.connect(self.delayed_update)
        
    def updateHandles(self):
        """更新控制柄位置和外观"""
        if not self.handles:
            for _ in range(4):
                handle = QGraphicsRectItem(self)
                handle.setZValue(1)  # 确保控制柄始终显示在矩形上方
                self.handles.append(handle)
        
        rect = self.rect()
        positions = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomRight(),
            rect.bottomLeft()
        ]
        
        for handle, pos in zip(self.handles, positions):
            handle.setRect(QRectF(
                pos.x() - self.handle_size/2,
                pos.y() - self.handle_size/2,
                self.handle_size,
                self.handle_size
            ))
        
        self.updateHandleColors()

    def updateHandleColors(self):
        """更新控制柄的颜色状态"""
        for i, handle in enumerate(self.handles):
            if i == self.current_handle:
                # 正在拖动的控制柄使用橙色
                colors = self.handle_colors['active']
                pen_width = 2
            elif i == self.hovered_handle:
                # 鼠标悬停的控制柄使用亮蓝色
                colors = self.handle_colors['hover']
                pen_width = 2
            elif self.isSelected():
                # 选中状态的控制柄使用蓝色
                colors = self.handle_colors['selected']
                pen_width = 1
            else:
                # 普通状态使用浅蓝色
                colors = self.handle_colors['normal']
                pen_width = 1
            
            handle.setPen(QPen(colors['pen'], pen_width))
            handle.setBrush(QBrush(colors['brush']))
            
            # 根据状态调整控制柄大小
            center = handle.rect().center()
            size = self.handle_size + (2 if i in (self.current_handle, self.hovered_handle) else 0)
            handle.setRect(QRectF(
                center.x() - size/2,
                center.y() - size/2,
                size,
                size
            ))

    def hoverMoveEvent(self, event):
        """处理鼠标悬停移动事件"""
        if not self.is_resizing:  # 只在非调整大小状态下检测悬停
            old_hover = self.hovered_handle
            self.hovered_handle = None
            
            for i, handle in enumerate(self.handles):
                if handle.contains(handle.mapFromScene(event.scenePos())):
                    self.hovered_handle = i
                    self.setCursor(self.getCursorForHandle(i))
                    break
            
            if self.hovered_handle is None:
                self.setCursor(Qt.ArrowCursor)
            
            # 只在状态改变时更新颜色
            if old_hover != self.hovered_handle:
                self.updateHandleColors()
        
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        """处理鼠标离开事件"""
        if not self.is_resizing:  # 只在非调整大小状态下重置悬停状态
            self.hovered_handle = None
            self.setCursor(Qt.ArrowCursor)
            self.updateHandleColors()
        super().hoverLeaveEvent(event)
    
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        handle_idx = self.handle_at(event.scenePos())
        if handle_idx is not None:
            self.current_handle = handle_idx
            self.original_rect = self.rect()
            self.start_pos = event.scenePos()
            self.is_resizing = True  # 设置调整大小状态
            self.updateHandleColors()
            event.accept()
            return
        super().mousePressEvent(event)


    
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if self.is_resizing:
            self.current_handle = None
            self.original_rect = None
            self.start_pos = None
            self.is_resizing = False  # 重置调整大小状态
            self.updateHandleColors()
            # 确保在释放鼠标时进行一次更新
            if self.main_window:
                self.main_window.update_category_list()
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def getCursorForHandle(self, handle_index):
        """根据控制柄位置返回对应的光标形状"""
        if handle_index in (self.TOP_LEFT, self.BOTTOM_RIGHT):
            return Qt.SizeFDiagCursor
        elif handle_index in (self.TOP_RIGHT, self.BOTTOM_LEFT):
            return Qt.SizeBDiagCursor
        return Qt.ArrowCursor
    
    def handle_at(self, pos):
        """检查给定位置是否在控制柄上"""
        for i, handle in enumerate(self.handles):
            handle_pos = handle.mapFromScene(pos)
            if handle.contains(handle_pos):
                return i
        return None


    def ensureWithinBounds(self, new_rect):
        """确保矩形在场景边界内"""
        if self.scene():
            scene_rect = self.scene().sceneRect()
            
            # 如果新位置超出场景边界，则调整到边界位置
            if new_rect.left() < scene_rect.left():
                new_rect.moveLeft(scene_rect.left())
            if new_rect.right() > scene_rect.right():
                new_rect.moveRight(scene_rect.right())
            if new_rect.top() < scene_rect.top():
                new_rect.moveTop(scene_rect.top())
            if new_rect.bottom() > scene_rect.bottom():
                new_rect.moveBottom(scene_rect.bottom())
                
        return new_rect
        

    def delayed_update(self):
        """延迟更新，减少更新频率"""
        if self.main_window:
            self.main_window.update_category_list()

    def trigger_update(self):
        """触发更新，使用计时器进行防抖"""
        if hasattr(self, 'update_timer'):
            self.update_timer.start()

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件，添加边界限制"""
        old_pos = self.pos()
        old_rect = self.rect()
        
        if self.is_resizing and self.current_handle is not None:
            # 处理调整大小
            offset = event.scenePos() - self.start_pos
            new_rect = QRectF(self.original_rect)
            
            if self.current_handle == self.TOP_LEFT:
                new_rect.setTopLeft(new_rect.topLeft() + offset)
            elif self.current_handle == self.TOP_RIGHT:
                new_rect.setTopRight(new_rect.topRight() + offset)
            elif self.current_handle == self.BOTTOM_RIGHT:
                new_rect.setBottomRight(new_rect.bottomRight() + offset)
            elif self.current_handle == self.BOTTOM_LEFT:
                new_rect.setBottomLeft(new_rect.bottomLeft() + offset)
            
            # 检查新矩形是否满足最小尺寸要求
            if new_rect.width() >= self.min_size and new_rect.height() >= self.min_size:
                # 确保在边界内
                new_rect = self.ensureWithinBounds(new_rect)
                self.setRect(new_rect)
                self.updateHandles()
                # 触发更新
                self.trigger_update()
            event.accept()
        else:
            # 处理拖动
            super().mouseMoveEvent(event)
            
            # 获取新位置下的矩形在场景坐标系中的位置
            new_rect = self.sceneBoundingRect()
            
            # 检查是否超出场景边界
            if self.scene():
                scene_rect = self.scene().sceneRect()
                if not scene_rect.contains(new_rect):
                    # 如果超出边界，将位置调整回去
                    self.setPos(old_pos)
                    return
                
            self.updateHandles()
            # 如果位置确实发生了变化，触发更新
            if self.pos() != old_pos or self.rect() != old_rect:
                self.trigger_update()


    def paint(self, painter, option, widget):
        """自定义绘制方法，添加选中状态的半透明蒙版"""
        # 保存painter状态
        painter.save()
        
        try:
            # 如果是选中状态，先绘制半透明填充
            if self.isSelected():
                painter.setPen(self.selected_style['pen'])
                painter.setBrush(self.selected_style['brush'])
                painter.drawRect(self.rect())
            else:
                # 非选中状态，使用普通样式
                painter.setPen(self.pen())
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(self.rect())
        finally:
            # 确保在所有情况下都恢复painter状态
            painter.restore()