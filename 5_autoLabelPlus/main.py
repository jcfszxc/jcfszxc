from PyQt5 import QtWidgets, QtCore, QtGui
from ui_main import Ui_autoLabel  # 导入UI类
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QDialog, QGraphicsLineItem
from PyQt5.QtGui import QPixmap, QPen
from PyQt5.QtCore import Qt, QRectF
import os
from resizeableRect import ResizableRectItem
from category_dialog_implementation import CategoryDialog

from PyQt5.QtCore import QRectF
import json
import os

class AnnotationStorage:
    def __init__(self):
        self.annotations = {}  # 存储所有图片的标注信息
        
    def save_annotation(self, image_path, rect_items):
        """保存单个图片的标注信息"""
        annotations = []
        for rect_item in rect_items:
            rect = rect_item.rect()
            scene_pos = rect_item.scenePos()
            annotation = {
                'category': getattr(rect_item, 'category', ''),
                'x': rect.x() + scene_pos.x(),
                'y': rect.y() + scene_pos.y(),
                'width': rect.width(),
                'height': rect.height()
            }
            annotations.append(annotation)
        
        # 获取相对路径作为键
        if image_path.startswith(self.base_dir):
            rel_path = os.path.relpath(image_path, self.base_dir)
        else:
            rel_path = image_path
            
        self.annotations[rel_path] = annotations
        self._save_to_file()
        
    def load_annotation(self, image_path):
        """加载单个图片的标注信息"""
        if image_path.startswith(self.base_dir):
            rel_path = os.path.relpath(image_path, self.base_dir)
        else:
            rel_path = image_path
            
        return self.annotations.get(rel_path, [])
    
    def set_base_directory(self, directory):
        """设置基础目录，用于生成相对路径"""
        self.base_dir = directory
        self._load_from_file()
    
    def _get_annotation_file_path(self):
        """获取标注文件的路径"""
        if hasattr(self, 'base_dir'):
            return os.path.join(self.base_dir, 'annotations.json')
        return None
    
    def _save_to_file(self):
        """将标注信息保存到文件"""
        file_path = self._get_annotation_file_path()
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)
                
    def _load_from_file(self):
        """从文件加载标注信息"""
        file_path = self._get_annotation_file_path()
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
            except json.JSONDecodeError:
                print("标注文件损坏，创建新的标注记录")
                self.annotations = {}
        else:
            self.annotations = {}

# 添加自定义代理类
class ThumbnailDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text_height = 20  # 文本区域高度
        self.spacing = 5      # 文本和图标之间的间距
    
    def paint(self, painter, option, index):
        # 保存画家状态
        painter.save()
        
        # 获取项目数据
        icon = index.data(QtCore.Qt.DecorationRole)
        text = index.data(QtCore.Qt.DisplayRole)
        
        # 计算绘制区域
        rect = option.rect
        
        # 绘制选中状态背景
        if option.state & QtWidgets.QStyle.State_Selected:
            painter.fillRect(rect, option.palette.highlight())
        
        # 计算文本区域
        text_rect = QtCore.QRect(
            rect.left(),
            rect.top(),
            rect.width(),
            self.text_height
        )
        
        # 绘制文本（居中对齐）
        painter.setPen(option.palette.color(
            QtGui.QPalette.HighlightedText if option.state & QtWidgets.QStyle.State_Selected
            else QtGui.QPalette.Text
        ))
        painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWrapAnywhere, text)
        
        # 如果有图标，绘制在文本下方
        if icon:
            icon_rect = QtCore.QRect(
                rect.left() + self.spacing,
                rect.top() + self.text_height + self.spacing,
                rect.width() - 2 * self.spacing,
                rect.height() - self.text_height - 2 * self.spacing
            )
            icon.paint(painter, icon_rect)
        
        # 恢复画家状态
        painter.restore()
    
    def sizeHint(self, option, index):
        # 返回项目的建议大小
        icon_size = option.decorationSize
        total_height = self.text_height + 2 * self.spacing + icon_size.height()
        return QtCore.QSize(icon_size.width() + 2 * self.spacing, total_height)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # 创建UI对象
        self.ui = Ui_autoLabel()
        # 设置UI
        self.ui.setupUi(self)

        # 获取屏幕尺寸并设置窗口大小为屏幕的一定比例
        screen = QtWidgets.QApplication.primaryScreen().size()
        # 设置窗口大小为屏幕的80%
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

        # 将窗口居中显示
        self.center_window()
        
        # 设置最小尺寸，防止窗口被缩放得太小
        self.setMinimumSize(400, 300)

        # 连接按钮点击信号到槽函数
        self.ui.pushButtonOpenDir.clicked.connect(self.open_directory)
        self.ui.pushButtonNextImage.clicked.connect(self.next_image)
        self.ui.pushButtonPrevImage.clicked.connect(self.previous_image)
        
        # 连接文件列表的项目点击信号
        self.ui.fileListWidget.itemClicked.connect(self.on_file_item_clicked)
        # 连接缩略图列表的项目点击信号
        self.ui.thumbnailPreview.itemClicked.connect(self.on_thumbnail_clicked)
        
        # # 设置缩略图列表的其他属性
        # self.ui.thumbnailPreview.setSpacing(10)  # 设置缩略图间距
        # self.ui.thumbnailPreview.setResizeMode(QtWidgets.QListWidget.Adjust)

        # 设置缩略图列表的显示模式和属性
        self.ui.thumbnailPreview.setViewMode(QtWidgets.QListView.IconMode)
        self.ui.thumbnailPreview.setIconSize(QtCore.QSize(100, 100))
        self.ui.thumbnailPreview.setSpacing(10)
        self.ui.thumbnailPreview.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.ui.thumbnailPreview.setMovement(QtWidgets.QListWidget.Static)
        self.ui.thumbnailPreview.setWordWrap(True)
        
        # 创建并设置自定义代理
        self.thumbnail_delegate = ThumbnailDelegate(self.ui.thumbnailPreview)
        self.ui.thumbnailPreview.setItemDelegate(self.thumbnail_delegate)

        # Add after other initializations
        # self.ui.label.setAlignment(Qt.AlignCenter)  # Optional: center the text

        # 存储当前选择的目录路径
        self.current_directory = None
        self.current_image_index = -1
        
        # 添加矩形框绘制相关的属性
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.rect_items = []
        self.selected_rect = None  # 添加选中矩形的引用
        
        # 连接创建矩形框按钮
        self.ui.pushButtonCreateRectBox.clicked.connect(self.toggle_draw_mode)
        self.ui.pushButtonCreateRectBox.setText("Create RectBox")

        # 为graphicsView添加自定义场景
        self.scene = QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene)
        
        # 添加场景选择变化的信号连接
        self.scene.selectionChanged.connect(self.handle_selection_changed)

        # 添加鼠标事件跟踪
        self.ui.graphicsView.setMouseTracking(True)
        self.ui.graphicsView.viewport().installEventFilter(self)
        self.ui.graphicsView.viewport().setMouseTracking(True)
        
        # 添加图片边界属性
        self.image_bounds = None
        
        # 添加辅助定位线属性
        self.guide_line_h = None  # 水平辅助线
        self.guide_line_v = None  # 垂直辅助线

            
        # 添加类别列表的选择响应
        self.ui.categoryListWidget.itemClicked.connect(self.on_category_item_clicked)

        
        # 创建标注存储对象
        self.annotation_storage = AnnotationStorage()


        # 在这里可以添加其他初始化代码


        # Add context menu support for categoryListWidget
        self.ui.categoryListWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.categoryListWidget.customContextMenuRequested.connect(self.show_category_context_menu)
        

    def show_category_context_menu(self, position):
        """显示类别列表的右键菜单"""
        # 获取点击位置的项
        item = self.ui.categoryListWidget.itemAt(position)
        
        if item is not None:
            # 创建菜单
            context_menu = QtWidgets.QMenu(self)
            
            # 添加删除动作
            delete_action = context_menu.addAction("删除标注框")
            
            # 显示菜单并获取选择的动作
            action = context_menu.exec_(self.ui.categoryListWidget.viewport().mapToGlobal(position))
            
            if action == delete_action:
                # 获取关联的矩形项
                rect_item = item.data(QtCore.Qt.UserRole)
                if rect_item:
                    # 从场景中移除矩形
                    self.scene.removeItem(rect_item)
                    # 从矩形项列表中移除
                    if rect_item in self.rect_items:
                        self.rect_items.remove(rect_item)
                    # 从类别列表中移除项
                    row = self.ui.categoryListWidget.row(item)
                    self.ui.categoryListWidget.takeItem(row)
                    # 保存更新后的标注
                    self.save_current_annotations()


    def center_window(self):
        """将窗口移动到屏幕中央"""
        # 获取屏幕几何信息
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        # 获取窗口几何信息
        size = self.geometry()
        # 计算居中位置
        new_left = (screen.width() - size.width()) // 2
        new_top = (screen.height() - size.height()) // 2
        # 移动窗口
        self.move(new_left, new_top)


    def on_category_item_clicked(self, item):
        """处理类别列表项被点击的事件"""
        # 获取对应的矩形项
        rect_item = item.data(QtCore.Qt.UserRole)
        if rect_item:
            # 清除其他项的选择
            self.scene.clearSelection()
            # 选中对应的矩形
            rect_item.setSelected(True)
            # 确保矩形可见
            self.ui.graphicsView.ensureVisible(rect_item)

    def toggle_draw_mode(self):
        """切换绘制模式"""
        if self.ui.pushButtonCreateRectBox.text() == "Create RectBox":
            self.ui.pushButtonCreateRectBox.setText("Finish creating RectBox")
            self.drawing = True
            self.ui.graphicsView.viewport().setCursor(Qt.CrossCursor)
        else:
            self.ui.pushButtonCreateRectBox.setText("Create RectBox")
            self.drawing = False
            self.ui.graphicsView.viewport().setCursor(Qt.ArrowCursor)
            # 退出绘制模式时移除辅助线
            if self.guide_line_h is not None:
                self.scene.removeItem(self.guide_line_h)
                self.scene.removeItem(self.guide_line_v)
                self.guide_line_h = None
                self.guide_line_v = None

    def eventFilter(self, source, event):
        """事件过滤器，处理鼠标事件"""
        if source == self.ui.graphicsView.viewport():
            # Handle mouse move for coordinate display
            if event.type() == QtCore.QEvent.MouseMove:
                view_pos = event.pos()
                scene_pos = self.ui.graphicsView.mapToScene(view_pos)
                # Update label with current coordinates
                self.ui.label.setText(f"X = {int(scene_pos.x())},  Y = {int(scene_pos.y())}")
                
                # 检查鼠标是否在控制柄上
            
                
                # Handle drawing mode mouse move
                if self.drawing:
                    return self.handle_mouse_move(event)
                    
            # Handle other existing mouse events
            if self.drawing:
                if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    return self.handle_mouse_press(event)
                elif event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                    return self.handle_mouse_release(event)
                    
        return super().eventFilter(source, event)

    
    def handle_mouse_move(self, event):
        """处理鼠标移动事件（添加边界限制和辅助定位线）"""
        view_pos = event.pos()
        scene_pos = self.ui.graphicsView.mapToScene(view_pos)
        
        # 更新或创建辅助定位线
        if self.drawing and self.image_bounds:
            try:
                # 确保在图片边界内
                x = max(self.image_bounds.left(), min(scene_pos.x(), self.image_bounds.right()))
                y = max(self.image_bounds.top(), min(scene_pos.y(), self.image_bounds.bottom()))
                
                # 如果辅助线不存在或已被删除，重新创建它们
                if self.guide_line_h is None or self.guide_line_v is None:
                    pen = QPen(Qt.green, 1, Qt.DashLine)  # 创建虚线画笔
                    
                    # 移除旧的辅助线（如果存在）
                    if self.guide_line_h:
                        self.scene.removeItem(self.guide_line_h)
                    if self.guide_line_v:
                        self.scene.removeItem(self.guide_line_v)
                        
                    # 创建新的辅助线
                    self.guide_line_h = self.scene.addLine(0, 0, 0, 0, pen)
                    self.guide_line_v = self.scene.addLine(0, 0, 0, 0, pen)
                
                # 更新辅助线位置
                # 使用scene.addLine返回的QGraphicsLineItem对象的setLine方法
                self.guide_line_h.setLine(
                    self.image_bounds.left(), y,
                    self.image_bounds.right(), y
                )
                self.guide_line_v.setLine(
                    x, self.image_bounds.top(),
                    x, self.image_bounds.bottom()
                )
                
            except RuntimeError:
                # 如果出现对象已删除的错误，重置辅助线引用
                self.guide_line_h = None
                self.guide_line_v = None
                return True
                
        # 原有的矩形绘制逻辑
        if not hasattr(self, 'current_rect') or self.current_rect is None:
            return False
            
        if not self.drawing:
            return False

        # 更新矩形大小
        if hasattr(self, 'start_point'):
            rect = QRectF(self.start_point, scene_pos).normalized()
            if self.image_bounds:
                rect = rect.intersected(self.image_bounds)
            self.current_rect.setRect(rect)
            self.current_rect.updateHandles()
        return True

    def handle_mouse_press(self, event):
        """处理鼠标按下事件（添加边界检查）"""
        if not self.drawing:
            return False
        
        view_pos = event.pos()
        scene_pos = self.ui.graphicsView.mapToScene(view_pos)
        
        # 确保起始点在图片边界内
        if self.image_bounds and not self.image_bounds.contains(scene_pos):
            return False
        
        self.start_point = scene_pos
        self.current_rect = ResizableRectItem(QRectF(scene_pos, scene_pos))
        self.scene.addItem(self.current_rect)
        return True

    def handle_selection_changed(self):
        """处理场景中的选择变化"""
        selected_items = self.scene.selectedItems()
        if selected_items:
            selected_rect = selected_items[0]
            if isinstance(selected_rect, ResizableRectItem):
                self.selected_rect = selected_rect
                self.show_category_dialog()
        else:
            self.selected_rect = None


    def show_category_dialog(self):
        """显示类别选择对话框"""
        dialog = CategoryDialog(self)
        
        # 如果矩形已有类别，预先填充
        if hasattr(self.selected_rect, 'category'):
            dialog.lineEdit.setText(self.selected_rect.category)
        
        if dialog.exec_() == QDialog.Accepted:
            category = dialog.get_selected_category()
            auto_label = dialog.autoLabelCheckBox.isChecked()
            
            # 保存类别到矩形对象
            self.selected_rect.category = category
            
            # 在矩形上显示类别标签
            self.update_rect_label(self.selected_rect, category)
            
            # 更新类别列表显示
            self.update_category_list()

            # 保存标注
            self.save_current_annotations()
            
            print(f"Category set to: {category}")
            print(f"Auto label: {auto_label}")

    def update_rect_label(self, rect_item, category):
        """更新矩形框上的类别标签"""
        # 删除现有的标签（如果有）
        for child in rect_item.childItems():
            if isinstance(child, QtWidgets.QGraphicsTextItem):
                rect_item.scene().removeItem(child)
                break
        
        # 创建新的标签
        text_item = self.scene.addText(category)
        text_item.setDefaultTextColor(Qt.red)
        text_item.setParentItem(rect_item)
        
        # 设置标签位置（在矩形框的左上角）
        rect = rect_item.rect()
        text_item.setPos(rect.left(), rect.top() - 20)  # 将标签放在矩形上方


    def handle_mouse_release(self, event):
        """处理鼠标释放事件（添加边界检查）"""
        if not self.drawing or not self.start_point or not self.current_rect:
            return False
        
        view_pos = event.pos()
        end_pos = self.ui.graphicsView.mapToScene(view_pos)
        
        # 确保结束点在图片边界内
        if self.image_bounds:
            end_pos.setX(max(self.image_bounds.left(), min(end_pos.x(), self.image_bounds.right())))
            end_pos.setY(max(self.image_bounds.top(), min(end_pos.y(), self.image_bounds.bottom())))
        
        rect = QRectF(self.start_point, end_pos).normalized()
        
        # 限制矩形在图片边界内
        if self.image_bounds:
            rect = rect.intersected(self.image_bounds)
        
        # 如果矩形太小，则删除
        if rect.width() < 5 or rect.height() < 5:
            self.scene.removeItem(self.current_rect)
        else:
            self.current_rect.setRect(rect)
            self.current_rect.updateHandles()
            # 设置主窗口引用
            self.current_rect.main_window = self
            self.rect_items.append(self.current_rect)
            # 设置矩形可选择和移动
            self.current_rect.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
            self.current_rect.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
            # 在创建完矩形后立即显示类别对话框
            self.selected_rect = self.current_rect
            self.show_category_dialog()
            # 保存标注
            self.save_current_annotations()
        
        self.start_point = None
        self.current_rect = None
        return True

    def open_directory(self):
        """打开文件夹选择对话框并显示第一张图片"""
        start_dir = self.current_directory if self.current_directory else "./"
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            start_dir,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )

        if directory:
            self.current_directory = directory
            print(f"选择的文件夹路径: {directory}")
            # 设置标注存储的基础目录
            self.annotation_storage.set_base_directory(directory)
            
            # 定义支持的图片格式
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
            
            # 递归获取目录下所有图片文件
            def find_images(folder):
                image_files = []
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    if os.path.isfile(item_path):
                        if item.lower().endswith(image_extensions):
                            image_files.append(item_path)
                    elif os.path.isdir(item_path):
                        image_files.extend(find_images(item_path))
                return sorted(image_files)
            
            # 获取所有图片
            self.image_files = find_images(directory)
                    
            if self.image_files:
                # 清空文件列表和缩略图列表
                self.ui.fileListWidget.clear()
                self.ui.thumbnailPreview.clear()
                
                # 添加所有图片到列表和缩略图
                for image_path in self.image_files:
                    # 获取相对路径
                    rel_path = os.path.relpath(image_path, directory)
                    
                    # 添加到文件列表
                    file_item = QtWidgets.QListWidgetItem(rel_path)
                    file_item.setData(QtCore.Qt.UserRole, image_path)
                    self.ui.fileListWidget.addItem(file_item)
                    

                    # 创建并添加缩略图的部分修改为：
                    thumbnail_item = QtWidgets.QListWidgetItem()
                    thumbnail_item.setData(QtCore.Qt.UserRole, image_path)
                    # 创建缩略图
                    thumbnail_icon = self.create_thumbnail(image_path)
                    if thumbnail_icon:
                        thumbnail_item.setIcon(thumbnail_icon)
                        thumbnail_item.setText(os.path.basename(image_path))
                        # 设置项目大小提示
                        size = self.thumbnail_delegate.sizeHint(
                            self.ui.thumbnailPreview.viewOptions(),
                            self.ui.thumbnailPreview.model().createIndex(0, 0)
                        )
                        thumbnail_item.setSizeHint(size)
                        self.ui.thumbnailPreview.addItem(thumbnail_item)
                
                # 设置当前索引为0并显示第一张图片
                self.current_image_index = 0
                self.display_current_image()
                # 选中第一个列表项
                self.ui.fileListWidget.setCurrentRow(0)
                self.ui.thumbnailPreview.setCurrentRow(0)
                # 更新按钮状态
                self.update_navigation_buttons()
            else:
                print("未在选择的目录中找到图片文件")

    def create_thumbnail(self, image_path):
        """创建图片缩略图"""
        # 加载图片
        image = QtGui.QImage(image_path)
        if image.isNull():
            return None
            
        # 创建缩略图（保持宽高比）
        thumbnail_size = QtCore.QSize(100, 100)
        scaled_image = image.scaled(thumbnail_size, 
                                  QtCore.Qt.KeepAspectRatio, 
                                  QtCore.Qt.SmoothTransformation)
        
        return QtGui.QIcon(QtGui.QPixmap.fromImage(scaled_image))
    def on_file_item_clicked(self, item):
        """处理文件列表项被点击的事件"""
        # 获取被点击项目的完整路径
        clicked_path = item.data(QtCore.Qt.UserRole)
        # 找到该路径在图片列表中的索引
        try:
            self.current_image_index = self.image_files.index(clicked_path)
            self.display_current_image()
            self.update_navigation_buttons()
        except ValueError:
            print("无法找到选中的图片")

    def on_thumbnail_clicked(self, item):
        """处理缩略图被点击的事件"""
        # 获取被点击项目的完整路径
        clicked_path = item.data(QtCore.Qt.UserRole)
        try:
            self.current_image_index = self.image_files.index(clicked_path)
            self.display_current_image()
            # 同步选中文件列表
            self.ui.fileListWidget.setCurrentRow(self.current_image_index)
            self.update_navigation_buttons()
        except ValueError:
            print("无法找到选中的图片")
            

    

    # 修改 display_current_image 方法：
    def display_current_image(self):
        """显示当前索引对应的图片并加载其标注"""
        if 0 <= self.current_image_index < len(self.image_files):
            current_image = self.image_files[self.current_image_index]
            
            # 清除现有的场景内容
            self.scene.clear()
            self.rect_items.clear()
            self.ui.categoryListWidget.clear()
            
            # 加载图片
            pixmap = QPixmap(current_image)
            view_size = self.ui.graphicsView.size()
            scaled_pixmap = pixmap.scaled(
                view_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # 将图片添加到场景中
            pixmap_item = self.scene.addPixmap(scaled_pixmap)
            self.scene.setSceneRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height())
            self.image_bounds = self.scene.sceneRect()
            
            # 加载已有的标注
            annotations = self.annotation_storage.load_annotation(current_image)
            for annotation in annotations:
                rect_item = ResizableRectItem(QRectF(
                    annotation['x'],
                    annotation['y'],
                    annotation['width'],
                    annotation['height']
                ))
                rect_item.category = annotation['category']
                rect_item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
                rect_item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
                rect_item.main_window = self
                self.scene.addItem(rect_item)
                self.rect_items.append(rect_item)
                # 更新矩形框上的标签
                self.update_rect_label(rect_item, rect_item.category)
            
            # 更新类别列表显示
            self.update_category_list()
            
            # 设置场景到GraphicsView
            self.ui.graphicsView.setScene(self.scene)
            self.ui.graphicsView.fitInView(
                self.scene.sceneRect(),
                Qt.KeepAspectRatio
            )

    def next_image(self):
        """切换到下一张图片"""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.display_current_image()
            self.update_navigation_buttons()

    def previous_image(self):
        """切换到上一张图片"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """更新导航按钮的启用状态"""
        # 当没有上一张图片时禁用上一张按钮
        self.ui.pushButtonPrevImage.setEnabled(self.current_image_index > 0)
        # 当没有下一张图片时禁用下一张按钮
        self.ui.pushButtonNextImage.setEnabled(
            self.current_image_index < len(self.image_files) - 1
        )



    def update_category_list(self):
        """更新类别列表显示，包含区域缩略图"""
        # 防止频繁更新导致的闪烁
        self.ui.categoryListWidget.setUpdatesEnabled(False)
        
        try:
            self.ui.categoryListWidget.clear()
            
            # 获取当前场景中的图片项
            pixmap_items = [item for item in self.scene.items() if isinstance(item, QtWidgets.QGraphicsPixmapItem)]
            if not pixmap_items:
                return
            
            # 获取原始图片
            original_pixmap = pixmap_items[0].pixmap()
            
            for rect_item in self.rect_items:
                if hasattr(rect_item, 'category'):
                    # 创建列表项
                    item = QtWidgets.QListWidgetItem()
                    item.setSizeHint(QtCore.QSize(200, 120))
                    
                    # 获取矩形框的坐标信息
                    rect = rect_item.rect()
                    scene_pos = rect_item.scenePos()
                    actual_rect = QRectF(
                        rect.x() + scene_pos.x(),
                        rect.y() + scene_pos.y(),
                        rect.width(),
                        rect.height()
                    )
                    
                    # 创建缩略图
                    region_pixmap = self.create_region_thumbnail(original_pixmap, actual_rect)
                    
                    # 创建自定义widget来显示信息和缩略图
                    widget = QtWidgets.QWidget()
                    layout = QtWidgets.QHBoxLayout()
                    
                    # 添加缩略图标签
                    thumbnail_label = QtWidgets.QLabel()
                    thumbnail_label.setFixedSize(100, 100)
                    thumbnail_label.setPixmap(region_pixmap.scaled(
                        100, 100,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    ))
                    thumbnail_label.setStyleSheet("border: 1px solid #cccccc;")
                    layout.addWidget(thumbnail_label)
                    
                    # 添加文本信息
                    text_widget = QtWidgets.QWidget()
                    text_layout = QtWidgets.QVBoxLayout()
                    
                    category_label = QtWidgets.QLabel(f"类别: {rect_item.category}")
                    position_label = QtWidgets.QLabel(
                        f"位置: ({int(actual_rect.x())}, {int(actual_rect.y())})"
                    )
                    size_label = QtWidgets.QLabel(
                        f"大小: {int(actual_rect.width())}×{int(actual_rect.height())}"
                    )
                    
                    # 设置字体
                    font = QtGui.QFont()
                    font.setPointSize(9)
                    category_label.setFont(font)
                    position_label.setFont(font)
                    size_label.setFont(font)
                    
                    text_layout.addWidget(category_label)
                    text_layout.addWidget(position_label)
                    text_layout.addWidget(size_label)
                    text_layout.addStretch()
                    
                    text_widget.setLayout(text_layout)
                    layout.addWidget(text_widget)
                    
                    # 设置布局
                    widget.setLayout(layout)
                    
                    # 存储对应的矩形项引用
                    item.setData(QtCore.Qt.UserRole, rect_item)
                    
                    # 将自定义widget设置为列表项的widget
                    self.ui.categoryListWidget.addItem(item)
                    self.ui.categoryListWidget.setItemWidget(item, widget)
                    
        finally:
            # 重新启用更新
            self.ui.categoryListWidget.setUpdatesEnabled(True)


    def create_region_thumbnail(self, original_pixmap, rect):
        """从原始图片中截取矩形区域创建缩略图"""
        # 确保矩形区域在有效范围内
        rect = rect.toRect()
        rect = rect.intersected(original_pixmap.rect())
        
        # 从原始图片中截取区域
        cropped_pixmap = original_pixmap.copy(rect)
        
        # 添加边框
        painter = QtGui.QPainter(cropped_pixmap)
        pen = QtGui.QPen(Qt.red, 2)
        painter.setPen(pen)
        painter.drawRect(cropped_pixmap.rect().adjusted(0, 0, -1, -1))
        painter.end()
        
        return cropped_pixmap


    # 添加保存标注的方法：
    def save_current_annotations(self):
        """保存当前图片的标注信息"""
        if 0 <= self.current_image_index < len(self.image_files):
            current_image = self.image_files[self.current_image_index]
            self.annotation_storage.save_annotation(current_image, self.rect_items)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    # 获取屏幕的DPI信息并设置
    screen = app.primaryScreen()
    dpi = screen.physicalDotsPerInch()
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 启用高DPI缩放
    app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)    # 使用高DPI图标
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())