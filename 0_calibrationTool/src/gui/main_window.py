#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/12 19:25
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : main_window.py
# @Description   : 该模块实现了标定工具的主界面，处理图像加载、特征点标记、矩阵计算、结果可视化和交互逻辑

from typing import Optional, List, Tuple

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QLabel,
    QScrollArea, QTextEdit
)

import numpy as np
import cv2

from config.settings import GUISettings
from gui.image_display_widget import ImageDisplayWidget
from utils.image_utils import convert_color_space, cv_imread, create_display_image

from PyQt5.QtCore import Qt, QPoint
from .widgets.point_pair_widget import PointPairWidget
from core.calibration import CalibrationProcessor
import json

from core.feature_matching import FeatureMatcher  # 导入新的特征匹配模块


class CalibrationGUI(QMainWindow):
    """标定工具主窗口"""

    def __init__(self):
        super().__init__()
        self._init_data()
        self._init_ui()
        self.showMaximized()

    def _init_data(self) -> None:
        """初始化数据"""
        self.visible_image: Optional[np.ndarray] = None
        self.ir_image: Optional[np.ndarray] = None
        self.selecting_visible: bool = True
        self.visible_points: List[Tuple[float, float]] = []
        self.ir_points: List[Tuple[float, float]] = []
        self.current_point_index: int = 0
        self.highlight_index: int = -1
        self.homography_matrix: Optional[np.ndarray] = None
        
        # 初始化完成后更新状态文本
        if hasattr(self, 'status_text'):
            self.update_status_text("请加载可见光和红外图像，然后标记至少 4 对特征点")

    def _init_ui(self) -> None:
        """初始化用户界面"""
        settings = GUISettings()
        self.setWindowTitle(settings.WINDOW_TITLE)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        button_layout = QHBoxLayout()

        self.btn_load_visible = QPushButton('加载可见光图像', self)
        self.btn_load_ir = QPushButton('加载红外图像', self)
        self.btn_clear_points = QPushButton('清除所有点', self)
        self.btn_undo = QPushButton('撤销上一个点', self)
        self.point_label = QLabel('当前点对序号: 0', self)
        # 添加各种按钮
        self.btn_calculate = QPushButton('自动计算映射矩阵', self)
        self.btn_save_matrix = QPushButton('保存矩阵', self)
        # self.btn_fuse = QPushButton('融合图像', self)

        self.btn_calculate.setEnabled(False)
        self.btn_save_matrix.setEnabled(False)
        # self.btn_fuse.setEnabled(False)

        button_layout.addWidget(self.btn_load_visible)
        button_layout.addWidget(self.btn_load_ir)
        button_layout.addWidget(self.btn_clear_points)
        button_layout.addWidget(self.btn_undo)
        button_layout.addWidget(self.point_label)

        button_layout.addWidget(self.btn_calculate)
        button_layout.addWidget(self.btn_save_matrix)
        # button_layout.addWidget(self.btn_fuse)

        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)

        top_image_layout = QHBoxLayout()

        self.visible_label = ImageDisplayWidget(
            "可见光图像",
            settings.MIN_IMAGE_SIZE,
            self.visible_click_event
        )
        self.ir_label = ImageDisplayWidget(
            "红外图像",
            settings.MIN_IMAGE_SIZE,
            self.ir_click_event
        )

        top_image_layout.addWidget(self.visible_label)
        top_image_layout.addWidget(self.ir_label)

        # 创建下排图像布局
        bottom_image_layout = QHBoxLayout()

        # 创建左下和右下图像显示组件
        self.bottom_left_label = ImageDisplayWidget(
            "放大镜区域",
            settings.MIN_IMAGE_SIZE,
            lambda x: None
        )
        self.fusion_label = ImageDisplayWidget(
            "融合结果",
            settings.MIN_IMAGE_SIZE,
            lambda x: None
        )

        # 添加下排图像显示组件，设置1:3的比例
        bottom_image_layout.addWidget(self.bottom_left_label, 1)
        bottom_image_layout.addWidget(self.fusion_label, 3)

        image_layout.addLayout(top_image_layout)
        image_layout.addLayout(bottom_image_layout)

        # 创建一个水平布局来容纳两个标题并排
        titles_layout = QHBoxLayout()

        # 创建特征点列表区域
        points_list_container = QWidget()
        points_list_layout = QVBoxLayout(points_list_container)

        # 创建标题标签
        points_list_title = QLabel("特征点对列表")
        points_list_title.setAlignment(Qt.AlignCenter)


        # 创建状态标签
        status_title = QLabel("状态参数")
        status_title.setAlignment(Qt.AlignCenter)
        
        # 创建矩阵显示标签
        matrix_title = QLabel("H矩阵")
        matrix_title.setAlignment(Qt.AlignCenter)

        # 将三个标题添加到水平布局中
        titles_layout.addWidget(points_list_title, 3)  # 分配更多空间给特征点列表
        titles_layout.addWidget(status_title, 2)       # 状态参数使用中等空间
        titles_layout.addWidget(matrix_title, 1)       # 矩阵显示使用较小空间

        # 创建一个水平布局来容纳列表和状态文本框以及矩阵显示
        content_layout = QHBoxLayout()

        # 继续创建特征点列表的滚动区域
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(settings.POINTS_LIST_HEIGHT[0])
        self.scroll.setMaximumHeight(settings.POINTS_LIST_HEIGHT[1])

        # 创建特征点列表容器
        self.points_list_widget = QWidget()
        self.points_list_layout = QHBoxLayout(self.points_list_widget)
        self.points_list_layout.setAlignment(Qt.AlignLeft)
        self.scroll.setWidget(self.points_list_widget)

        points_list_layout.addWidget(self.scroll)

        # 创建状态文本框
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(50)
        self.status_text.setMaximumHeight(100)
        self.status_text.setStyleSheet("background-color: #f0f0f0;")

        # 创建矩阵显示文本框
        self.matrix_text = QTextEdit()
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setMinimumHeight(50)
        self.matrix_text.setMaximumHeight(100)
        self.matrix_text.setStyleSheet("background-color: #f0f0f0; font-family: monospace;")
        self.matrix_text.setAlignment(Qt.AlignCenter)
        self.matrix_text.setText("等待计算...")

        # 将列表容器、状态文本框和矩阵显示添加到水平布局
        content_layout.addWidget(points_list_container, 3)  # 分配更多空间给特征点列表
        content_layout.addWidget(self.status_text, 2)       # 状态参数使用中等空间
        content_layout.addWidget(self.matrix_text, 1)       # 矩阵显示使用较小空间

        main_layout.addLayout(button_layout)
        main_layout.addWidget(image_container, stretch=1)

        # 将水平布局添加到主布局
        main_layout.addLayout(titles_layout)
        # 将水平内容布局添加到主布局
        main_layout.addLayout(content_layout)
        
        self.update_status_text("请加载可见光和红外图像，然后标记至少 4 对特征点")
        
        self._connect_signals()

    def _connect_signals(self) -> None:
        """连接信号和槽"""
        self.btn_load_visible.clicked.connect(
            lambda: self.load_image('visible'))
        self.btn_load_ir.clicked.connect(lambda: self.load_image('ir'))
        # self.btn_fuse.clicked.connect(self.fuse_images)

        self.btn_clear_points.clicked.connect(self.clear_points)
        self.btn_undo.clicked.connect(self.undo_last_point)

        self.btn_calculate.clicked.connect(self.calculate_matrix)
        self.btn_save_matrix.clicked.connect(self.save_matrix)

        # 连接放大镜信号
        self.visible_label.magnifier_moved.connect(self.update_magnifier)
        self.ir_label.magnifier_moved.connect(self.update_magnifier)
        self.fusion_label.magnifier_moved.connect(self.update_magnifier)
        self.bottom_left_label.magnifier_moved.connect(self.update_magnifier)

    def load_image(self, image_type: str) -> None:
        """加载图像

        Args:
            image_type: 'visible' 或 'ir'
        """
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            f"选择{'可见光' if image_type == 'visible' else '红外'}图像",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
        )

        if file_name:
            image = cv_imread(file_name)
            if image is None:
                return

            # 转换颜色空间
            image = convert_color_space(image, to_rgb=True)

            # 存储图像
            if image_type == 'visible':
                self.visible_image = image
            else:
                self.ir_image = image

            self.update_display()

            # # 检查是否两个图像都已加载，如是则启用自动匹配按钮
            # if self.visible_image is not None and self.ir_image is not None:
            #     self.btn_calculate.setEnabled(True)


            if self.visible_image is not None and self.ir_image is not None:
                # Both images are loaded, prompt user to add feature points
                self.btn_calculate.setEnabled(True)
                point_count = min(len(self.visible_points), len(self.ir_points))
                if point_count < 4:
                    points_needed = 4 - point_count
                    self.update_status_text(f"两张图像已加载，还需要 {points_needed} 对特征点才能计算映射矩阵（当前：{point_count}/4）")


                # 更新计算按钮状态并触发自动融合
                can_calculate = (len(self.visible_points) >= 4 and
                                len(self.visible_points) == len(self.ir_points))
                # self.btn_calculate.setEnabled(can_calculate)

                # 尝试自动计算和融合
                if can_calculate:
                    self.auto_calculate_and_fuse()



            else:
                # Only one image is loaded
                if self.visible_image is not None and self.ir_image is None:
                    self.update_status_text("可见光图像已加载，请加载红外图像")
                elif self.visible_image is None and self.ir_image is not None:
                    self.update_status_text("红外图像已加载，请加载可见光图像")


    def update_display(self) -> None:
        """更新图像显示"""
        if self.visible_image is not None:
            display_image = create_display_image(
                self.visible_image,
                self.visible_points,
                self.highlight_index,
                (self.visible_label.width(), self.visible_label.height())
            )
            self.visible_label.display_image(display_image)

        if self.ir_image is not None:
            display_image = create_display_image(
                self.ir_image,
                self.ir_points,                self.highlight_index,
                (self.ir_label.width(), self.ir_label.height())
            )
            self.ir_label.display_image(display_image)

        # # 更新点对序号显示
        # self.point_label.setText(f'当前点对序号: {self.current_point_index}')

    def resizeEvent(self, event) -> None:
        """重写窗口大小改变事件"""
        super().resizeEvent(event)
        self.update_display()


    def update_magnifier(self, source_label: QLabel, pos: QPoint, original_image: np.ndarray) -> None:
        if original_image is None:
            return

        # 获取图像在标签中的实际显示尺寸和位置
        label_size = source_label.size()
        image_height, image_width = original_image.shape[:2]
        scale = min(label_size.width() / image_width,
                    label_size.height() / image_height)
        display_width = int(image_width * scale)
        display_height = int(image_height * scale)

        # 计算图像在标签中的偏移量（居中显示）
        offset_x = (label_size.width() - display_width) // 2
        offset_y = (label_size.height() - display_height) // 2

        # 调整鼠标位置以考虑偏移
        x = pos.x() - offset_x
        y = pos.y() - offset_y

        # 检查鼠标是否在图像区域内
        if (0 <= x < display_width and 0 <= y < display_height):
            # 转换回原始图像坐标
            original_x = int((x / display_width) * image_width)
            original_y = int((y / display_height) * image_height)

            # 使用固定大小的放大区域
            magnifier_size = 120  # 放大区域的固定大小

            # 计算放大区域的边界
            start_x = max(0, original_x - magnifier_size // 8)
            start_y = max(0, original_y - magnifier_size // 8)
            end_x = min(image_width, original_x + magnifier_size // 8)
            end_y = min(image_height, original_y + magnifier_size // 8)

            # 提取要放大的区域
            magnifier_region = original_image[start_y:end_y, start_x:end_x]

            # 放大到固定尺寸
            magnified = cv2.resize(magnifier_region,
                                   (magnifier_size, magnifier_size),
                                   interpolation=cv2.INTER_NEAREST)

            # 添加十字线
            center_x = magnified.shape[1] // 2
            center_y = magnified.shape[0] // 2
            color = (255, 0, 0)  # 红色
            thickness = 1

            # 绘制十字线
            cv2.line(magnified, (center_x, 0), (center_x,
                     magnified.shape[0]), color, thickness)
            cv2.line(magnified, (0, center_y),
                     (magnified.shape[1], center_y), color, thickness)

            # 显示当前坐标
            coordinate_text = f"({original_x}, {original_y})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            cv2.putText(magnified, coordinate_text, (5, 15),
                        font, font_scale, color, thickness)

            # 显示放大的图像
            self.bottom_left_label.display_image(magnified)

    def visible_click_event(self, event) -> None:
        """处理可见光图像的点击事件"""
        if self.visible_image is None or not self.selecting_visible:
            return

        coords = self._get_image_coordinates(event, self.visible_label)
        if coords:
            self.visible_points.append(coords)
            self.selecting_visible = False
            self.update_display()
            self.update_points_list()

    def ir_click_event(self, event) -> None:
        """处理红外图像的点击事件"""
        if self.ir_image is None or self.selecting_visible:
            return

        coords = self._get_image_coordinates(event, self.ir_label)
        if coords:
            self.ir_points.append(coords)
            self.selecting_visible = True
            self.current_point_index += 1
            self.update_display()
            self.update_points_list()

    def _get_image_coordinates(self, event, label) -> Optional[Tuple[float, float]]:
        """获取鼠标点击在原始图像上的坐标"""
        # 获取标签的大小和图像的大小
        label_size = label.size()
        if label == self.visible_label and self.visible_image is not None:
            image = self.visible_image
        elif label == self.ir_label and self.ir_image is not None:
            image = self.ir_image
        else:
            return None

        image_height, image_width = image.shape[:2]

        # 计算实际显示图像的尺寸（保持宽高比）
        scale = min(label_size.width() / image_width,
                    label_size.height() / image_height)
        display_width = int(image_width * scale)
        display_height = int(image_height * scale)

        # 计算图像在标签中的偏移量（居中显示）
        offset_x = (label_size.width() - display_width) // 2
        offset_y = (label_size.height() - display_height) // 2

        # 获取点击位置并调整偏移
        pos = event.pos()
        x = pos.x() - offset_x
        y = pos.y() - offset_y

        # 检查点击是否在图像范围内
        if x < 0 or x >= display_width or y < 0 or y >= display_height:
            return None

        # 转换回原始图像坐标
        original_x = (x / display_width) * image_width
        original_y = (y / display_height) * image_height

        return (original_x, original_y)


    def update_points_list(self) -> None:
        """更新特征点对列表并触发自动融合"""
        # 清除现有的点对显示
        for i in reversed(range(self.points_list_layout.count())):
            self.points_list_layout.itemAt(i).widget().setParent(None)

        # 添加新的点对显示
        for i in range(min(len(self.visible_points), len(self.ir_points))):
            point_widget = PointPairWidget(
                i,
                self.visible_points[i],
                self.ir_points[i],
                self.highlight_point_pair,
                self.delete_point_pair  # 添加删除点对的处理函数
            )
            if i == self.highlight_index:
                point_widget.setProperty("selected", True)
                point_widget.style().unpolish(point_widget)
                point_widget.style().polish(point_widget)
            self.points_list_layout.addWidget(point_widget)

        # 检查特征点对数量并更新状态
        current_pairs = min(len(self.visible_points), len(self.ir_points))
        if current_pairs < 4:
            pairs_needed = 4 - current_pairs
            message = f"还需要 {pairs_needed} 对特征点才能计算映射矩阵（当前：{current_pairs}/4）"
            self.update_status_text(message)
        
        # 更新计算按钮状态并触发自动融合
        can_calculate = (len(self.visible_points) >= 4 and
                        len(self.visible_points) == len(self.ir_points))
        # self.btn_calculate.setEnabled(can_calculate)

        # 尝试自动计算和融合
        if can_calculate:
            self.auto_calculate_and_fuse()

    # 添加删除点对的方法
    def delete_point_pair(self, index: int) -> None:
        """删除指定索引的特征点对"""
        if 0 <= index < min(len(self.visible_points), len(self.ir_points)):
            # 删除选中的点对
            self.visible_points.pop(index)
            self.ir_points.pop(index)
            
            # 如果当前选中的是被删除的点，重置高亮
            if self.highlight_index == index:
                self.highlight_index = -1
            # 如果高亮的点在被删除点之后，需要调整高亮索引
            elif self.highlight_index > index:
                self.highlight_index -= 1
                
            # 调整当前点索引
            if self.current_point_index > 0:
                self.current_point_index = min(self.current_point_index - 1, len(self.visible_points))
            
            # 保持选择可见光图像的状态（如果已全部删除）
            if not self.visible_points:
                self.selecting_visible = True
                
            # 更新显示
            self.update_display()
            self.update_points_list()
            
            # 更新状态文本
            if len(self.visible_points) < 4:
                remaining = 4 - len(self.visible_points)
                self.update_status_text(f"已删除特征点对，还需要添加 {remaining} 对特征点才能计算映射矩阵")

    def highlight_point_pair(self, index: int) -> None:
        """高亮显示选中的特征点对"""
        self.highlight_index = index if self.highlight_index != index else -1
        self.update_display()
        self.update_points_list()

    # 修改clear_points方法，清除矩阵显示
    def clear_points(self) -> None:
        """清除所有特征点"""
        self.visible_points.clear()
        self.ir_points.clear()
        self.current_point_index = 0
        self.selecting_visible = True
        self.highlight_index = -1
        self.homography_matrix = None  # 清除现有的映射矩阵
        
        # 清除矩阵显示
        self.matrix_text.setText("等待计算...")
        
        self.update_display()
        self.update_points_list()

        # 清除融合结果显示
        self.fusion_label.clear()
        self.btn_save_matrix.setEnabled(False)

        # 更新状态文本，提示需要添加特征点
        self.update_status_text("已清除所有点，需要添加 4 对特征点才能计算映射矩阵")

    def undo_last_point(self) -> None:
        """撤销上一个点"""
        if self.selecting_visible:
            if self.ir_points:
                self.ir_points.pop()
                self.selecting_visible = False
                self.current_point_index -= 1
        else:
            if self.visible_points:
                self.visible_points.pop()
                self.selecting_visible = True
        self.highlight_index = -1
        self.update_display()
        self.update_points_list()  # 这里会触发自动融合

    # 修改calculate_matrix方法，添加更新矩阵显示的调用
    def calculate_matrix(self) -> None:
        """自动计算红外到可见光的映射矩阵，使用特征检测和匹配替代手动标定点"""
        if self.visible_image is None or self.ir_image is None:
            QMessageBox.warning(
                self,
                "警告",
                "请先加载可见光和红外图像。"
            )
            return

        # 调用特征匹配模块计算映射矩阵和匹配点
        homography_matrix, visible_points, ir_points, error_message = FeatureMatcher.calculate_matrix(
            self.visible_image, 
            self.ir_image
        )

        # 如果有错误信息，显示错误并返回
        if error_message:
            QMessageBox.warning(self, "警告", error_message)
            return
            
        # 如果成功计算出矩阵
        if homography_matrix is not None:
            # 更新GUI中的点
            self.clear_points()  # 清除现有的点
            self.visible_points = visible_points
            self.ir_points = ir_points
            self.current_point_index = len(ir_points)
            self.homography_matrix = homography_matrix

            # 更新矩阵显示
            self.update_matrix_display()

            # 验证变换结果
            is_valid, max_error = CalibrationProcessor.validate_transformation(
                self.homography_matrix,
                self.ir_points,
                self.visible_points
            )

            # 自动计算并融合图像
            self.auto_calculate_and_fuse()

            # 更新显示
            self.update_display()
            self.update_points_list()

    def save_matrix(self) -> None:
        """保存计算得到的映射矩阵"""
        if self.homography_matrix is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "保存映射矩阵",
            "",
            "JSON文件 (*.json)"
        )

        if file_name:
            # 格式化矩阵
            matrix_dict = CalibrationProcessor.format_matrix(
                self.homography_matrix)

            # 保存到文件
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(matrix_dict, f, indent=2)
                QMessageBox.information(
                    self,
                    "成功",
                    "映射矩阵已成功保存！"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "错误",
                    f"保存文件时出错：{str(e)}"
                )

    # 修改auto_calculate_and_fuse方法，添加更新矩阵显示的调用
    def auto_calculate_and_fuse(self) -> None:
        """自动计算映射矩阵并融合图像"""
        # 检查是否满足计算条件
        if (len(self.visible_points) < 4 or
            len(self.visible_points) != len(self.ir_points) or
            self.visible_image is None or
                self.ir_image is None):
            return

        # 计算单应性矩阵
        self.homography_matrix = CalibrationProcessor.calculate_homography_matrix(
            self.ir_points,  # 源图像点（红外）
            self.visible_points  # 目标图像点（可见光）
        )

        if self.homography_matrix is None:
            return

        # 验证变换结果
        is_valid, max_error = CalibrationProcessor.validate_transformation(
            self.homography_matrix,
            self.ir_points,
            self.visible_points
        )

        # 更新矩阵显示
        self.update_matrix_display()

        if is_valid:
            self.update_status_text(f"映射矩阵计算成功，最大误差: {max_error:.2f} 像素")
        else:
            self.update_status_text(f"映射矩阵计算完成，但精度可能不足，最大误差: {max_error:.2f} 像素")

        self.btn_save_matrix.setEnabled(True)
        
        # 执行图像融合
        try:
            # 将红外图像对齐到可见光图像
            aligned_ir = cv2.warpPerspective(
                self.ir_image,
                self.homography_matrix,
                (self.visible_image.shape[1], self.visible_image.shape[0])
            )

            # 将对齐后的红外图像转换为灰度图像
            aligned_ir_gray = cv2.cvtColor(aligned_ir, cv2.COLOR_RGB2GRAY)

            # 应用伪彩色映射
            colormap = cv2.applyColorMap(aligned_ir_gray, cv2.COLORMAP_JET)

            # 将colormap从BGR转换为RGB
            colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

            # 融合图像
            fused_image = cv2.addWeighted(
                self.visible_image, 0.3, colormap, 0.7, 0)

            # 显示融合结果
            display_image = create_display_image(
                fused_image,
                [],  # 无需显示点
                -1,  # 无需高亮
                (self.fusion_label.width(), self.fusion_label.height())
            )
            self.fusion_label.display_image(display_image)
        except Exception:
            # 静默处理异常，因为这是自动过程
            pass


    def update_status_text(self, text):
        """更新状态文本框的内容"""
        self.status_text.setText(text)

        
    # 添加一个新方法用于更新矩阵显示
    def update_matrix_display(self) -> None:
        """更新H矩阵显示"""
        if self.homography_matrix is None:
            self.matrix_text.setText("等待计算...")
            return
            
        # 格式化矩阵以漂亮地显示
        matrix_text = "单应性矩阵 H:\n"
        for i in range(self.homography_matrix.shape[0]):
            row = []
            for j in range(self.homography_matrix.shape[1]):
                value = self.homography_matrix[i, j]
                # 使用固定格式显示矩阵值，保留4位小数
                row.append(f"{value:.4f}")
            matrix_text += " ".join(row) + "\n"
        
        self.matrix_text.setText(matrix_text)
