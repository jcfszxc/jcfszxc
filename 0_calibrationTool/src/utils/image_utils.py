#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/13 16:30
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : image_utils.py
# @Description   : 

from typing import Tuple

import numpy as np
import cv2

from core.image_processor import ImageProcessor  # 避免循环导入

## 读取图像，解决imread不能读取中文路径的问题    
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def convert_color_space(image: np.ndarray, to_rgb: bool = True) -> np.ndarray:
    """转换图像颜色空间
    
    Args:
        image: 输入图像
        to_rgb: True 转换为 RGB，False 转换为 BGR
        
    Returns:
        转换后的图像
    """
    if to_rgb:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def create_display_image(
    image: np.ndarray,
    points: list,
    highlight_index: int,
    label_size: Tuple[int, int]
) -> np.ndarray:
    """创建用于显示的图像
    
    Args:
        image: 原始图像
        points: 特征点列表
        highlight_index: 高亮点索引
        label_size: 标签尺寸
        
    Returns:
        处理后的图像
    """
    
    # 先绘制点
    image_with_points = ImageProcessor.draw_points(
        image,
        points,
        highlight_index
    )
    
    # 调整大小
    return resize_image(image_with_points, label_size)


def resize_image(
    image: np.ndarray,
    label_size: Tuple[int, int]
) -> np.ndarray:
    """调整图像大小以适应显示区域，保持宽高比
    
    Args:
        image: 输入图像
        label_size: 标签尺寸 (width, height)
        
    Returns:
        调整后的图像
    """
    # 创建固定大小的背景
    background = np.zeros(
        (label_size[1], label_size[0], 3),
        dtype=np.uint8
    )
    
    if image is None:
        return background
        
    # 获取图像尺寸
    height, width = image.shape[:2]
    target_width, target_height = label_size
    
    # 计算缩放比例
    scale = min(target_width / width, target_height / height)
    
    # 计算新的尺寸（确保是整数）
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 调整图像大小
    resized_image = cv2.resize(
        image, 
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )
    
    # 计算偏移量以居中放置图像
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # 将调整大小的图像放置在背景中心
    background[
        y_offset:y_offset+new_height,
        x_offset:x_offset+new_width
    ] = resized_image
    
    return background