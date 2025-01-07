from typing import Tuple, Optional
import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel

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

def get_image_coordinates(
    event,
    label: QLabel,
    image: Optional[np.ndarray]
) -> Optional[Tuple[float, float]]:
    """获取鼠标点击在原始图像上的坐标
    
    Args:
        event: 鼠标事件
        label: 图像标签
        image: 原始图像
        
    Returns:
        图像坐标 (x, y) 或 None（如果点击在图像外）
    """
    if image is None:
        return None
        
    # 获取标签和图像的尺寸
    label_size = label.size()
    image_height, image_width = image.shape[:2]
        
    # 计算实际显示图像的尺寸（保持宽高比）
    scale = min(
        label_size.width() / image_width,
        label_size.height() / image_height
    )
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

def prepare_image_for_display(
    image: np.ndarray,
    label_size: Tuple[int, int]
) -> Tuple[np.ndarray, int, int, int]:
    """准备图像用于显示
    
    Args:
        image: 输入图像
        label_size: 标签尺寸 (width, height)
        
    Returns:
        (处理后的图像, 高度, 宽度, 通道数)
    """
    # 调整图像大小
    display_image = resize_image(image, label_size)
    
    # 获取图像属性
    height, width = display_image.shape[:2]
    channel = 3  # 假设输入为RGB图像
    
    return display_image, height, width, channel

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
    from ..core.image_processor import ImageProcessor  # 避免循环导入
    
    # 先绘制点
    image_with_points = ImageProcessor.draw_points(
        image,
        points,
        highlight_index
    )
    
    # 调整大小
    return resize_image(image_with_points, label_size)