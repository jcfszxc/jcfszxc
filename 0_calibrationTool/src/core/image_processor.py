#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/14 18:29
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : image_processor.py
# @Description   : 该模块提供图像处理功能，主要用于在图像上绘制和显示标定特征点


from typing import List, Tuple, Optional
import numpy as np
import cv2
from config.settings import ImageSettings

class ImageProcessor:
    """图像处理核心类"""
    
    @staticmethod
    def draw_points(
        image: np.ndarray,
        points: List[Tuple[float, float]],
        highlight_index: int = -1
    ) -> np.ndarray:
        """在图像上绘制特征点
        
        Args:
            image: 输入图像
            points: 特征点列表
            highlight_index: 高亮显示的点的索引
            
        Returns:
            绘制了特征点的图像
        """
        img_copy = image.copy()
        settings = ImageSettings()
        
        for i, point in enumerate(points):
            x, y = int(point[0]), int(point[1])
            
            if i == highlight_index:
                color = settings.HIGHLIGHT_POINT_COLOR
                radius = settings.HIGHLIGHT_POINT_RADIUS
                thickness = 2
            else:
                color = settings.NORMAL_POINT_COLOR
                radius = settings.NORMAL_POINT_RADIUS
                thickness = -1
                
            cv2.circle(img_copy, (x, y), radius, color, thickness)
            cv2.putText(
                img_copy,
                str(i+1),
                (x+10, y+10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
            
        return img_copy