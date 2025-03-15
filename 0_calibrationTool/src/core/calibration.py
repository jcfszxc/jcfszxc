#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/15 18:21
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : calibration.py
# @Description   : 该模块实现图像标定处理核心功能，包括单应性矩阵计算、矩阵格式化和变换验证



import numpy as np
import cv2
from typing import List, Tuple

class CalibrationProcessor:
    """处理图像标定和矩阵计算的类"""
    
    @staticmethod
    def calculate_homography_matrix(
        src_points: List[Tuple[float, float]],
        dst_points: List[Tuple[float, float]]
    ) -> np.ndarray:
        # 默认返回单位矩阵作为标准H（恒等变换）
        default_H = np.eye(3, dtype=np.float32)
        
        if len(src_points) < 4 or len(src_points) != len(dst_points):
            return default_H
            
        src_points_arr = np.float32(src_points)
        dst_points_arr = np.float32(dst_points)
        
        # 添加最小内点比例参数
        H, mask = cv2.findHomography(
            src_points_arr,
            dst_points_arr,
            cv2.RANSAC,
            ransacReprojThreshold=5.0,
            confidence=0.95,  # 提高置信度
        )
        
        # 检查H是否为None
        if H is None:
            return default_H
            
        # 检查内点比例
        if mask is not None:
            inlier_ratio = np.sum(mask) / len(mask)
            if inlier_ratio < 0.6:  # 如果内点比例太低，认为结果不可靠
                return default_H
        
        return H
        
    @staticmethod
    def format_matrix(matrix: np.ndarray) -> dict:
        """将矩阵格式化为指定的JSON格式
        
        Args:
            matrix: 3x3的单应性矩阵
            
        Returns:
            格式化后的矩阵字典
        """
        # 将矩阵转换为指定格式的列表
        formatted_matrix = [
            [float(value) for value in row]
            for row in matrix
        ]
        
        return {
            "registration_h": formatted_matrix
        }
        

    @staticmethod
    def validate_transformation(
        H: np.ndarray,
        src_points: List[Tuple[float, float]],
        dst_points: List[Tuple[float, float]],
        threshold: float = 8.0
    ) -> Tuple[bool, float]:
        if not src_points or not dst_points:
            return False, float('inf')
            
        src_points = np.float32(src_points)
        dst_points = np.float32(dst_points)
        
        src_points_homogeneous = np.hstack(
            (src_points, np.ones((len(src_points), 1)))
        )
        transformed_points = np.dot(H, src_points_homogeneous.T).T
        
        # 添加数值稳定性检查
        z_coordinates = transformed_points[:, 2:]
        valid_points = np.abs(z_coordinates) > 1e-10
        
        if not np.all(valid_points):
            return False, float('inf')
        
        # 使用更稳定的归一化方法
        transformed_points_normalized = np.zeros_like(transformed_points)
        transformed_points_normalized[valid_points.flatten(), :2] = (
            transformed_points[valid_points.flatten(), :2] / 
            transformed_points[valid_points.flatten(), 2:]
        )
        
        # 计算误差并去除异常值
        errors = np.linalg.norm(
            transformed_points_normalized[:, :2] - dst_points, 
            axis=1
        )
        
        # 使用中位数绝对偏差(MAD)来检测和过滤异常值
        mad = np.median(np.abs(errors - np.median(errors)))
        modified_z_scores = 0.6745 * (errors - np.median(errors)) / (mad + 1e-10)
        valid_errors = errors[np.abs(modified_z_scores) < 3.5]
        
        if len(valid_errors) < len(errors) * 0.7:  # 如果太多点被识别为异常值
            return False, float('inf')
            
        max_error = np.max(valid_errors)
        return max_error <= threshold, max_error