#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/15 
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : feature_matching.py
# @Description   : 特征匹配和自动标定算法模块

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

class FeatureMatcher:
    """特征匹配器：自动检测和匹配图像特征点"""

    @staticmethod
    def calculate_matrix(visible_image: np.ndarray, ir_image: np.ndarray) -> Tuple[Optional[np.ndarray], List[Tuple[float, float]], List[Tuple[float, float]], Optional[str]]:
        """
        自动计算红外到可见光的映射矩阵，使用特征检测和匹配替代手动标定点
        
        Args:
            visible_image: 可见光图像
            ir_image: 红外图像
            
        Returns:
            Tuple 包含:
                - 计算出的单应性矩阵（如果成功）
                - 可见光图像特征点列表
                - 红外图像特征点列表
                - 错误信息（如果有）
        """
        # 初始化返回值
        homography_matrix = None
        visible_points = []
        ir_points = []
        error_message = None
        
        try:
            # 转换图像为灰度，以便特征检测
            visible_gray = cv2.cvtColor(visible_image, cv2.COLOR_RGB2GRAY)
            ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_RGB2GRAY)

            # 尝试使用SIFT特征检测器，如果不可用则回退到ORB
            try:
                detector = cv2.SIFT_create()
            except AttributeError:
                try:
                    detector = cv2.xfeatures2d.SIFT_create()
                except AttributeError:
                    detector = cv2.ORB_create(nfeatures=1000)

            # 在两个图像上检测关键点和描述符
            visible_keypoints, visible_descriptors = detector.detectAndCompute(visible_gray, None)
            ir_keypoints, ir_descriptors = detector.detectAndCompute(ir_gray, None)

            if visible_descriptors is None or ir_descriptors is None or len(visible_keypoints) < 10 or len(ir_keypoints) < 10:
                return None, [], [], "无法在图像中检测到足够的特征点，请尝试调整图像或手动标定。"

            # 根据特征检测器类型选择合适的匹配器
            if isinstance(detector, cv2.ORB):
                # 对于ORB特征使用暴力匹配器
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(ir_descriptors, visible_descriptors)

                # 根据距离对匹配进行排序
                matches = sorted(matches, key=lambda x: x.distance)

                # 选择前N个最佳匹配
                good_matches = matches[:min(50, len(matches))]
            else:
                # 对于SIFT特征使用FLANN匹配器
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)

                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                matches = matcher.knnMatch(ir_descriptors, visible_descriptors, k=2)

                # 应用Lowe's比率测试筛选好的匹配
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            # 确保有足够的好匹配点
            if len(good_matches) < 4:
                return None, [], [], f"找到的良好匹配点数量不足，只有{len(good_matches)}个，需要至少4个。"

            # 获取匹配点的坐标
            ir_points = []
            visible_points = []

            for match in good_matches:
                ir_points.append((ir_keypoints[match.queryIdx].pt[0], ir_keypoints[match.queryIdx].pt[1]))
                visible_points.append((visible_keypoints[match.trainIdx].pt[0], visible_keypoints[match.trainIdx].pt[1]))

            # 使用RANSAC筛选内点
            src_points = np.float32([point for point in ir_points]).reshape(-1, 1, 2)
            dst_points = np.float32([point for point in visible_points]).reshape(-1, 1, 2)

            # 使用RANSAC计算变换矩阵
            homography_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

            if homography_matrix is None:
                return None, [], [], "无法计算有效的变换矩阵，请尝试手动标定。"

            # 筛选内点
            inlier_mask = mask.ravel() == 1
            inlier_ir_points = [ir_points[i] for i in range(len(ir_points)) if inlier_mask[i]]
            inlier_visible_points = [visible_points[i] for i in range(len(visible_points)) if inlier_mask[i]]

            return homography_matrix, inlier_visible_points, inlier_ir_points, None

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return None, [], [], f"自动特征匹配过程中出错：{str(e)}\n\n详细信息：{error_details}"

    @staticmethod
    def draw_matches(visible_image: np.ndarray, ir_image: np.ndarray, 
                    visible_points: List[Tuple[float, float]], 
                    ir_points: List[Tuple[float, float]]) -> np.ndarray:
        """
        绘制两张图片的特征点匹配结果
        
        Args:
            visible_image: 可见光图像
            ir_image: 红外图像
            visible_points: 可见光图像特征点
            ir_points: 红外图像特征点
            
        Returns:
            带有匹配线的拼接图像
        """
        # 确保两组点的数量相同
        assert len(visible_points) == len(ir_points), "特征点数量不匹配"
        
        # 创建匹配点的可视化图像
        h1, w1 = visible_image.shape[:2]
        h2, w2 = ir_image.shape[:2]
        
        # 创建拼接图像
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = visible_image
        vis[:h2, w1:w1+w2] = ir_image
        
        # 绘制匹配线
        for i in range(len(visible_points)):
            pt1 = (int(visible_points[i][0]), int(visible_points[i][1]))
            pt2 = (int(ir_points[i][0]) + w1, int(ir_points[i][1]))
            
            # 随机颜色
            color = np.random.randint(0, 255, 3).tolist()
            
            # 画线
            cv2.line(vis, pt1, pt2, color, 1)
            
            # 画点
            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)
            
            # 添加索引标签
            cv2.putText(vis, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(vis, str(i), pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis
