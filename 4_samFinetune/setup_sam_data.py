# sam-data-setup.py
import os
import numpy as np
import cv2
from pathlib import Path

def create_project_structure():
    """创建项目所需的目录结构"""
    # 创建主目录
    directories = [
        './stamps/images',
        './stamps/masks',
        './checkpoints'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return directories

def create_sample_data(num_samples=5):
    """创建示例训练数据"""
    # 创建示例图像和掩码
    annotations = []
    
    for i in range(num_samples):
        # 创建示例图像 (500x500)
        image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        # 添加一个示例印章 (随机位置的圆形)
        center_x = np.random.randint(150, 350)
        center_y = np.random.randint(150, 350)
        radius = np.random.randint(50, 100)
        
        # 绘制印章
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 255), -1)
        
        # 创建对应的掩码
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # 保存图像和掩码
        image_path = f'./stamps/images/sample_{i}.jpg'
        mask_path = f'./stamps/masks/sample_{i}_mask.png'
        
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)
        
        # 计算边界框
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(500, center_x + radius)
        y2 = min(500, center_y + radius)
        
        # 添加到注释列表
        annotations.append(f'sample_{i}.jpg,{x1},{y1},{x2},{y2}\n')
    
    # 保存注释文件
    with open('./stamps/annotations.txt', 'w') as f:
        f.writelines(annotations)

def main():
    print("开始创建项目结构...")
    directories = create_project_structure()
    for dir_path in directories:
        print(f"创建目录: {dir_path}")
    
    print("\n创建示例训练数据...")
    create_sample_data()
    print("示例数据创建完成！")
    
    print("\n项目结构：")
    for root, dirs, files in os.walk('./stamps'):
        level = root.replace('./stamps', '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

if __name__ == '__main__':
    main()
