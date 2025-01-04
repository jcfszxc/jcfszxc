import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import cv2
from pathlib import Path

class SAMPredictor:
    def __init__(self, checkpoint_path, model_type="vit_b", device="cuda"):
        """
        初始化SAM预测器
        Args:
            checkpoint_path: 模型权重路径
            model_type: 模型类型 ("vit_h", "vit_l", "vit_b")
            device: 使用设备 ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam_model.to(self.device)
        
        # 创建图像变换器
        self.transform = ResizeLongestSide(1024)
    
    def resize_bbox(self, bbox, original_size, target_size=(1024, 1024)):
        """
        调整边界框坐标以匹配调整大小后的图像
        Args:
            bbox: 原始边界框坐标 [x1, y1, x2, y2]
            original_size: 原始图像尺寸 (height, width)
            target_size: 目标图像尺寸 (height, width)
        Returns:
            resized_bbox: 调整后的边界框坐标
        """
        orig_h, orig_w = original_size
        target_h, target_w = target_size
        
        # 计算缩放比例
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # 调整边界框坐标
        x1, y1, x2, y2 = bbox
        resized_bbox = [
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        ]
        
        return resized_bbox
        
    def preprocess_image(self, image):
        """预处理输入图像"""
        # 保存原始尺寸
        original_size = image.shape[:2]
        
        # 确保图像是RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # 调整图像大小
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        
        # 转换为float32并归一化
        input_image = image.astype(np.float32) / 255.0
        
        # 转换为tensor并添加batch维度
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0)
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        input_image = (input_image - mean) / std
        
        return input_image.to(self.device), original_size, image
        
    def predict(self, image, bbox):
        """
        预测单个图像的分割掩码
        Args:
            image: numpy array 格式的图像
            bbox: [x1, y1, x2, y2] 格式的边界框
        Returns:
            binary_mask: 二值化的分割掩码
            confidence: 预测的置信度
        """
        # 预处理图像
        input_image, original_size, resized_image = self.preprocess_image(image)
        
        # 调整边界框大小
        resized_bbox = self.resize_bbox(bbox, original_size)
        print(resized_bbox, image.shape, resized_image.shape)
        
        # 准备边界框
        bbox_torch = torch.tensor(resized_bbox, dtype=torch.float, device=self.device).unsqueeze(0)
        
        # 获取图像嵌入
        with torch.no_grad():
            image_embedding = self.sam_model.image_encoder(input_image)
            
            # 获取提示嵌入
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=None,
                boxes=bbox_torch,
                masks=None,
            )
            
            # 生成掩码预测
            mask_predictions, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # 后处理掩码
            upscaled_masks = self.sam_model.postprocess_masks(
                mask_predictions,
                input_size=input_image.shape[-2:],
                original_size=original_size
            ).to(self.device)
            
            # 转换为二值掩码
            binary_mask = torch.sigmoid(upscaled_masks) > 0.5
            
        return binary_mask[0, 0].cpu().numpy(), iou_predictions[0, 0].item()

def visualize_prediction(image, mask, bbox, confidence, save_path=None):
    """
    可视化预测结果
    Args:
        image: 原始图像
        mask: 预测的掩码
        bbox: 边界框坐标
        confidence: 预测置信度
        save_path: 保存路径（可选）
    """
    # 创建图形
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    # 绘制边界框
    x1, y1, x2, y2 = map(int, bbox)
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
    plt.axis('off')
    
    # 显示预测掩码
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Predicted Mask\nConfidence: {confidence:.2f}')
    plt.axis('off')
    
    # 显示叠加结果
    plt.subplot(133)
    overlay = image.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([0, 255, 0], dtype=np.uint8) * 0.3
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"结果已保存到: {save_path}")
    
    plt.show()

def main():
    # 配置参数
    checkpoint_path = "./checkpoints/sam_finetuned_final.pth"  # 使用微调后的模型
    test_image_path = "./stamps/images/sample_0.jpg"
    output_dir = "./predictions"
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化预测器
    predictor = SAMPredictor(checkpoint_path)
    
    # 读取测试图像
    image = cv2.imread(test_image_path)
    # image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    # 读取边界框（这里使用示例边界框，实际应用中可能需要从标注文件读取）
    with open('./stamps/annotations.txt', 'r') as f:
        first_line = f.readline().strip()
        _, x1, y1, x2, y2 = first_line.split(',')
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        print(bbox)
    
    # 进行预测
    mask, confidence = predictor.predict(image, bbox)
    
    # 可视化结果
    save_path = str(Path(output_dir) / "prediction_result.png")
    visualize_prediction(image, mask, bbox, confidence, save_path)

if __name__ == "__main__":
    main()

