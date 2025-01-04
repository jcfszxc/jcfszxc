import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
import os
from tqdm import tqdm
import logging
import json
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StampDataset(Dataset):
    def __init__(self, image_dir, mask_dir, bbox_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform if transform else ResizeLongestSide(1024)
        
        # 加载标注文件
        self.annotations = []
        with open(bbox_file, 'r') as f:
            for line in f:
                img_name, x1, y1, x2, y2 = line.strip().split(',')
                self.annotations.append({
                    'image': img_name,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 读取图像
        image_path = os.path.join(self.image_dir, ann['image'])
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取mask
        mask_name = ann['image'].replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        mask = mask.astype(np.float32) / 255.0
        
        # 准备图像
        original_size = image.shape[:2]
        input_image = self.transform.apply_image(image)
        input_image = input_image.astype(np.float32) / 255.0
        
        # 转换为tensor并进行ImageNet归一化
        input_image = torch.from_numpy(input_image).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        input_image = (input_image - mean) / std
        
        # 准备bbox
        bbox = self.transform.apply_boxes(np.array([ann['bbox']]), original_size)[0]
        bbox_torch = torch.tensor(bbox, dtype=torch.float)
        
        # 准备mask
        mask_torch = torch.from_numpy(mask).float()
        
        return {
            'image': input_image,
            'original_size': original_size,
            'bbox': bbox_torch,
            'mask': mask_torch,
            'image_path': image_path  # 用于调试
        }

class SAMFineTuner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_datasets()
        self.setup_training()
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(config['output_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def setup_model(self):
        logger.info(f"Loading SAM model: {self.config['model_type']}")
        self.model = sam_model_registry[self.config['model_type']](
            checkpoint=self.config['checkpoint_path']
        )
        self.model.to(self.device)
    
    def setup_datasets(self):
        logger.info("Setting up datasets")
        self.train_dataset = StampDataset(
            self.config['train_image_dir'],
            self.config['train_mask_dir'],
            self.config['train_bbox_file']
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        if self.config.get('val_bbox_file'):
            self.val_dataset = StampDataset(
                self.config['val_image_dir'],
                self.config['val_mask_dir'],
                self.config['val_bbox_file']
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
    
    def setup_training(self):
        logger.info("Setting up training components")
        # 分别设置encoder和decoder的学习率
        self.optimizer = torch.optim.Adam([
            {'params': self.model.image_encoder.parameters(), 
             'lr': self.config['encoder_lr']},
            {'params': self.model.mask_decoder.parameters(), 
             'lr': self.config['decoder_lr']}
        ])
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.loss_fn = torch.nn.MSELoss()
        self.scaler = GradScaler()
        
        # 记录最佳模型
        self.best_loss = float('inf')
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}')
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到GPU
            input_image = batch['image'].to(self.device)
            bbox = batch['bbox'].to(self.device)
            gt_mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                # 前向传播
                image_embedding = self.model.image_encoder(input_image)
                
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None,
                        boxes=bbox,
                        masks=None,
                    )
                
                mask_predictions, _ = self.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                upscaled_masks = self.model.postprocess_masks(
                    mask_predictions,
                    input_size=input_image.shape[-2:],
                    original_size=batch['original_size'][0]
                ).to(self.device)
                
                binary_masks = torch.sigmoid(upscaled_masks)
                loss = self.loss_fn(binary_masks, gt_mask.unsqueeze(1))
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        if not hasattr(self, 'val_loader'):
            return None
            
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            input_image = batch['image'].to(self.device)
            bbox = batch['bbox'].to(self.device)
            gt_mask = batch['mask'].to(self.device)
            
            with autocast():
                image_embedding = self.model.image_encoder(input_image)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=bbox,
                    masks=None,
                )
                
                mask_predictions, _ = self.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                upscaled_masks = self.model.postprocess_masks(
                    mask_predictions,
                    input_size=input_image.shape[-2:],
                    original_size=batch['original_size'][0]
                ).to(self.device)
                
                binary_masks = torch.sigmoid(upscaled_masks)
                loss = self.loss_fn(binary_masks, gt_mask.unsqueeze(1))
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # 保存最新的checkpoint
        checkpoint_path = os.path.join(
            self.config['output_dir'],
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(self.config['output_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with loss: {loss:.4f}")
    
    def train(self):
        logger.info("Starting training")
        for epoch in range(self.config['num_epochs']):
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            
            # 验证
            val_loss = self.validate()
            if val_loss is not None:
                logger.info(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")
                self.scheduler.step(val_loss)
                
                # 检查是否是最佳模型
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
            else:
                is_best = False
                self.scheduler.step(train_loss)
            
            # 保存checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(
                    epoch,
                    val_loss if val_loss is not None else train_loss,
                    is_best
                )

def main():
    # 训练配置
    config = {
        'model_type': 'vit_b',
        'checkpoint_path': 'sam_vit_b_01ec64.pth',
        'train_image_dir': './data/train/images',
        'train_mask_dir': './data/train/masks',
        'train_bbox_file': './data/train/annotations.txt',
        'val_image_dir': './data/val/images',
        'val_mask_dir': './data/val/masks',
        'val_bbox_file': './data/val/annotations.txt',
        'output_dir': f'./outputs/sam_finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'num_epochs': 50,
        'batch_size': 4,
        'num_workers': 4,
        'encoder_lr': 1e-6,
        'decoder_lr': 1e-5,
        'save_interval': 5
    }
    
    # 创建训练器并开始训练
    trainer = SAMFineTuner(config)
    trainer.train()

if __name__ == '__main__':
    main()
