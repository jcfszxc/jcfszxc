# sam-finetune.py
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class StampDataset(Dataset):
    def __init__(self, image_dir, mask_dir, bbox_file):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = ResizeLongestSide(1024)  # SAM default size
        
        # Load bbox annotations
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
        
        # Load image
        image = cv2.imread(os.path.join(self.image_dir, ann['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = ann['image'].replace('.jpg', '_mask.png')
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
        
        # Prepare image
        original_size = image.shape[:2]
        input_image = self.transform.apply_image(image)
        
        # Convert to float32 and normalize to 0-1 range
        input_image = input_image.astype(np.float32) / 255.0
        
        # Convert to tensor and normalize according to ImageNet stats
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).contiguous()
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        input_image = (input_image - mean) / std
        
        # Prepare bbox
        bbox = self.transform.apply_boxes(np.array([ann['bbox']]), original_size)[0]
        bbox_torch = torch.tensor(bbox, dtype=torch.float).unsqueeze(0)
        
        # Prepare mask
        mask_torch = torch.from_numpy(mask).float().unsqueeze(0)
        
        return {
            'image': input_image.float(),  # ensure float tensor
            'original_size': original_size,
            'bbox': bbox_torch,
            'mask': mask_torch
        }

def train_sam(
    model_type='vit_b',
    checkpoint_path='sam_vit_b_01ec64.pth',
    image_dir='./stamps/images',
    mask_dir='./stamps/masks',
    bbox_file='./stamps/annotations.txt',
    output_dir='./checkpoints',
    num_epochs=10,
    batch_size=1,
    learning_rate=1e-5
):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam_model.to(device)
    
    # Prepare dataset
    dataset = StampDataset(image_dir, mask_dir, bbox_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=learning_rate)
    
    # Loss function
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # Move inputs to device
            input_image = batch['image'].to(device)
            original_size = batch['original_size']
            bbox = batch['bbox'].to(device)
            gt_mask = batch['mask'].to(device)
            
            # Print shapes and types for debugging
            if batch_idx == 0 and epoch == 0:
                print(f"Input image shape: {input_image.shape}")
                print(f"Input image type: {input_image.dtype}")
                print(f"Input image range: [{input_image.min():.2f}, {input_image.max():.2f}]")
            
            # Get image embedding (without gradient)
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)
                
                # Get prompt embeddings
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=bbox,
                    masks=None,
                )
            
            # Generate mask prediction
            mask_predictions, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # Upscale masks to original size
            upscaled_masks = sam_model.postprocess_masks(
                mask_predictions,
                input_size=input_image.shape[-2:],
                original_size=original_size[0]
            ).to(device)
            
            # Convert to binary mask
            binary_masks = torch.sigmoid(upscaled_masks)
            
            # Calculate loss
            loss = loss_fn(binary_masks, gt_mask)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_file = os.path.join(output_dir, f'sam_finetuned_epoch_{epoch+1}.pth')
            torch.save(sam_model.state_dict(), checkpoint_file)
            print(f'Checkpoint saved: {checkpoint_file}')
    
    # Save final model
    final_checkpoint = os.path.join(output_dir, 'sam_finetuned_final.pth')
    torch.save(sam_model.state_dict(), final_checkpoint)
    print(f'Final model saved to {final_checkpoint}')

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('./checkpoints', exist_ok=True)
    
    # Start training
    train_sam()
