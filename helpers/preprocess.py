# Define the Enhanced PyTorch Dataset class
import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataAugmentor():
    """Dataset for cropped ROI mammography images with proper mask handling"""

    

    def __init__(self, dataframe, transforms=None):
        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row["cropped_image_path"]
        mask_path = row["roi_mask_path"]
        mask_status = row["mask_status"]
        pathology = row["pathology"]  
        
        # Determine classification label (0=BENIGN, 1=MALIGNANT)
        label = 1 if 'MALIGNANT' in str(pathology).upper() else 0  
        
        # Determine if mask is available
        has_mask = (mask_status == 'valid')  
        
        # Read cropped image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Could not load image at {image_path}")
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        c,h, w = image.shape
        
        # Handle mask based on mask_status
        if mask_status == 'n/a' or mask_path == 'n/a':
            # Case 1: Benign without mask → Create black mask matching image size
            mask = np.zeros((h, w), dtype=np.uint8)
            has_mask = False  
            
        else:
            # Case 2: Valid mask exists → Crop mask using bounding box
            full_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if full_mask is None:
                # Fallback: mask file doesn't exist
                mask = np.zeros((h, w), dtype=np.uint8)
                has_mask = False  
            else:
                # Extract bounding box coordinates
                x_min = row['x_min']
                y_min = row['y_min']
                width = row['width']
                height = row['height']
                
                # Check if bbox coordinates are valid
                if str(x_min) == 'n/a' or str(width) == 'n/a':
                    # No valid bbox → resize full mask to match cropped image
                    mask = cv2.resize(full_mask, (w, h))
                else:
                    # Crop mask using bounding box to match cropped image
                    x_min, y_min = int(x_min), int(y_min)
                    width, height = int(width), int(height)
                    x_max = x_min + width
                    y_max = y_min + height
                    
                    # Ensure coordinates are within mask bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(full_mask.shape[1], x_max)
                    y_max = min(full_mask.shape[0], y_max)
                    
                    # Crop the mask
                    mask = full_mask[y_min:y_max, x_min:x_max]
                    
                    # Resize cropped mask to match cropped image size
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h))
        
        # Resize both image and mask to fixed size
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Threshold mask to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0
        
        # Apply augmentations
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Ensure mask has correct shape [1, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        
        return {
            'image': image,
            'mask': mask,
            'label': label,
            'has_mask': has_mask
        }