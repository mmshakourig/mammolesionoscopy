import os
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

DEFAULT_BASE_DIR = './data/'
TARGET_SIZE = (640, 640) # Standard for ResNet, can be adjusted

def getJPGPath(base_dir, dicom_path_from_csv):
    """Converts the original CSV dicom paths to the JPG paths."""
    if pd.isna(dicom_path_from_csv):
        print("Warning: Missing DICOM path in CSV, cannot convert to JPG path.")
        return None
    
    relative_path = dicom_path_from_csv.split('/', 1)[-1] if '/' in dicom_path_from_csv else dicom_path_from_csv
    jpg_path = relative_path.replace('.dcm', '.jpg')
    return os.path.join(base_dir, 'jpeg', jpg_path)

def loadJPG2Tensor(path, size=TARGET_SIZE, is_mask=False):

    if not path or not os.path.exists(path):
        print(f"Warning: Image path does not exist: {path}")
        return torch.zeros((1, *size))
        
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Failed to read image at path: {path}")
        return torch.zeros((1, *size))
    
    img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)
    
    if not is_mask:
        # Keep as grayscale (1 channel) with values [0, 1]
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0)
    else:
        # Mask as single channel binary (0 or 1)
        img = (img > 127).astype(np.float32)
        tensor = torch.from_numpy(img).unsqueeze(0)
    
    return tensor

def getJPGTensorsFromDS(dataset_df, base_dir=DEFAULT_BASE_DIR, target_size=TARGET_SIZE, device=torch.device('cpu')):
    
    all_images = []
    masks = []

    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        # Full Image
        img_tensor = loadJPG2Tensor(
            getJPGPath(base_dir, row.get('image file path')), 
            size=target_size, is_mask=False
        )
        all_images.append(img_tensor)
        
        # Mask
        mask_tensor = loadJPG2Tensor(
            getJPGPath(base_dir, row.get('ROI mask file path')), 
            size=target_size, is_mask=True
        )
        
        masks.append(mask_tensor)
            
    # Stack lists into single tensors
    images_tensor = torch.stack(all_images) if all_images else torch.empty((0, 1, *TARGET_SIZE))
    masks_tensor = torch.stack(masks) if masks else torch.empty((0, 1, *TARGET_SIZE))
    
    # --- 4. Move to GPU (if available) ---
    images_tensor = images_tensor.to(device)
    masks_tensor = masks_tensor.to(device)

    # clear memory of individual tensors to free up RAM
    del all_images, masks

    return images_tensor, masks_tensor


# def build_unique_samples_and_tensors(base_dir=DEFAULT_BASE_DIR, save_csv_path='unique_samples.csv'):
#     """
#     1. Creates a CSV of all unique samples with: sampleID, abnormality type, assessment, pathology
#     2. Creates PyTorch tensors for unique samples, mass masks, and calc masks.
#     3. Moves tensors to GPU if available.
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # --- 1. Load DataFrames ---
#     print("Loading CSVs...")
#     try:
#         mass_train = pd.read_csv(os.path.join(base_dir, 'csv', 'mass_case_description_train_set.csv'))
#         mass_test = pd.read_csv(os.path.join(base_dir, 'csv', 'mass_case_description_test_set.csv'))
#         mass_df = pd.concat([mass_train, mass_test], ignore_index=True)
#         mass_df['abnormality type'] = 'mass'
        
#         calc_train = pd.read_csv(os.path.join(base_dir, 'csv', 'calc_case_description_train_set.csv'))
#         calc_test = pd.read_csv(os.path.join(base_dir, 'csv', 'calc_case_description_test_set.csv'))
#         calc_df = pd.concat([calc_train, calc_test], ignore_index=True)
#         calc_df['abnormality type'] = 'calc'
        
#         full_df = pd.concat([mass_df, calc_df], ignore_index=True)
#     except Exception as e:
#         print(f"Error loading CSV files from {base_dir}/csv : {e}")
#         return None, None, None

#     # --- 2. Extract Unique Samples ---
#     # A single mammogram image can have multiple ROIs, we drop duplicates by 'image file path'
#     unique_df = full_df.drop_duplicates(subset=['image file path']).copy()
#     unique_df.reset_index(drop=True, inplace=True)
    
#     # Create sampleID (1 to N)
#     unique_df.insert(0, 'sampleID', range(1, len(unique_df) + 1))
    
#     # Keep requested columns.
#     csv_cols = ['sampleID', 'abnormality type', 'assessment', 'pathology']
    
#     print(f"Found {len(unique_df)} unique samples. Saving to '{save_csv_path}'...")
#     unique_df[csv_cols].to_csv(save_csv_path, index=False)
    
#     # --- 3. Build Tensors ---
#     print("Building Image Tensors (this might take a while and require significant RAM)...")
#     all_images = []
#     masks = []

#     for _, row in tqdm(unique_df.iterrows(), total=len(unique_df)):
#         # Full Image
#         img_tensor = loadJPG2Tensor(
#             getJPGPath(base_dir, row.get('image file path')), 
#             size=TARGET_SIZE, is_mask=False
#         )
#         all_images.append(img_tensor)
        
#         # Mask
#         mask_tensor = loadJPG2Tensor(
#             getJPGPath(base_dir, row.get('ROI mask file path')), 
#             size=TARGET_SIZE, is_mask=True
#         )
        
#         # Segregate masks
#         if row['abnormality type'] == 'mass':
#             mass_masks.append(mask_tensor)
#         else:
#             calc_masks.append(mask_tensor)
            
#     # Stack lists into single tensors
#     print("Stacking tensors...")
#     try:
#         all_images_tensor = torch.stack(all_images) if all_images else torch.empty((0, 1, *TARGET_SIZE))
#         mass_masks_tensor = torch.stack(mass_masks) if mass_masks else torch.empty((0, 1, *TARGET_SIZE))
#         calc_masks_tensor = torch.stack(calc_masks) if calc_masks else torch.empty((0, 1, *TARGET_SIZE))
        
#         # --- 4. Move to GPU (if available) ---
#         print(f"Moving tensors to {device}...")
#         all_images_tensor = all_images_tensor.to(device)
#         mass_masks_tensor = mass_masks_tensor.to(device)
#         calc_masks_tensor = calc_masks_tensor.to(device)
        
#         print("\nDone!")
#         print(f"All Images Tensor Shape: {all_images_tensor.shape}")
#         print(f"Mass Masks Tensor Shape: {mass_masks_tensor.shape}")
#         print(f"Calc Masks Tensor Shape: {calc_masks_tensor.shape}")
        
#         return all_images_tensor, mass_masks_tensor, calc_masks_tensor
        
#     except RuntimeError as e:
#         print(f"RuntimeError while stacking/moving tensors: {e}")
#         print("Note: If you run out of GPU/CPU memory, consider processing in smaller batches instead of loading everything at once.")
#         return None, None, None

