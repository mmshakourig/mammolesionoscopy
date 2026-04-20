import pandas as pd
import os
import cv2
from tqdm import tqdm
import glob
import csv
import torch, torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


BASE_INPUT_PATH = "./data/"
CSV_FOLDER_PATH = os.path.join(BASE_INPUT_PATH, "csv")
IMAGE_FOLDER_PATH = os.path.join(BASE_INPUT_PATH, "jpeg")
BASE_OUTPUT_PATH = "./data/modified/"

BASE_CLEAN_PATH = "./data-cleaned/"

# Define the Enhanced PyTorch Dataset class
class MammographyDataset(Dataset):
    """Dataset for cropped ROI mammography images with proper mask handling"""

    # Data Augmentation Strategy
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.3),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])
    
    def __init__(self, dataframe, image_size=640, device=None):
        self.dataframe = dataframe
        self.transforms = self.train_transforms
        self.image_size = image_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row["cropped_image_path"]
        mask_path = row["roi_mask_path"]
        mask_status = row["mask_status"]
        pathology = row["pathology"] 
        birad = row["assessment"] 
        abnormality = row["abnormality_type"]
        
        # Determine classification label (0=BENIGN, 1=MALIGNANT)
        pathology_label = 1 if 'MALIGNANT' in str(pathology).upper() else 2 if 'WITH' in str(pathology).upper() else 0
        birad_label = birad  
        abnormality_label = 1 if 'MASS' in str(abnormality).upper() else 0

        # Determine if mask is available
        has_mask = (mask_status == 'valid')  
        
        # Read cropped image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not load image at {image_path}")
            image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        h, w = image.shape
        
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
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # Threshold mask to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0
        
        # Apply augmentations
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Ensure image/mask are tensors
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)

        # Ensure mask has correct shape [1, H, W]
        if mask.ndim == 2:
            mask = torch.unsqueeze(mask, 0)
        
        # Move tensors to device
        image = image.to(self.device)
        mask = mask.to(self.device)
        
        return {
            'image': image,
            'mask': mask,
            'pathology_label': torch.tensor(pathology_label, device=self.device),
            'birad_label': torch.tensor(birad_label, device=self.device),
            'abnormality_label': torch.tensor(abnormality_label, device=self.device),
            'has_mask': has_mask
        }


class MammographySegmentationDataset():
    # Data Augmentation Strategy
    img_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.3),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])

    def __init__(self, dataframe, datatype, image_size, device):
        self.dataframe = dataframe
        self.image_size = image_size
        self.device = device
        self.transforms = self.img_transforms

    def save_images_mask_pairs(self, output_folder=None):
        if output_folder is None:
            raise ValueError("output_folder must be provided")
        os.makedirs(output_folder, exist_ok=True)
        
        for _, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe), desc="Generating image-mask pairs"):
            image_path = row["cropped_image_path"]
            mask_path = row["roi_mask_path"]
            mask_status = row["mask_status"]

            # Determine if mask is available
            if (mask_status != 'valid'):
                continue  
            
            # Read cropped image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load image at {image_path}")
                image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            
            h, w = image.shape
            
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
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            
            # Threshold mask to binary
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask.astype(np.float32) / 255.0
            
            # Apply augmentations
            if self.transforms:
                transformed = self.transforms(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            
            # Ensure image/mask are tensors
            if not torch.is_tensor(image):
                image = torch.from_numpy(image)
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(mask)

            # Ensure mask has correct shape [1, H, W]
            if mask.ndim == 2:
                mask = torch.unsqueeze(mask, 0)
            
            # Move tensors to device
            # image = image.to(self.device)
            # mask = mask.to(self.device)

            # save masks to output folder + masks
            cv2.imwrite(os.path.join(output_folder, f"mask_{os.path.basename(image_path)}"), (mask.squeeze().cpu().numpy() * 255).astype(np.uint8))


            # save images to output folder + images

        



def find_image_path_in_folder(folder_path):
    """
    Finds the first .jpg or .png file in a given folder.
    """
    if not folder_path or not os.path.isdir(folder_path):
        return None
        
    image_files = (
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.png"))
    )
    
    if image_files:
        return image_files[0] 
    return None

def compute_all_bounding_boxes(mask_path, min_area=100):
    """
    Returns a list of bounding boxes [[x_min, y_min, width, height],...]
    Returns None if mask doesn't exist or is invalid
    """
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        boxes.append([x, y, w, h])

    return boxes if boxes else None

def build_metadata_lookup(dicom_info_path, jpeg_base_dir, *args):
    print(f"Building metadata lookup from: {dicom_info_path}")
    master_map = {}
    
    try:
        dicom_info = pd.read_csv(dicom_info_path, dtype=str)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {dicom_info_path}")
        return master_map

    valid_descriptions = {arg for arg in args}
    filtered_df = dicom_info[dicom_info['SeriesDescription'].isin(valid_descriptions)]

    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Building lookup map"):
        series_desc = row['SeriesDescription'] 
        patient_id_composite = row['PatientID'] 

        if 'image_path' in row and pd.notna(row['image_path']):
            rel_path = row['image_path']

            # if series_desc is "full mammogram images":

            if 'jpeg' in rel_path:
                clean_rel_path = rel_path.split('jpeg')[-1].strip("/\\")
                full_path = os.path.join(jpeg_base_dir, clean_rel_path)
            else:
                full_path = os.path.join(jpeg_base_dir, rel_path)
        else:
            series_uid = row['SeriesInstanceUID']
            folder_path = os.path.join(jpeg_base_dir, series_uid)
            full_path = find_image_path_in_folder(folder_path)
            
        if not full_path: 
            continue

        if patient_id_composite not in master_map:
            master_map[patient_id_composite] = {}
            
        master_map[patient_id_composite][series_desc] = full_path
            
    print(f"Metadata lookup map built. Found {len(master_map)} unique composite keys.")
    return master_map


def build_master_dataset(MASTER_LIST_PATH="./data/modified/master_dataset.csv", 
                       argument1="cropped images", 
                       argument2="ROI mask images",
                       argument3="full mammogram images"):
    """
    Build master dataset. If ROI mask is missing:
    - For BENIGN cases: sets mask path to 'n/a' and bbox to 'n/a'
    - For MALIGNANT cases: skips the image entirely
    """
    IMAGE_FOLDER_PATH = os.path.join(BASE_INPUT_PATH, "jpeg")
    DICOM_INFO_PATH = os.path.join(BASE_INPUT_PATH, "csv/dicom_info.csv")
    
    INPUT_CSVS = [
        os.path.join(BASE_INPUT_PATH, "csv/mass_case_description_train_set.csv"),
        os.path.join(BASE_INPUT_PATH, "csv/mass_case_description_test_set.csv"),
        os.path.join(BASE_INPUT_PATH, "csv/calc_case_description_train_set.csv"),
        os.path.join(BASE_INPUT_PATH, "csv/calc_case_description_test_set.csv")
    ]

    # Build the authoritative map
    master_map = build_metadata_lookup(DICOM_INFO_PATH, IMAGE_FOLDER_PATH, argument1, argument2, argument3)
    
    if not master_map:
        return

    found_pairs_count = 0
    missing_mask_count = 0
    skipped_malignant_count = 0
    
    with open(MASTER_LIST_PATH, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow([
            'cropped_image_path', 'roi_mask_path', 'full_image_path',
            'x_min', 'y_min', 'width', 'height',
            'pathology', 'assessment', 'patient_id', 'series_type', 'abnormality_type', 'mask_status'
        ])

        for filepath in INPUT_CSVS:
            filename = os.path.basename(filepath)
            if not os.path.exists(filepath):
                continue
            
            # Determine prefixes
            if "mass" in filename.lower():
                type_prefix = "Mass"
            elif "calc" in filename.lower():
                type_prefix = "Calc"
            else:
                continue
                
            if "train" in filename.lower():
                split_prefix = "Training"
            elif "test" in filename.lower():
                split_prefix = "Test"
            else:
                continue
                
            full_prefix = f"{type_prefix}-{split_prefix}"

            with open(filepath, "r") as infile:
                csv_reader = csv.reader(infile)
                header = next(csv_reader)
                
                pathology_idx = header.index('pathology')
                assessment_idx = header.index('assessment')
                patient_id_idx = header.index('patient_id')
                breast_idx = header.index('left or right breast')
                view_idx = header.index('image view')
                abnormality_id_idx = header.index('abnormality id')

                for row in tqdm(csv_reader, desc=f"Processing {filename}"):
                    if not any(row):
                        continue
                    
                    pathology = row[pathology_idx]
                    assessment = row[assessment_idx]
                    patient_id = row[patient_id_idx]
                    side = row[breast_idx]
                    view = row[view_idx]
                    abn_id = row[abnormality_id_idx]
                    
                    try:
                        abn_id_clean = str(int(float(abn_id)))
                    except ValueError:
                        abn_id_clean = str(abn_id).strip()

                    composite_key = f"{full_prefix}_{patient_id}_{side}_{view}_{abn_id_clean}"
                    
                    study_data = master_map.get(composite_key)
                    if not study_data:
                        continue
                    
                    full_crop_path = study_data.get('cropped images')
                    full_mask_path = study_data.get('ROI mask images')
                    full_mammogram_path = study_data.get('full mammogram images')
                    
                    # Check if cropped image exists
                    if not full_crop_path:
                        continue
                    
                    mask_status = 'valid'
                    
                    # Handle missing mask based on pathology
                    if not full_mask_path:
                        # Check if pathology is BENIGN or BENIGN_WITHOUT_CALLBACK
                        pathology_upper = str(pathology).upper()
                        is_benign = 'BENIGN' in pathology_upper and 'MALIGNANT' not in pathology_upper
                        
                        if is_benign:
                            # Set mask path to 'n/a' for benign cases without mask
                            full_mask_path = 'n/a'
                            mask_status = 'n/a'
                            missing_mask_count += 1
                            
                            # Write entry with n/a values
                            csv_writer.writerow([
                                full_crop_path,
                                'n/a',  # roi_mask_path
                                full_mammogram_path,
                                'n/a', 'n/a', 'n/a', 'n/a',  # bbox coordinates
                                pathology,
                                assessment,
                                patient_id,
                                full_prefix,
                                type_prefix,  # abnormality_type
                                'n/a'  # mask_status
                            ])
                            found_pairs_count += 1
                        else:
                            # Skip malignant cases without masks
                            skipped_malignant_count += 1
                        continue
                    
                    # Safety check: if paths are identical, skip
                    if full_crop_path == full_mask_path and mask_status == 'valid':
                        continue
                    
                    # Compute bounding boxes
                    boxes = compute_all_bounding_boxes(full_mask_path, min_area=100)
                    
                    # If mask is valid but no boxes found
                    if boxes is None:
                        # Write entry with n/a values for bounding box
                        csv_writer.writerow([
                            full_crop_path,
                            full_mask_path,
                            full_mammogram_path,
                            'n/a', 'n/a', 'n/a', 'n/a',
                            pathology,
                            assessment,
                            patient_id,
                            full_prefix,
                            type_prefix,
                            mask_status
                        ])
                        found_pairs_count += 1
                    else:
                        # Write entry for each bounding box
                        for (x_min, y_min, width, height) in boxes:
                            csv_writer.writerow([
                                full_crop_path,
                                full_mask_path,
                                full_mammogram_path,
                                x_min, y_min, width, height,
                                pathology,
                                assessment,
                                patient_id,
                                full_prefix,
                                type_prefix, # abnormality_type
                                mask_status
                            ])
                            found_pairs_count += 1

    print(f"\n{'='*60}")
    print(f"DATASET BUILD SUMMARY")
    print(f"{'='*60}")
    print(f"Master list saved to: {MASTER_LIST_PATH}")
    print(f"Valid pairs found: {found_pairs_count}")
    print(f"Benign cases without masks (n/a): {missing_mask_count}")
    print(f"Malignant cases skipped (no mask): {skipped_malignant_count}")
    print(f"{'='*60}")

    return master_map
