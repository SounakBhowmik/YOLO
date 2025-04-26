# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import os
import cv2
import xmltodict
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class_map = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]

class VOCDataset(Dataset):
    def __init__(self, image_dir, annot_dir, split_file, S=7, B=1, C=20, transform=None):
        """
        Args:
            image_dir (str): Path to 'JPEGImages' folder.
            annot_dir (str): Path to 'Annotations' folder.
            split_file (str): Path to 'ImageSets/Main/train.txt'.
            S (int): Grid size (7x7 for Fast YOLO).
            B (int): Number of boxes per grid cell.
            C (int): Number of classes (20 for Pascal VOC).
            transform (callable, optional): Optional transform for images.
        """
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        # Read image file names
        with open(split_file, "r") as f:
            self.image_filenames = f.read().splitlines()
        self.image_filenames = [f.split(' ')[0] for f in self.image_filenames]
        
        # Pascal VOC classes
        self.class_map = {cls: i for i, cls in enumerate(class_map)}

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.image_filenames[idx] + ".jpg")
        annot_path = os.path.join(self.annot_dir, self.image_filenames[idx] + ".xml")

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load and parse XML annotation
        with open(annot_path) as f:
            annot = xmltodict.parse(f.read())["annotation"]

        # Image dimensions
        W, H = int(annot["size"]["width"]), int(annot["size"]["height"])
        cell_W, cell_H = W/self.S, H/self.S

        # Convert annotations to YOLO format
        labels = []
        objects = annot["object"]
        if isinstance(objects, dict):  # If only one object, convert to list
            objects = [objects]

        for obj in objects:
            class_label = self.class_map[obj["name"]]
            bbox = obj["bndbox"]
            xmin, ymin, xmax, ymax = map(int, [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]])

            # Convert to relative center coordinates and width/height
            x_c = ((xmin + xmax) / 2) / W
            y_c = ((ymin + ymax) / 2) / H
            
            w_box = (xmax - xmin) / W
            h_box = (ymax - ymin) / H

            labels.append([x_c, y_c, w_box, h_box, class_label])

        labels = torch.tensor(labels)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, bboxes=labels[:, :4].tolist(), labels=labels[:, 4].tolist())
            image = transformed["image"]
            labels = torch.cat([torch.tensor(transformed["bboxes"]), torch.tensor(transformed["labels"]).unsqueeze(1)], dim=1)
            #print(f'labels\' shape = {labels.shape}')
            
        # Convert to YOLO grid format
        label_matrix = torch.zeros((self.S, self.S, self.B * 5 + self.C))
        for box in labels:
            x, y, w, h, class_label = box.tolist()
            g_x, g_y = int(self.S * x), int(self.S * y)
            label_matrix[g_y, g_x, 0:5] = torch.tensor([(x*W - g_x*cell_W)/W, (y*H - g_y*cell_H)/H, w, h, 1])  # Object present
            label_matrix[g_y, g_x, 5 + int(class_label)] = 1  # One-hot class encoding

        return image, label_matrix

#%%
train_transform = A.Compose([
    A.Resize(448, 448),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    
    # Color transforms
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.5
    ),
    
    A.GaussNoise(p=0.4),
    
    # Blur/noise
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.3),
    
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        min_holes=1,
        min_height=8,
        min_width=8,
        fill_value=(114, 114, 114),
        p=0.4
    ),
    
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], min_visibility=0.3))


train_transform2 = A.Compose([
        # Basic spatial transforms
        A.LongestMaxSize(max_size=448),  # Resize maintaining aspect ratio
        A.PadIfNeeded(
            min_height=448,
            min_width=448,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # YOLO-style padding (gray)
        ),
        
        # Color transforms
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.ToGray(p=0.05),  # Small chance to convert to grayscale
        
        # Blur/noise
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Weather effects
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        A.RandomSunFlare(src_radius=100, p=0.1),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),  # Less common than horizontal
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
            p=0.5
        ),
        
        # Advanced augmentations
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=(114, 114, 114),
            p=0.4
        ),
        A.RandomGridShuffle(grid=(3, 3), p=0.1),
        
        # Final processing
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],    # ImageNet std
            max_pixel_value=255.0
        ),
        ToTensorV2()  # Convert to PyTorch tensor
    ], bbox_params=A.BboxParams(
        format='yolo',  # Use 'yolo' format (normalized cx, cy, w, h)
        min_visibility=0.2,  # Discard boxes with <20% visibility after transform
        label_fields=["labels"]  # Required for class labels
    ))



val_transform = A.Compose([
    A.Resize(448, 448),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], min_visibility=0.3))

#%% Download and preprocess data
import kagglehub
from Utils import plot_object_distribution_from_files
import random
from torch.utils.data import DataLoader

def download_and_preprocess_data():
    # Download latest version
    root_data_path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")

    train_val_data_path = f'{root_data_path}/VOC2012_train_val/VOC2012_train_val'
    test_data_path = f'{root_data_path}/VOC2012_test/VOC2012_test'

    # Plot the data distribution
    #plot_object_distribution_from_files(train_val_data_path + '/ImageSets/Main')

    image_dir =  f"{train_val_data_path}/JPEGImages"
    annot_dir =  f"{train_val_data_path}/Annotations"
    split_file = f"{train_val_data_path}/ImageSets/Main/train.txt"
    
    # Paths to your dataset split files
    train_split_file = f"{train_val_data_path}/ImageSets/Main/train.txt"
    val_split_file = f"{train_val_data_path}/ImageSets/Main/val.txt"


    # Read file paths
    with open(train_split_file, "r") as f:
        train_files = f.read().splitlines()

    with open(val_split_file, "r") as f:
        val_files = f.read().splitlines()


    # Randomly select 75% of val files to move to train
    num_to_move = int(0.75 * len(val_files))
    selected_files = random.sample(val_files, num_to_move)

    # Update train and val lists
    train_files.extend(selected_files)
    val_files = [f for f in val_files if f not in selected_files]

    # Save updated splits
    train_split_file = "train.txt"
    val_split_file   = "val.txt"

    with open(train_split_file, "w") as f:
        f.write("\n".join(train_files))

    with open(val_split_file, "w") as f:
        f.write("\n".join(val_files))

    print(f"Moved {num_to_move} images from validation to training set.")
    print(f"New train size: {len(train_files)}, New val size: {len(val_files)}")
    
    # Define dataset paths
    image_dir = f"{train_val_data_path}/JPEGImages"
    annot_dir = f"{train_val_data_path}/Annotations"

    # Create dataset and dataloader
    train_dataset = VOCDataset(image_dir, annot_dir, train_split_file, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=6)

    val_dataset = VOCDataset(image_dir, annot_dir, val_split_file, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=6)
    
    return train_loader, val_loader

