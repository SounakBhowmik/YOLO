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
    A.RandomBrightnessContrast(p=0.2),
    A.Blur(p=0.1),
    A.GaussNoise(p=0.2),
    A.MotionBlur(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], min_visibility=0.3))

val_transform = A.Compose([
    A.Resize(448, 448),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], min_visibility=0.3))
