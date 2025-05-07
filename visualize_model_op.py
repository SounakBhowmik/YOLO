# -*- coding: utf-8 -*-
import torch
from Models import FastYOLO_mobile01
from Data import VOCDataset, train_transform, val_transform
from torch.utils.data import DataLoader
from Utils import visualize_bboxes


model_path = 'mobilenet_v3_3.pth'
model = FastYOLO_mobile01()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

root_data_path = '/home/sbhowmi2/.cache/kagglehub/datasets/gopalbhattrai/pascal-voc-2012-dataset/versions/1'
train_val_data_path = f'{root_data_path}/VOC2012_train_val/VOC2012_train_val'
test_data_path = f'{root_data_path}/VOC2012_test/VOC2012_test'


image_dir = f"{train_val_data_path}/JPEGImages"
annot_dir = f"{train_val_data_path}/Annotations"
split_file = "val.txt"

test_dataset = VOCDataset(image_dir, annot_dir, split_file, transform = val_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, pin_memory=True)

model.eval()


test_images, test_targets = next(iter(test_loader))  # Load a batch
test_images = test_images.to(device)

with torch.no_grad():  # No need to track gradients
    predictions = model(test_images)
    for image, pred in zip(test_images, predictions):
        visualize_bboxes(image.detach().cpu(), pred.detach().cpu())
        
        
#%% Compute Map
from Utils import compute_map
compute_map(model, test_loader, device, iou_threshold=0.5)









