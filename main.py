# -*- coding: utf-8 -*-

import kagglehub
from Utils import plot_object_distribution_from_files, YOLOloss
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import random
import torch
from Models import FastYOLO_mobile01
from Data import VOCDataset, train_transform, val_transform

# Download latest version
root_data_path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")

train_val_data_path = f'{root_data_path}/VOC2012_train_val/VOC2012_train_val'
test_data_path = f'{root_data_path}/VOC2012_test/VOC2012_test'

# Plot the data distribution
plot_object_distribution_from_files(train_val_data_path + '/ImageSets/Main')

image_dir =  f"{train_val_data_path}/JPEGImages"
annot_dir =  f"{train_val_data_path}/Annotations"
split_file = f"{train_val_data_path}/ImageSets/Main/train.txt"

#%% Build the training pipeline

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



#%%
model = FastYOLO_mobile01()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Your model should have been loaded already, don't instantiate it again here, it breaks the kernel for some reason I am unable to understand
# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5, verbose=True)


# loss function instantiate
loss_fn = YOLOloss()

# Define dataset paths
image_dir = f"{train_val_data_path}/JPEGImages"
annot_dir = f"{train_val_data_path}/Annotations"
train_split_file = train_split_file
val_split_file = val_split_file

# Create dataset and dataloader
train_dataset = VOCDataset(image_dir, annot_dir, train_split_file, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)

val_dataset = VOCDataset(image_dir, annot_dir, val_split_file, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, pin_memory=True)



# Save results function
results_path = '/content/drive/MyDrive/YOLO_Results'
model_name = 'mobilenet_v3_3' # With the dense layers on the head

#%%
from Utils import train_fn, validate_fn, write_results
import torch
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import os



#writer = SummaryWriter()

num_epochs = 100

train_res = {'epoch':[],
            'total_loss': [],
            'loss_box': [],
            'loss_obj_noobj': [],
            'loss_class': []}

val_res = {'epoch':[],
            'total_loss': [],
            'loss_box': [],
            'loss_obj_noobj': [],
            'loss_class': []}


for epoch in range(num_epochs):
    train_loss = train_fn(model, train_loader, optimizer, loss_fn, device)

    # Add to tensorboard
    # writer.add_scalars('Train', {'total_loss': train_loss[0],
    #                             'loss_box': train_loss[1],
    #                             'loss_obj_noobj': train_loss[2],
    #                             'loss_class': train_loss[3]}, epoch)

    # Add the log values to the dictionary, later we shall convert it into a csv file
    train_res['epoch'].append(epoch)
    for k, i in zip(list(train_res.keys()), range(5)):
        if(k != 'epoch'):
            train_res[k].append(train_loss[i-1].item())

    if(epoch%5 == 0):
        print('Validation -->')
        val_loss = validate_fn(model, val_loader, loss_fn, device)

        # Add to tensorboard
        # writer.add_scalars('Val', {'total_loss': val_loss[0],
        #                             'loss_box': val_loss[1],
        #                             'loss_obj_noobj': val_loss[2],
        #                             'loss_class': val_loss[3]}, epoch)

        # Add the val log values to the dictionary, later we shall convert it into a csv file
        val_res['epoch'].append(epoch)
        for k, i in zip(list(val_res.keys()), range(5)):
            if(k != 'epoch'):
                val_res[k].append(val_loss[i-1].item())

    scheduler.step(val_loss[0])
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss[0]:.4f}, Val Loss: {val_loss[0]:.4f}")

# writer.flush()
# writer.close()


# Save results
results_path = '/content/drive/MyDrive/YOLO/results'
models_path = '/content/drive/MyDrive/YOLO/models'

model_name = 'mobilenet_v3_3'

write_results(result = train_res, results_path = results_path, model_name  = model_name, suffix = 'train')
write_results(result = val_res, results_path = results_path, model_name  = model_name, suffix = 'val')

# Save model
model_path = os.path.join(models_path, model_name) + '.pth'
torch.save(model.state_dict(), model_path)







