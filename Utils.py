# -*- coding: utf-8 -*-

'''
### **Step 3**: Define a Loss class that combines the three kinds of losses in YOLO,

1.   Bounding box definition loss - use *mse* (Mean squared error)
2.   Objectness (if there is any object in this particular grid) - use *bce* (binary cross entropy with logit loss)
3.   No object (If there is no object as per the ground truth, but the model detects an object in there, penalize that) - *bce*
4.   Class error (Which class of object there is) - *bce*
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import torchvision.transforms as transforms
import cv2
import xmltodict
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm


#%%
class_map = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]

#%%
class YOLOloss(nn.Module):
    def __init__(self, lamda_coordinate=5, lamda_noobj = 0.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        #self.bce = nn.BCEWithLogitsLoss(reduction='sum')

        self.lamda_coordinate = lamda_coordinate
        self.lamda_noobj = lamda_noobj
        self.noise = 1e-6

    def forward(self, prediction: torch.tensor, targets: torch.tensor):
        # Assuming prediction is a (-1, 7, 7, 5 + 20) dimensional array
        objmask = targets[:, :, :, 4] ==   1   # cells with object
        noobjmask = targets[:, :, :, 4] == 0     # empty cells
        #print(prediction[objmask].shape)
        loss_box = self.lamda_coordinate*(
            
            self.mse(prediction[objmask][..., :2], targets[objmask][..., :2])
            + self.mse(torch.sqrt(prediction[objmask][..., 2:4]+self.noise), torch.sqrt(targets[objmask][..., 2:4]))
            
        )

        loss_obj = self.mse(prediction[objmask][...,4], targets[objmask][...,4])
        
        loss_noobj = self.lamda_noobj * self.mse(prediction[noobjmask][...,4], targets[noobjmask][...,4])
        
        loss_class = self.mse(prediction[objmask][...,5:], targets[objmask][...,5:])

        loss_obj_noobj = loss_obj + loss_noobj
        total_loss = loss_box + loss_obj_noobj + loss_class

        return total_loss, loss_box, loss_obj_noobj, loss_class

'''
# test-case
preds = torch.rand(1,7,7,25)
target = torch.vstack((torch.tensor([[0.5, 0.5, 0.1, 0.2, 1, 0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0.6, 0.7, 0.1, 0.2, 1, 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]), torch.zeros(47,25))).reshape(1,7,7,25)
loss = YOLOloss()
print(loss(preds, target))
'''
#%% Custom lr_scheduler
import math

def set_lr(optimizer, epoch):
    # increase the lr from 1e-3 to 1e-2 in a scope of 15 epochs, then continue with 1e-2 for 60 epochs, 
    cur_lr = optimizer.param_groups[0]['lr']

    if(epoch<15):
        cur_lr *= math.pow(10, 1/15)
        
    elif(epoch>=75):
        if(epoch<105):
            cur_lr = 1e-3
        else:
            cur_lr = 1e-4
    
    optimizer.param_groups[0]['lr'] = cur_lr    
    return cur_lr



#%%

import os
import matplotlib.pyplot as plt

def plot_object_distribution_from_files(folder_path , save_path="object_distribution"):
    """
    Generates a pie chart to visualize the distribution of different object types
    based on the number of file paths inside text files named in the format 'object_trainval.txt'.

    Parameters:
    folder_path (str): Path to the folder containing the text files.
    save_path (str): Path to save the figure (without extension).

    Returns:
    None (Saves the figure as PNG and PDF)
    """
    object_counts = {}

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith("_trainval.txt"):  # Match the pattern
            object_type = filename.replace("_trainval.txt", "")  # Extract object name
            file_path = os.path.join(folder_path, filename)

            # Count the number of lines in the file (assuming each line is a data entry)
            with open(file_path, "r") as f:
                count = sum(1 for _ in f)  # Efficient line counting

            object_counts[object_type] = count

    # Sort objects by count for better visualization
    sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
    object_types, counts = zip(*sorted_objects)  # Unpack sorted values

    # Define a color palette for better visualization
    colors = plt.cm.Paired.colors[:len(object_types)]

    # Create pie chart
    plt.figure(figsize=(7, 7), dpi=300)
    wedges, texts, autotexts = plt.pie(
        counts, labels=object_types, autopct='%1.1f%%',
        colors=colors, startangle=140, textprops={'fontsize': 12},
        wedgeprops={"edgecolor": "black", "linewidth": 1}
    )

    # Title and styling
    #plt.title("Object Type Distribution in Dataset", fontsize=14, fontweight="bold")

    # Save figure in high resolution
    plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")

    # Show plot
    plt.show()





#%%



def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizes an image from [0,1] or [-1,1] to original pixel range [0,255].

    Args:
    - image (numpy array): Normalized image of shape (H, W, C).
    - mean (list): Mean values used for normalization.
    - std (list): Std deviation values used for normalization.

    Returns:
    - Denormalized image in the range [0,255].
    """
    image = image * std + mean  # Reverse normalization
    image = np.clip(image * 255, 0, 255).astype(np.uint8)  # Scale to 0-255
    return image

# Add the predicted class
def visualize_bboxes(image, label_matrix):
    """
    Visualizes bounding boxes on an image.

    Args:
    - image (numpy array): The input image (H, W, C) in RGB format.
    - label_matrix (torch.Tensor): The YOLO label matrix of shape (7, 7, 25).

    Returns:
    - Displays the image with bounding boxes.
    """
    S = 7  # Grid size
    W, H = 448, 448  # Image dimensions
    cell_size = W / S  # 64 pixels per grid cell

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # Convert (C, H, W) → (H, W, C)

    image = denormalize_image(image)  # Fix high contrast issue

    img_copy = image.copy()

    # Iterate over each cell in the grid
    for i in range(S):
        for j in range(S):
            cell_data = label_matrix[i, j]
            predicted_class = class_map[int(cell_data[5:].argmax())]  # Predicted class index
            obj_prob = cell_data[4]  # Objectness score

            if obj_prob > 0.5:  # Threshold to visualize only detected objects
                x_offset, y_offset, w, h = cell_data[:4]

                n_cell_W = n_cell_H = cell_size
                xc, yc = x_offset * W + j * n_cell_W, y_offset * H + i * n_cell_H

                w_box, h_box = w * W, h * H  # Absolute values
                x_min, x_max = int(xc - w_box/2), int(xc + w_box/2) # Absolute values
                y_min, y_max = int(yc - h_box/2), int(yc + h_box/2) # Absolute values

                # Draw bounding box
                cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(img_copy, f"{obj_prob:.2f}", (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.putText(img_copy, f"{predicted_class}", (x_min, y_min + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display image
    plt.figure(figsize=(8, 8))
    plt.imshow(img_copy)
    plt.axis("off")
    plt.show()

'''
# testing
test_batch_image, test_batch_labels = next(iter(train_loader))
for img, label in zip(test_batch_image, test_batch_labels):
    visualize_bboxes(img.permute(1, 2, 0).numpy(), label)

'''

#%%

# Display a random image along with its bounding boxes
import matplotlib.pyplot as plt
from math import ceil, floor


def visualize_image_with_bb(image_dir, annot_dir, split_file):
    with open(split_file, "r") as f:
        image_filenames = f.read().splitlines()
    
    n = 0 #10
    for i in range(n):
        img_path  = os.path.join(image_dir, image_filenames[i] + ".jpg")
        annot_path = os.path.join(annot_dir, image_filenames[i] + ".xml")
        #print(img_path)
        image = cv2.imread(img_path)
        S = 7 # 7x7 grid
    
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        img_copy = image.copy()
        img_reshaped = cv2.resize(image, (448, 448))
    
        # Load and parse XML annotation
        with open(annot_path) as f:
            annot = xmltodict.parse(f.read())["annotation"]
    
        # Image dimensions
        W, H = int(annot["size"]["width"]), int(annot["size"]["height"])
        cell_W = W / S
        cell_H = H / S
    
        objects = annot["object"]
        if isinstance(objects, dict):  # If only one object, convert to list
            objects = [objects]
    
        for obj in objects:
            class_label = obj["name"]
            bbox = obj["bndbox"]
            xmin, ymin, xmax, ymax = map(int, [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]])
    
            xc, yc = (xmax + xmin)/2, (ymax + ymin)/2 # absolute centre coordinates
            #print(xc/W*448, yc/H*448, xmin/W*448, xmax/W*448, ymin/H*448, ymax/H*448)
    
            #cv2.circle(img_copy, (int(xc), int(yc)), 1, (0,255,0), 10)
    
    
            xc, yc =  xc/W, yc/H  # standardised centre coordinates
            w_box, h_box = (xmax - xmin)/W, (ymax - ymin)/H # standardised box wdith and heights
    
            # Now convert them to bbox specific coordinates
            g_x, g_y = int(xc*S), int(yc*S)                         # They are index of the grid cell, which the object belongs to
            xc, yc = (xc*W - g_x*cell_W)/W, (yc*H - g_y*cell_H)/H   # These are the bbox coordinates relative to the grid cell <-- Standardised values
    
    
            # Now we have the relative xc, yc, w_box and h_box. Now we shall draw the bbox on the transformed image
            n_cell_W, n_cell_H = 448 / S, 448 / S
            xc, yc = xc * 448 + g_x * n_cell_W, yc * 448 + g_y * n_cell_H # Absolute values of the centre coordinates in the reshaped image
    
    
            w_box, h_box = w_box * 448, h_box * 448  # Absolute values
            x_min, x_max = int(xc - w_box/2), int(xc + w_box/2) # Absolute values
            y_min, y_max = int(yc - h_box/2), int(yc + h_box/2) # Absolute values
    
            #print(xc, yc, x_min, x_max, y_min, y_max)
    
            cv2.rectangle(img_reshaped, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(img_reshaped, f"{class_label}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            '''
            xc = xc*448 + g_x*448/7
            yc = yc*448 + g_y*448/7
            #print(xc, yc)'''
    
            #cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            #cv2.circle(img_copy, (int(xc*W + g_x*cell_W), int(yc*W + g_y*cell_H)), 1, (0,0,255), 10)
        #plt.imshow(img_copy)
        plt.imshow(img_reshaped)
        plt.show()

#%%
def write_results(result, results_path , model_name , suffix = 'train'):

    with open(f'{results_path}/{model_name}_{suffix}_results.csv', 'w') as f:
        writer = csv.writer(f)

        list_keys = list(result.keys())

        limit = len(result[list_keys[0]])

        writer.writerow(result.keys())

        for i in range(limit):
            writer.writerow([result[x][i] for x in list_keys])

# The train and validate functions
def train_fn(model, dataloader, optimizer, loss_fn, device):
    model.train()
    Loss = np.array([0.,0.,0.,0.])

    for images, targets in tqdm(dataloader):
        images, targets = images.to(device), targets.to(device), #[t.to(device) for t in targets]

        optimizer.zero_grad()
        predictions = model(images)
        #print(predictions.shape)
        total_loss, loss_box, loss_obj_noobj, loss_class = loss_fn(predictions, targets)
        Loss = Loss + np.array([total_loss.item(), loss_box.item(), loss_obj_noobj.item(), loss_class.item()])

        total_loss.backward()
        optimizer.step()

    return Loss / len(dataloader)



@torch.no_grad()
def validate_fn(model, dataloader, loss_fn, device):
    model.eval()
    Loss = np.array([0.,0.,0.,0.])

    for images, targets in tqdm(dataloader):
        images, targets = images.to(device), targets.to(device) #[t.to(device) for t in targets]
        predictions = model(images)
        total_loss, loss_box, loss_obj_noobj, loss_class = loss_fn(predictions, targets)
        Loss = Loss + np.array([total_loss.item(), loss_box.item(), loss_obj_noobj.item(), loss_class.item()])

    return Loss / len(dataloader)





