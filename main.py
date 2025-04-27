# -*- coding: utf-8 -*-

import kagglehub
from Utils import plot_object_distribution_from_files, YOLOloss, set_lr, compute_map
import torch.optim as optim
from tqdm import tqdm
import torch
from Models import FastYOLO_mobile01, FastYOLO_resnet
from Data import download_and_preprocess_data


#%%
model = FastYOLO_resnet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Your model should have been loaded already, don't instantiate it again here, it breaks the kernel for some reason I am unable to understand
# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)


# loss function instantiate
loss_fn = YOLOloss()


train_loader, val_loader = download_and_preprocess_data()

#%%
from Utils import train_fn, validate_fn, write_results
import torch
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter()

num_epochs = 150

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
    writer.add_scalars('Train', {'total_loss': train_loss[0],
                                'loss_box': train_loss[1],
                                'loss_obj_noobj': train_loss[2],
                                'loss_class': train_loss[3]}, epoch)

    # Add the log values to the dictionary, later we shall convert it into a csv file
    train_res['epoch'].append(epoch)
    for k, i in zip(list(train_res.keys()), range(5)):
        if(k != 'epoch'):
            train_res[k].append(train_loss[i-1].item())

    if((epoch)%5 == 0):
        print('Validation -->')
        val_loss = validate_fn(model, val_loader, loss_fn, device)

        #Add to tensorboard
        writer.add_scalars('Val', {'total_loss': val_loss[0],
                                    'loss_box': val_loss[1],
                                    'loss_obj_noobj': val_loss[2],
                                    'loss_class': val_loss[3]}, epoch)

        # Add the val log values to the dictionary, later we shall convert it into a csv file
        val_res['epoch'].append(epoch)
        for k, i in zip(list(val_res.keys()), range(5)):
            if(k != 'epoch'):
                val_res[k].append(val_loss[i-1].item())
                
    scheduler.step()
    
    #lr = set_lr(optimizer, epoch)
    print(f"lr = {scheduler.get_last_lr()}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss[0]:.4f}, Val Loss: {val_loss[0]:.4f}")

writer.flush()
writer.close()


# Save results
results_path = 'results'
models_path = 'models'

model_name = 'FastYOLO_resnet'

write_results(result = train_res, results_path = results_path, model_name  = model_name, suffix = 'train')
write_results(result = val_res, results_path = results_path, model_name  = model_name, suffix = 'val')

# Save model
model_path = os.path.join(models_path, model_name) + '.pth'
torch.save(model.state_dict(), model_path)


#%%
'''
# Compute map
# Load model
models_path = 'Replace with your model path'#'/content/drive/MyDrive/YOLO/models'
model_name = 'mobilenet_v3'
model_path = os.path.join(models_path, model_name)+'.pth'
model.load_state_dict(torch.load(model_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

compute_map(model, val_loader, device, iou_threshold=0.5)
'''






