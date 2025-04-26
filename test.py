# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from Utils import visualize_bboxes
from Data import download_and_preprocess_data

train_loader, val_loader = download_and_preprocess_data()

# testing
test_batch_image, test_batch_labels = next(iter(train_loader))
for img, label in zip(test_batch_image, test_batch_labels):
    #print(label)
    
    visualize_bboxes(img.permute(1, 2, 0).numpy(), label)
    