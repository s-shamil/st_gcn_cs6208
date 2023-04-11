import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import ipdb
import numpy as np
import pickle as pkl
import os
import sys
from torch.optim.lr_scheduler import MultiStepLR

from data import SkeletonDataset
from model_assembly import ST_GCN
from tqdm import tqdm

########################################################################################################################
########################################################################################################################

# Check Free GPU before running
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Change in model.py as well

data_dir = '/home/salman/Datasets/data/Assembly101/'
model_details = "distancePartition_stgcn_EdgeImp_handAction"

phase = 'train' # 'test': load best model and run on val set, 
               # 'train': train model

BATCH_SIZE = 32
LEARING_RATE = 0.01
NUM_CLASSES = 1380
NUM_EPOCHS = 50

DEBUG_MODE = False

train_data_file = data_dir + 'train_data.npy'
train_label_file = data_dir + 'train_label.pkl'
val_data_file = data_dir + 'val_data.npy'
val_label_file = data_dir + 'val_label.pkl'

########################################################################################################################
########################################################################################################################

if phase == 'train':
    train_dataset = SkeletonDataset(train_data_file, train_label_file, debug_mode=DEBUG_MODE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = SkeletonDataset(val_data_file, val_label_file, debug_mode=DEBUG_MODE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ipdb.set_trace()


# Model, loss function and optimizer
model = ST_GCN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


if phase == 'train':
    # Log file
    f_log = open("logs/" + model_details+"_log.txt", "w")

    # train model
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        for i, (inputs, labels) in enumerate(tqdm(train_loader, total=len(train_loader))):
            with torch.no_grad():
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        # evaluate your model on the validation set
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for inputs, labels in tqdm(val_loader, total=len(val_loader)):
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                outputs = model(inputs)

                val_loss += criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1) # Predicted ~ argmax
                val_acc += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_acc /= len(val_dataset)
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), 'saved_models/best_model_for_'+model_details+'.pt')
                best_val_loss = val_loss

        info = "Epoch [{}/{}], Training loss: {:.4f}, Validation loss: {:.4f}, Validation accuracy: {:.2f}%".format(epoch+1, NUM_EPOCHS, train_loss.item(), val_loss.item(), val_acc*100)
        
        print(info)
        f_log.write(info+'\n')
        # decay learning rate every 10 epochs
        scheduler.step()

    f_log.close()

if phase == 'test':
    model.load_state_dict(torch.load('saved_models/best_model_for_'+model_details+'.pt'))
    with torch.no_grad():
        val_acc = 0
        for inputs, labels in val_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Predicted ~ argmax
            val_acc += (predicted == labels).sum().item()

        val_acc /= len(val_dataset)

        print("TEST ACC: ", val_acc)
    
