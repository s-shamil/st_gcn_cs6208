import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pkl

class SkeletonDataset(Dataset):
    def __init__(self, data_file, label_file, debug_mode=False):
        self.data = torch.from_numpy(np.load(data_file))
        # Label file is pickle, with one pair-> (list of json files, list of labels)
        with open(label_file, 'rb') as f:
            lbl = pkl.load(f)[1]
        self.labels = torch.Tensor(lbl)

        if debug_mode:
            subset_len = int((self.data.shape[0])*0.10) # Just taking 10% of the data
            self.data = self.data[0:subset_len]
            self.labels = self.labels[0:subset_len]
            print("Dataset size in debug mode: ", subset_len)
            print("Data shape: ", self.data.shape)
            print("Label shape: ", self.labels.shape)
            

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]