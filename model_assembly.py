import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pkl
import sys

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_Assembly_NormalizedAdj():
    # 0-indexed edges
    edges = [
        (5, 17), (17, 18), (18, 19), (19, 4),  # pinky
        (5, 14), (14, 15), (15, 16), (16, 3),  # ring
        (5, 11), (11, 12), (12, 13), (13, 2),  # middle
        (5, 8), (8, 9), (9, 10), (10, 1),  # index
        (5, 6), (6, 7), (7, 0),  # thumb
    ]
    
    # If the partitioning is uni-labeling
    # Init with self-loops
    # A = torch.eye(21,dtype=torch.float)

    # For distance partitioning, return without self loop
    A = torch.zeros(21,21,dtype=torch.float)

    # Add edges in both direction
    for (i,j) in edges:
        # 0-indexed already
        A[i,j] = 1
        A[j,i] = 1

    # degree matrix
    degree = torch.sum(A, dim=1) + 0.001 ### Adding 0.001 is to avoid empty rows in A. See Eq.10
                                         ### It's not really necessary if the skeleton graph is connected.
    degree_matrix = torch.diag(degree)

    # Calculate the normalized adjacency matrix
    degree_matrix_sqrt = torch.sqrt(torch.inverse(degree_matrix)) # D^(-1/2)
    A_norm = torch.matmul(torch.matmul(degree_matrix_sqrt, A), degree_matrix_sqrt)

    return A_norm.to(device)

class ST_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A_norm, temporal_stride=1):
        """
        The paper always uses temporal kernel size 9, while the stride can be 2 for 4th and 7th unit
        Working on shape like (C,V,T).
        """
        super(ST_GCN_unit, self).__init__()
        
        self.A_Norm = A_norm
        # Trainable mask
        self.M = nn.Parameter(torch.ones_like(A_norm), requires_grad=True)

        # Partition-type: Uni-labeling. 
        # We can do the aggregation first with einsum on C,V,T feature mat and A_norm
        # Then use 1x1 conv to get output channels
        # The variable name `spatial_conv` might be misleading, it's just a 1x1 conv for each node after aggregation
        
        # Distance Partitioning in the paper has two gcns, one for self and another for neighbors
        self.spatial_conv_neighbor = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        self.spatial_conv_self = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))

        # Temporal conv will assume that the input is in shape (b,c,v,t)
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1,9), stride=(1,temporal_stride), padding=(0,4)) # Padding for matching size

        # Residual connection for each unit
        if in_channels==out_channels:
            # Directly pass
            self.res_layer = nn.Identity()
            
        else:
            # Need to adjust number of channels
            if in_channels==3: # First unit. Change #channels but temporal stride=1 in the block
                self.res_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
            else: # 4th and 7th unit. Changes #channels and also temporal stride=2
                self.res_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,temporal_stride))
        
        # Adding some batchNorm layers
        self.bn1 = nn.BatchNorm2d(num_features=out_channels) # will apply after spatial conv
        self.bn2 = nn.BatchNorm2d(num_features=out_channels) # will apply after temporal conv
        
    def forward(self, x):
        # (batchsize,C,V,T) feature matrix x
        res = self.res_layer(x) # Keeping residual info for using at the end

        # Construct adjacency matrices (self and neighbor) with trainable masks.
        # Use modified adjacency matrix with learnable edge importance. Elementwise multiplication.
        ## `Neighbor`
        A = torch.mul(self.A_Norm, self.M)
        # Adj matrix is simply (V,V) now. Repeat for all channels
        A = A.unsqueeze(0)
        # x.shape[1] is the number of channels
        A = torch.tile(A, dims=(x.shape[1],1,1)) # Shape becomes (C,V,V)
        ## For now, doing the same for `self`. However, it will occupy extra space. Later, use diagonal(M) directly
        A_self = torch.diag(torch.diagonal(self.M)) # Take the diagonal entries of the mask
        A_self = A_self.unsqueeze(0)
        A_self = torch.tile(A_self, dims=(x.shape[1],1,1)) # Shape becomes (C,V,V)
        
        # Permute x for multiplication
        x = x.permute(0,1,3,2).contiguous() # Shape becomes (batchsize,C,T,V)
        
        # Distance partitioning gcns
        neighbor_agg = torch.einsum('bctv,cvu->bctu', x, A) # Matrix multiplication to aggregate neighbor feat
        neighbor_agg = self.spatial_conv_neighbor(neighbor_agg) # 1x1 convolution for each node
        
        self_feat = torch.einsum('bctv,cvu->bctu', x, A_self) # Matrix multiplication to masked self feature
        self_feat = self.spatial_conv_self(self_feat) # 1x1 convolution for each node
        
        # update x
        x = self_feat + neighbor_agg

        x = F.relu(x, inplace=True)
        x = self.bn1(x)

        x = x.permute(0,1,3,2).contiguous() # Back to shape (batchsize,C,V,T) for temporal conv
        x = self.temporal_conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)

        # Adding dropout layer(0.5) at the end as mentioned in the paper
        x = F.dropout(x, p=0.5)

        # Add residual connection
        x = F.relu(x + res, inplace=True)
        return x

class ST_GCN(nn.Module):
    def __init__(self, num_classes):
        super(ST_GCN, self).__init__()
        
        # Adj matrix. Finding normalized adjacency once for all
        self.A_Norm = get_Assembly_NormalizedAdj() # Adjacency matrix for Assembly

        # define all layers
        self.bn = nn.BatchNorm2d(num_features=3) # Input channel-> 3d coordinate

        # Use modulelist and loop later
        # self.st_gcn1 = ST_GCN_unit(3,64,self.A_Norm)

        self.st_gcn_networks = nn.ModuleList()
        in_channels = [3,64,64,64,128,128,128,256,256]
        out_channels = [64,64,64,128,128,128,256,256,256]
        strides = [1,1,1,2,1,1,2,1,1] # 4th and 7th block for pooling

        for i in range(len(in_channels)):
            self.st_gcn_networks.append(ST_GCN_unit(in_channels[i], out_channels[i], self.A_Norm, temporal_stride=strides[i]))
        
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # We want to process individual persons separately in graph conv
        # Reason: There is no meaningful connection between two persons to consider as edge
        # Double the batchsize by decouplong two persons
        batch_size, num_channels, num_frames, num_joints, num_persons = x.shape
        x = x.permute(0,4,1,2,3).contiguous()
        x = x.view(batch_size*num_persons, num_channels, num_frames, num_joints)
        x = self.bn(x)
        # The paper treats each frame as (C,V,T). So permuting to match.
        x = x.permute(0,1,3,2).contiguous()
        
        # lid = 0
        for layer in self.st_gcn_networks:
            x = layer(x)
            # print(lid, ": ", x.shape)
            # lid += 1
        
        num_channels_new = x.shape[1]
        num_frames_new = x.shape[3]
        x = x.view(batch_size, num_persons, num_channels_new, num_joints, num_frames_new)
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(batch_size, num_channels_new, -1)
        x = x.mean(2) # Global Avg Pool
        out = self.fc(x)
        return out.to(device)


if __name__=="__main__":
    # test model code
    mdl = ST_GCN(1380).to(device)

    batch_size = 32
    num_channels = 3 # Input (x,y,z) 3d coord
    num_frames = 200 # Data either repeated or trimmed to adjust
    num_joints = 21 # For Assembly101, 21 joints
    num_persons = 2 # 2 hands in the scene

    x = torch.randn(batch_size, num_channels, num_frames, num_joints, num_persons).to(device)
    
    out = mdl(x)
    print(out.shape)
    num_params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {num_params}")