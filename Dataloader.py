import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#batch_size = 32
#data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#for pose, imu, text in data_loader:
    #print(pose.shape, imu.shape, text.shape)
    #break
