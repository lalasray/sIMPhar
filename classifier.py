import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from text_enc import TextEncoder
from imp_enc import ImpEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from config import embedding_dim
from loss import InfonceLoss
import torch.nn as nn
import torch.nn.functional as F

class TriDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        mag = data['simp_mag']
        phase = data['simp_phase']
        imp = torch.stack((mag, phase), dim=0)
        text = data['embedding'].reshape(-1)
        text= torch.tensor(text)
        aclass = data['aclass'].float() 
        aclass = aclass.view(1)
        
        return imp,text,aclass

def get_data_files(data_path):
    data_files = []
    for file in os.listdir(data_path):
        if file.endswith('.pth'):
            data_files.append(os.path.join(data_path, file))
    return data_files

dataset_train = TriDataset(get_data_files(r"C:\Users\lalas\Desktop\n\out\real"))

batch_size = 32
data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

unique_aclass = set()
for imp, text, aclass in dataset_train:
    unique_aclass.add(aclass.item())

print("Unique aclass values:", unique_aclass)