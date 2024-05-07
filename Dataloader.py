import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TriDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        imp = data[0]
        text = data[1]
        return imp,text

def get_data_files(data_path):
    data_files = []
    for file in os.listdir(data_path):
        if file.endswith('.pt'):
            data_files.append(os.path.join(data_path, file))
    return data_files

dataset_train = TriDataset(get_data_files(r"D:\Dataset\Meta\simphar\sIMphar_synth\processed"))

batch_size = 32
data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

for imp,text in data_loader:
    print(imp.shape)
    print(text.shape)
    break
