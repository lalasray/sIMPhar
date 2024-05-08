import os
import numpy as np
import torch
import torch.optim as optim  # Importing optimization module
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
        return imp,text

def get_data_files(data_path):
    data_files = []
    for file in os.listdir(data_path):
        if file.endswith('.pth'):
            data_files.append(os.path.join(data_path, file))
    return data_files

dataset_train = TriDataset(get_data_files(r"C:\Users\lalas\Desktop\n"))

batch_size = 32
data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

text_encoder = TextEncoder(embedding_dim=embedding_dim).to(device)
imp_encoder = ImpEncoder(embedding_dim=embedding_dim).to(device)

class BiModalModel(nn.Module):
    def __init__(self, text_encoder, imp_encoder):
        super(BiModalModel, self).__init__()
        self.text_encoder = text_encoder
        self.imp_encoder = imp_encoder
        
    def forward(self, text_input, imp_input):
        text_output = self.text_encoder(text_input.float())
        imp_output = self.imp_encoder(imp_input.float())
        text_output = F.normalize(text_output, p=2, dim=1)
        imp_output = F.normalize(imp_output, p=2, dim=1)

        return text_output, imp_output

model = BiModalModel(text_encoder, imp_encoder).to(device)
criterion = InfonceLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Define the number of epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # Initialize total_loss
    for imp, text in data_loader:
        optimizer.zero_grad()
        text_output, imp_output = model(text.to(device), imp.to(device))  # Send data to device
        loss = criterion(text_output, imp_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    total_loss /= len(data_loader)
    print(epoch," : ", total_loss)
