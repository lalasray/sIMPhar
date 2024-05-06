import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

def generate_data(num_samples, seq_len, input_size, output_size):
    inputs = torch.randn(num_samples, seq_len, input_size)
    targets = torch.randn(num_samples, output_size)
    return inputs, targets

class BiLSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, ff_dim, output_size):
        super(BiLSTMTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer_encoder(lstm_out) 
        output = self.fc(transformer_out[:, -1, :])
        
        return output

input_size = 10
hidden_size = 64
num_layers = 2
num_heads = 4
ff_dim = 128
output_size = 2
seq_len = 32
num_samples = 1000
batch_size = 32
num_epochs = 200

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, outputs, targets):
        outputs_normalized = F.normalize(outputs, p=2, dim=-1)
        targets_normalized = F.normalize(targets, p=2, dim=-1)
        cosine_sim = torch.sum(outputs_normalized * targets_normalized, dim=-1)
        loss = 1 - cosine_sim.mean()
        
        return loss

model = BiLSTMTransformer(input_size, hidden_size, num_layers, num_heads, ff_dim, output_size)
criterion = CosineSimilarityLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    total_loss = 0
    for batch in range(num_samples // batch_size):
        inputs, targets = generate_data(batch_size, seq_len, input_size, output_size)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / (num_samples // batch_size):.4f}')