import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadClassifier(nn.Module):
    def __init__(self, input_size, num_classes, cnn_channels=128, cnn_kernel_size=1, num_heads=8):
        super(MultiheadClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=cnn_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.multihead_attention = nn.MultiheadAttention(embed_dim=cnn_channels, num_heads=num_heads)
        
        self.fc = nn.Linear(cnn_channels, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.permute(2, 0, 1)  
        attended_features, _ = self.multihead_attention(features, features, features)
        attended_features = attended_features.permute(1, 2, 0)  
        output = self.fc(attended_features.mean(dim=2))
        return output
