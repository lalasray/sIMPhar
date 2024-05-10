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

# Define your decoder for classification
class ClassifierDecoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassifierDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained model
model_checkpoint_path = "model_checkpoint_.pt"
pretrained_model = BiModalModel(text_encoder, imp_encoder).to(device)
checkpoint = torch.load(model_checkpoint_path)
pretrained_model.load_state_dict(checkpoint['model_state_dict'])

# Add the classifier decoder
num_classes = ...  # Define the number of classes for your classification task
classifier_decoder = ClassifierDecoder(embedding_dim, num_classes).to(device)

# Define the complete model including the pre-trained model and the classifier decoder
class FineTunedModel(nn.Module):
    def __init__(self, pretrained_model, classifier_decoder):
        super(FineTunedModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier_decoder = classifier_decoder
        
    def forward(self, text_input, imp_input):
        text_output, imp_output = self.pretrained_model(text_input, imp_input)
        classification_logits = self.classifier_decoder(text_output)
        return classification_logits

# Initialize the fine-tuned model
fine_tuned_model = FineTunedModel(pretrained_model, classifier_decoder).to(device)

# Define criterion and optimizer for fine-tuning
criterion = ...  # Define your classification loss function
optimizer = optim.Adam(fine_tuned_model.parameters(), lr=0.001)

# Fine-tune the model
num_epochs = 10  # Define the number of epochs for fine-tuning
for epoch in range(num_epochs):
    fine_tuned_model.train()
    total_loss = 0
    for imp, text in data_loader:
        optimizer.zero_grad()
        text_output = fine_tuned_model(text.to(device), imp.to(device))
        loss = criterion(text_output, ...)  # Pass your target labels here
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    total_loss /= len(data_loader)
    print("Epoch:", epoch, "Loss:", total_loss)

# Save the fine-tuned model
torch.save({
    'model_state_dict': fine_tuned_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "fine_tuned_model_checkpoint.pt")
