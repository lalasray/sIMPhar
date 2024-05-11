import torch
from Multihead import MultiheadClassifier

batch_size = 32
input_size = 2048
random_input = torch.randn(batch_size, input_size)
print("Shape of random_input:", random_input.shape)  # Verify the shape

num_classes = 10
model = MultiheadClassifier(input_size=input_size, num_classes=num_classes)

# Now, let's check the forward pass
output = model(random_input)
print("Output shape:", output.shape)
