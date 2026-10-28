
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Create 4 samples, each containing 10 numbers
x = torch.randn(4, 10)

print("Input shape:", x.shape)
print(x)

# 2. Apply activation functions
relu_out = F.relu(x)
gelu_out = F.gelu(x)
sigmoid_out = torch.sigmoid(x)
swish_out = F.silu(x)

print("\nReLU shape:", relu_out.shape)
print("GELU shape:", gelu_out.shape)
print("Sigmoid shape:", sigmoid_out.shape)
print("Swish shape:", swish_out.shape)

# 3. Convert 5 scores into probabilities for each sample
logits = torch.randn(4, 5)
probs = F.softmax(logits, dim=1)

print("\nLogits:")
print(logits)

print("\nSoftmax probabilities:")
print(probs)
print("Probability sums:", probs.sum(dim=1))

# 4. Define a neural network
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.GELU(),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 5),
)

# 5. Pass input through the model
output = model(x)

print("\nModel output shape:", output.shape)
print(output)
