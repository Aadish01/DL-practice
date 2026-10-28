
import torch
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

relu_result = F.relu(x)
gelu_result = F.gelu(x)

print("Original:", x)
print("ReLU:", relu_result)
print("GELU:", gelu_result)
