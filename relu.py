
import torch
import torch.nn.functional as F

x = torch.tensor([2.0, -3.0, 5.0, -1.0])

result = F.relu(x)

print("Original:", x)
print("After ReLU:", result)
