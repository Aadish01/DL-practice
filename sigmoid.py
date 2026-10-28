
import torch

x = torch.tensor([ -5.0, -1.0, 0.0, 1.0, 5.0 ])

result = torch.sigmoid(x)

print("Original:", x)
print("After Sigmoid:", result)
