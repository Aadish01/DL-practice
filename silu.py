
import torch
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

result = F.silu(x)

print("Original:", x)
print("After SiLU:", result)

