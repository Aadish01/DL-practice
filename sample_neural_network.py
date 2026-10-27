
import torch
import torch.nn as nn

# Build the neural network
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

# Four examples, each containing two input features
x = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

# Make predictions
output = model(x)

print(output)


# Output:  tensor([[0.5428], [0.5498], [0.5322], [0.5387]], grad_fn=<SigmoidBackward0>)
# But why are all the predictions around 0.5?
# Because the network hasn't learned anything yet. Its weights and biases were initialized with starting values, we haven't trained it.
