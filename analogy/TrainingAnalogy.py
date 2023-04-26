import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 8) # Input layer with 12 input units and 8 hidden units
        self.fc2 = nn.Linear(8, 4)  # Output layer with 8 hidden units and 4 output units

    def forward(self, x):
        x = x.view(-1, 12) # Flatten the input tensor
        x = torch.relu(self.fc1(x)) # Apply ReLU activation to hidden layer
        x = self.fc2(x) # Output layer
        return x

# Generate random data with shape 2x2x3
torch.manual_seed(42) # Set random seed for reproducibility
data = torch.randn((2, 2, 3))
print(data)
print("----------------------")

# Instantiate the model, loss function, and optimizer
model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Convert the data to 1D tensor and forward pass
input_data = data.view(1, -1) # Flatten the data to 1D
output = model(input_data)

# Generate random target
target = torch.randn((1, 4))

# Training loop
for epoch in range(100): # Train for 100 epochs
    optimizer.zero_grad() # Reset gradients
    output = model(input_data) # Forward pass
    loss = criterion(output, target) # Compute loss
    loss.backward() # Backward pass
    optimizer.step() # Update weights

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}')

print('Training completed!')