import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the preprocessed data
print("Loading preprocessed data...")
X_train = np.load("/home/wahid/dev/nanofuse/datasets/X_train.npy")
y_train = np.load("/home/wahid/dev/nanofuse/datasets/y_train.npy")
X_test = np.load("/home/wahid/dev/nanofuse/datasets/X_test.npy")
y_test = np.load("/home/wahid/dev/nanofuse/datasets/y_test.npy")

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)
X_test_tensor = torch.from_numpy(X_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).float().to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Transformer model
class TransformerFusionModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerFusionModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        x = self.transformer_encoder(src)
        # Use the output of the last time step
        x = x[:, -1, :]
        return self.fc(x)

# Model parameters
input_dim = X_train.shape[2]
output_dim = y_train.shape[1]
nhead = 1
num_encoder_layers = 3
dim_feedforward = 256
dropout = 0.1

model = TransformerFusionModel(input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward, dropout).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25
print("Starting model training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    print(f"Validation Loss: {val_loss:.4f}")

print("Finished Training")

# Save the model
model_path = "/home/wahid/dev/nanofuse/transformer_fusion_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
