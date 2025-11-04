import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the preprocessed data
DATA_DIR = "datasets"
print("Loading preprocessed data...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)
X_test_tensor = torch.from_numpy(X_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).float().to(device)

# Create DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class PositionalEncoding(nn.Module):
    """Add positional encoding to input sequences"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TransformerFusionModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=200,
    ):
        super(TransformerFusionModel, self).__init__()
        
        # Input projection layer to match d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, dim_feedforward // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dim_feedforward // 2, output_dim)
        
    def forward(self, src):
        # src shape: (batch, seq_len, input_dim)
        # Project input to d_model
        x = self.input_projection(src)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Use the output of the last time step
        x = x[:, -1, :]
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
        
        return x


# Model parameters
input_dim = X_train.shape[2]
output_dim = y_train.shape[1]
d_model = 128
nhead = 8
num_encoder_layers = 4
dim_feedforward = 512
dropout = 0.1

print(f"\n=== Model Configuration ===")
print(f"Input dimension: {input_dim}")
print(f"Output dimension: {output_dim}")
print(f"Model dimension: {d_model}")
print(f"Number of heads: {nhead}")
print(f"Encoder layers: {num_encoder_layers}")
print(f"Feedforward dimension: {dim_feedforward}")
print(f"Dropout: {dropout}")

model = TransformerFusionModel(
    input_dim=input_dim,
    output_dim=output_dim,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    max_seq_len=X_train.shape[1],
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Loss function - use Huber loss for robustness
criterion = nn.HuberLoss(delta=1.0)

# Optimizer with weight decay for regularization
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
num_epochs = 50
print(f"\n=== Starting Training ===")
print(f"Epochs: {num_epochs}, Batch size: {batch_size}")

best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 10

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate MAE
            mae = torch.mean(torch.abs(outputs - labels))
            val_mae += mae.item()
    
    val_loss /= len(test_loader)
    val_mae /= len(test_loader)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {epoch_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val MAE: {val_mae:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        model_path = os.path.join(DATA_DIR, "transformer_fusion_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'model_config': {
                'input_dim': input_dim,
                'output_dim': output_dim,
                'd_model': d_model,
                'nhead': nhead,
                'num_encoder_layers': num_encoder_layers,
                'dim_feedforward': dim_feedforward,
                'dropout': dropout,
            }
        }, model_path)
        print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\n=== Training Complete ===")
print(f"Best validation loss: {best_val_loss:.6f}")

# Final evaluation
model.eval()
with torch.no_grad():
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        outputs = model(inputs)
        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics
    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))
    rmse = np.sqrt(mse)
    
    print(f"\n=== Final Test Metrics ===")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")

print(f"\nModel saved to: {os.path.join(DATA_DIR, 'transformer_fusion_model.pth')}")
