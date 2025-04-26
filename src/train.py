import torch
import torch.nn as nn
import torch.optim as optim
from src.model import BreakoutPredictorMLP

"""
train.py

Training loop for the AFL Fantasy breakout MLP model.
Architecture, dropout, epochs, and learning rate are controlled via config.yaml.
Includes:
- Early stopping (patience-based).
- Learning rate scheduler (ReduceLROnPlateau).
- BCEWithLogitsLoss (stable binary classification).
"""

def train_model(train_loader, val_loader, input_dim, config):
    hidden_dims = config['model']['hidden_dims']
    input_dropout = config['model']['input_dropout']
    output_dropout = config['model']['output_dropout']
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    patience = config['training'].get('patience', 25)  # Default patience if not in config

    model = BreakoutPredictorMLP(input_dim, hidden_dims, input_dropout, output_dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation loss for scheduler and early stopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X).squeeze()
                loss = criterion(outputs, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1} (patience {patience})")
                break

    return model