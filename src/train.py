"""
Training script for the QuarterCarNet using a Physics-Informed Neural Network (PINN) approach.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from src.model import QuarterCarNet
from src.config import A_tensor, B_tensor, B_w_tensor

def train_model(use_physics_loss=True, epochs=1000, lr=0.005, lambda_physics=1.0, data_file='quarter_car_data.csv'):
    """Trains the Neural Network and saves the weights."""
    print(f"Loading data from {data_file}...")
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run data_generation.py first.")
        return
        
    df = pd.read_csv(data_file)
    t = df['t'].values
    dt = t[1] - t[0]

    # Extract states and inputs
    X_data = df[['x1', 'x2', 'x3', 'x4']].values
    U_data = df[['u']].values
    W_data = df[['w']].values

    # True derivatives via numerical differentiation (Ground Truth)
    X_dot_data = np.gradient(X_data, dt, axis=0)

    # Convert to Tensors
    x_tensor = torch.tensor(X_data, dtype=torch.float32)
    u_tensor = torch.tensor(U_data, dtype=torch.float32)
    w_tensor = torch.tensor(W_data, dtype=torch.float32)
    x_dot_true = torch.tensor(X_dot_data, dtype=torch.float32)

    nn_inputs = torch.cat((x_tensor, u_tensor, w_tensor), dim=1)

    model = QuarterCarNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"Starting training... (Physics Loss: {use_physics_loss})")
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Predict derivative
        x_dot_pred = model(nn_inputs)
        
        # Data loss
        loss_data = loss_fn(x_dot_pred, x_dot_true)
        
        # Physics loss
        if use_physics_loss:
            physics_expected_x_dot = torch.matmul(x_tensor, A_tensor.T) + \
                                     torch.matmul(u_tensor, B_tensor.T) + \
                                     torch.matmul(w_tensor, B_w_tensor.T)
            loss_physics = loss_fn(x_dot_pred, physics_expected_x_dot)
        else:
            loss_physics = torch.tensor(0.0)
            
        total_loss = loss_data + (lambda_physics * loss_physics)
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f} | Data Loss: {loss_data.item():.6f} | Phys Loss: {loss_physics.item():.6f}")

    # Save Model
    save_name = "model_pinn.pth" if use_physics_loss else "model_pure_data.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Training complete. Model saved as '{save_name}'")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Total Training Loss")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Training Curve (Physics Included: {use_physics_loss})")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"training_curve_{'pinn' if use_physics_loss else 'data'}.png")
    plt.close()
    print("Saved training curve plot.")

if __name__ == "__main__":
    train_model(use_physics_loss=True)
