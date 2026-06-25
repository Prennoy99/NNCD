"""
Evaluation script to load a trained model and compare it against the true physics simulation.
"""
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from src.model import QuarterCarNet

def evaluate_model(model_path='model_pinn.pth', data_path='quarter_car_data.csv', output_img='evaluation.png'):
    """Simulates trajectory using the NN and compares with the ground truth from data_path."""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
        
    print(f"Loading model '{model_path}' and data '{data_path}'...")
    
    # Load Model
    model = QuarterCarNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load Data
    df = pd.read_csv(data_path)
    # If the data has multiple episodes, we might just take the first one or a specific one.
    # For now, let's assume it's the standard file or we just take the first episode if 'episode_id' exists.
    if 'episode_id' in df.columns:
        df = df[df['episode_id'] == df['episode_id'].unique()[0]].copy()

    t = df['t'].values if 't' in df.columns else df['time'].values
    dt = t[1] - t[0]

    X_true = df[['x1', 'x2', 'x3', 'x4']].values
    U_data = df[['u']].values
    W_data = df[['w']].values

    print("Simulating trajectory using the Neural Network...")
    X_pred = np.zeros((len(t), 4))
    X_pred[0] = X_true[0]

    # Euler Integration
    for i in range(len(t) - 1):
        x_current = torch.tensor(X_pred[i], dtype=torch.float32).unsqueeze(0)
        u_current = torch.tensor([[U_data[i][0]]], dtype=torch.float32)
        w_current = torch.tensor([[W_data[i][0]]], dtype=torch.float32)

        nn_input = torch.cat((x_current, u_current, w_current), dim=1)

        with torch.no_grad():
            x_dot_pred = model(nn_input).numpy()[0]

        X_pred[i+1] = X_pred[i] + (x_dot_pred * dt)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Neural Network vs Ground Truth Physics\nData: {data_path}", fontsize=16)

    plt.subplot(2, 2, 1)
    plt.plot(t, X_true[:, 0], 'k-', linewidth=2, label="True Physics (ODE)")
    plt.plot(t, X_pred[:, 0], 'r--', linewidth=2, label="Neural Network")
    plt.title("State 1: Suspension Deflection ($z_s - z_u$)")
    plt.ylabel("[m]")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t, X_true[:, 1], 'k-', linewidth=2, label="True Physics (ODE)")
    plt.plot(t, X_pred[:, 1], 'b--', linewidth=2, label="Neural Network")
    plt.title("State 2: Body Velocity ($\dot{z}_s$)")
    plt.ylabel("[m/s]")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t, X_true[:, 2], 'k-', linewidth=2, label="True Physics (ODE)")
    plt.plot(t, X_pred[:, 2], 'g--', linewidth=2, label="Neural Network")
    plt.title("State 3: Tire Deflection ($z_u - z_r$)")
    plt.xlabel("Time [s]")
    plt.ylabel("[m]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"Evaluation complete. Plot saved to {output_img}")

if __name__ == "__main__":
    evaluate_model()
