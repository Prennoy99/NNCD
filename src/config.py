"""
Configuration parameters for the Quarter-Car Model and System Matrices.
"""
import numpy as np
import torch

# ==========================================
# 1. DEFINE PARAMETERS (Quarter Car)
# ==========================================
# Standard values for a passenger car
ms = 250.0   # Sprung mass (kg) - 1/4 of car body
mu = 35.0    # Unsprung mass (kg) - Tire + Wheel
ks = 10000.0 # Suspension stiffness (N/m)
bs = 1000.0  # Suspension damping (N*s/m)
ku = 150000.0# Tire stiffness (N/m)

# ==========================================
# 2. STATE SPACE MATRICES (Numpy)
# ==========================================
# State x = [zs-zu,  zs_dot,  zu-zr,  zu_dot]^T
# x1: Suspension deflection
# x2: Sprung mass velocity
# x3: Tire deflection
# x4: Unsprung mass velocity

# Matrix A (System Matrix)
A = np.array([
    [0, 1, 0, -1],
    [-ks/ms, -bs/ms, 0, bs/ms],
    [0, 0, 0, 1],
    [ks/mu, bs/mu, -ku/mu, -bs/mu]
])

# Matrix B (Control Input Matrix - Force F)
B = np.array([
    [0],
    [1/ms],
    [0],
    [-1/mu]
])

# Matrix B_w (Disturbance Input Matrix - Road Velocity zr_dot)
B_w = np.array([
    [0],
    [0],
    [-1],
    [0]
])

# ==========================================
# 3. PyTorch Tensors for the Physics Loss
# ==========================================
A_tensor = torch.tensor(A, dtype=torch.float32)
B_tensor = torch.tensor(B, dtype=torch.float32)
B_w_tensor = torch.tensor(B_w, dtype=torch.float32)

# ==========================================
# 4. SIMULATION SETTINGS
# ==========================================
NUM_EPISODES = 50        # Total distinct simulations
TIME_PER_EPISODE = 10.0  # Duration of each simulation
DT = 0.02                # Sampling time (50Hz)
