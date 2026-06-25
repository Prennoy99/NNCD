import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# ==========================================
# 1. SETUP & HYPERPARAMETERS
# ==========================================
MAX_FORCE = 2500.0
DT = 0.02
ROLLOUT_STEPS = 50   # 1 second of simulation per training step
EPOCHS = 800         # Allow the stable RK4 to converge
LEARNING_RATE = 0.001

# --- Physical Normalization Constants ---
MAX_ACCEL = 10.0     # ~1G of acceleration (m/s²)
MAX_SUSP = 0.05      # 5 cm max suspension travel (m)
MAX_TIRE = 0.02      # 2 cm max tire deflection (m)

# --- Normalized Cost Weights (LQR Style) ---
W_ACCEL = 1.0        # Primary goal: Ride Comfort
W_SUSP = 0.5         # Safety: Don't bottom out the springs
W_TIRE = 0.1         # Handling: Keep the tire on the road
W_CONTROL = 0.1      # Energy: Penalize using max force constantly to prevent hacking

# Paths
DATA_FILE = os.path.join('datasets', 'quarter_car_robust_with_Actuator_constraint.csv')
TWIN_MODEL_PATH = os.path.join('trained_models', 'model_pure_data_normalized.pth')
SCALER_IN_PATH = os.path.join('trained_models', 'scaler_in.pkl')
SCALER_OUT_PATH = os.path.join('trained_models', 'scaler_out.pkl')
CONTROLLER_SAVE_PATH = os.path.join('trained_controller_models', 'controller_stable_lqr.pth')

# ==========================================
# 2. LOAD DIGITAL TWIN & SCALERS
# ==========================================
class QuarterCarNet(nn.Module):
    def __init__(self):
        super(QuarterCarNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)
        )
    def forward(self, x): return self.net(x)

print("Loading frozen Digital Twin...")
digital_twin = QuarterCarNet()
digital_twin.load_state_dict(torch.load(TWIN_MODEL_PATH))
digital_twin.eval()
for param in digital_twin.parameters():
    param.requires_grad = False # FREEZE THE TWIN

# Load Scalers and convert to PyTorch tensors
scaler_in = joblib.load(SCALER_IN_PATH)
scaler_out = joblib.load(SCALER_OUT_PATH)

in_mean = torch.tensor(scaler_in.mean_, dtype=torch.float32)
in_scale = torch.tensor(scaler_in.scale_, dtype=torch.float32)
out_mean = torch.tensor(scaler_out.mean_, dtype=torch.float32)
out_scale = torch.tensor(scaler_out.scale_, dtype=torch.float32)

# ==========================================
# 3. DEFINE THE CONTROLLER
# ==========================================
class ControllerNet(nn.Module):
    def __init__(self):
        super(ControllerNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x) * MAX_FORCE

controller = ControllerNet()
optimizer = optim.Adam(controller.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. PREPARE ROAD DISTURBANCES
# ==========================================
print("Loading road disturbances from data...")
df = pd.read_csv(DATA_FILE)
w_data = torch.tensor(df['w'].values, dtype=torch.float32).view(-1, 1)

# ==========================================
# 5. HELPER: DIGITAL TWIN DERIVATIVE PREDICTOR
# ==========================================
def predict_derivative(x_t, u_t, w_t):
    """Handles normalization, asks twin for derivative, and denormalizes."""
    raw_twin_input = torch.cat((x_t, u_t, w_t), dim=1)
    norm_twin_input = (raw_twin_input - in_mean) / in_scale
    norm_x_dot = digital_twin(norm_twin_input)
    raw_x_dot = (norm_x_dot * out_scale) + out_mean
    return raw_x_dot

# ==========================================
# 6. TRAINING LOOP (BPTT with RK4)
# ==========================================
print("Starting Controller Training (Stable RK4 + Normalized LQR Mode)...")
loss_history = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # Start car at rest
    x_current = torch.zeros((1, 4), dtype=torch.float32)
    start_idx = np.random.randint(0, len(w_data) - ROLLOUT_STEPS)
    total_cost = 0.0

    for step in range(ROLLOUT_STEPS):
        # 1. Controller decides force
        u_current = controller(x_current)
        w_current = w_data[start_idx + step].view(1, 1)

        # 2. Differentiable Runge-Kutta 4 (RK4) Integration
        # We assume u and w are constant over this specific DT (Zero-Order Hold)
        k1 = predict_derivative(x_current, u_current, w_current)
        k2 = predict_derivative(x_current + 0.5 * DT * k1, u_current, w_current)
        k3 = predict_derivative(x_current + 0.5 * DT * k2, u_current, w_current)
        k4 = predict_derivative(x_current + DT * k3, u_current, w_current)

        # 3. Calculate Normalized Costs
        # Sprung mass acceleration is just the derivative of x2 (velocity) at the start of the step (k1)
        accel = k1[:, 1]
        susp_def = x_current[:, 0]
        tire_def = x_current[:, 2]

        # Scale the physics before squaring!
        cost_accel = W_ACCEL * (accel / MAX_ACCEL)**2
        cost_susp = W_SUSP * (susp_def / MAX_SUSP)**2
        cost_tire = W_TIRE * (tire_def / MAX_TIRE)**2
        cost_control = W_CONTROL * (u_current / MAX_FORCE)**2

        step_cost = cost_accel + cost_susp + cost_tire + cost_control
        total_cost += step_cost

        # 4. Step the environment forward using the RK4 weighted average
        x_current = x_current + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Backpropagate through time
    loss = total_cost / ROLLOUT_STEPS
    loss.backward()

    # Clip gradients to be extra safe against exploding gradients
    torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=5.0)

    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Rollout Cost: {loss.item():.4f}")

# Save the Controller
os.makedirs(os.path.dirname(CONTROLLER_SAVE_PATH), exist_ok=True)
torch.save(controller.state_dict(), CONTROLLER_SAVE_PATH)
print(f"Training complete. Saved as '{CONTROLLER_SAVE_PATH}'")

# Plot training curve
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.title("Controller Training Curve (RK4 + Normalized Cost)")
plt.xlabel("Epoch")
plt.ylabel("Averaged Rollout Cost")
plt.grid(True)
plt.savefig(os.path.join('images', 'controller_training_curve.png'))
print("Training curve saved to images/controller_training_curve.png")
