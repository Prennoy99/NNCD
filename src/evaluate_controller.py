import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ==========================================
# 1. SETUP & MATRICES (THE "REAL WORLD")
# ==========================================
MAX_FORCE = 2500.0

# Physical parameters
ms, mu = 250.0, 35.0
ks, bs = 10000.0, 1000.0
ku = 150000.0

A = np.array([
    [0, 1, 0, -1],
    [-ks/ms, -bs/ms, 0, bs/ms],
    [0, 0, 0, 1],
    [ks/mu, bs/mu, -ku/mu, -bs/mu]
])
B = np.array([[0], [1/ms], [0], [-1/mu]])
B_w = np.array([[0], [0], [-1], [0]])

# Paths
TEST_DATA_FILE = os.path.join('datasets', 'unseen_road_test_data.csv')
CONTROLLER_MODEL_PATH = os.path.join('trained_controller_models', 'controller_stable_lqr.pth')

# ==========================================
# 2. LOAD THE TRAINED CONTROLLER
# ==========================================
class ControllerNet(nn.Module):
    def __init__(self):
        super(ControllerNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1), nn.Tanh()
        )
    def forward(self, x): return self.net(x) * MAX_FORCE

print("Loading Stable LQR Controller...")
controller = ControllerNet()
controller.load_state_dict(torch.load(CONTROLLER_MODEL_PATH))
controller.eval()

# ==========================================
# 3. LOAD UNSEEN TEST DATA
# ==========================================
print(f"Loading unseen test data from {TEST_DATA_FILE}...")
df = pd.read_csv(TEST_DATA_FILE)

# Pick an episode to test
TEST_EPISODE = 0
if 'episode_id' in df.columns:
    ep_data = df[df['episode_id'] == TEST_EPISODE].reset_index(drop=True)
else:
    ep_data = df

t_eval = ep_data['time'].values if 'time' in ep_data.columns else ep_data['t'].values
w_data = ep_data['w'].values
road_type = ep_data['road_type'].iloc[0] if 'road_type' in ep_data.columns else "Unseen Road"

# ==========================================
# 4. SIMULATION FUNCTIONS (solve_ivp format: t, x)
# ==========================================
def get_w_continuous(t):
    return np.interp(t, t_eval, w_data)

def passive_dynamics(t, x): 
    w = get_w_continuous(t)
    x_dot = A @ x + (B_w * w).flatten()
    return x_dot

def active_dynamics(t, x):  
    w = get_w_continuous(t)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        u = controller(x_tensor).numpy()[0, 0]
    x_dot = A @ x + (B * u).flatten() + (B_w * w).flatten()
    return x_dot

# ==========================================
# 5. RUN SIMULATIONS (Modern solve_ivp)
# ==========================================
print(f"Simulating Episode {TEST_EPISODE} (Road: {road_type})...")
x0 = np.zeros(4)

# Limit max_step to capture high frequency dynamics
dt_val = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 0.02
sol_passive = solve_ivp(passive_dynamics, [t_eval[0], t_eval[-1]], x0, t_eval=t_eval, method='RK45', max_step=dt_val)
sol_active = solve_ivp(active_dynamics, [t_eval[0], t_eval[-1]], x0, t_eval=t_eval, method='RK45', max_step=dt_val)

x_passive = sol_passive.y.T
x_active = sol_active.y.T

# Calculate Accelerations
accel_passive = A[1,0]*x_passive[:,0] + A[1,1]*x_passive[:,1] + A[1,3]*x_passive[:,3]

u_active = np.zeros_like(t_eval)
for i in range(len(t_eval)):
    x_tensor = torch.tensor(x_active[i], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        u_active[i] = controller(x_tensor).numpy()[0,0]

accel_active = A[1,0]*x_active[:,0] + A[1,1]*x_active[:,1] + A[1,3]*x_active[:,3] + (1/ms)*u_active

# ==========================================
# 6. VISUALIZE PERFORMANCE
# ==========================================
plt.figure(figsize=(14, 10))
plt.suptitle(f"Passive vs. Neural LQR Suspension | Unseen Road: {road_type}", fontsize=16)

# 1. Ride Comfort (Sprung Mass Acceleration)
plt.subplot(2, 2, 1)
plt.plot(t_eval, accel_passive, 'k-', alpha=0.6, label='Passive (No Control)')
plt.plot(t_eval, accel_active, 'b-', linewidth=2, label='Neural Controller')
plt.title("Ride Comfort (Body Acceleration $\ddot{z}_s$) - LOWER IS BETTER")
plt.ylabel("[m/s²]")
plt.grid(True)
plt.legend()

# 2. Handling (Tire Deflection)
plt.subplot(2, 2, 2)
plt.plot(t_eval, x_passive[:, 2], 'k-', alpha=0.6, label='Passive')
plt.plot(t_eval, x_active[:, 2], 'g-', linewidth=2, label='Neural Controller')
plt.title("Handling (Tire Deflection $z_u - z_r$) - LOWER IS BETTER")
plt.ylabel("[m]")
plt.grid(True)
plt.legend()

# 3. Suspension Safety (Suspension Deflection)
plt.subplot(2, 2, 3)
plt.plot(t_eval, x_passive[:, 0], 'k-', alpha=0.6, label='Passive')
plt.plot(t_eval, x_active[:, 0], 'r-', linewidth=2, label='Neural Controller')
plt.title("Safety (Suspension Deflection $z_s - z_u$)")
plt.xlabel("Time [s]")
plt.ylabel("[m]")
plt.grid(True)

# 4. Actuator Effort
plt.subplot(2, 2, 4)
plt.plot(t_eval, u_active, 'm-', linewidth=2, label='Control Force (u)')
plt.axhline(MAX_FORCE, color='r', linestyle='--', label='Max Limit')
plt.axhline(-MAX_FORCE, color='r', linestyle='--')
plt.title("Actuator Effort")
plt.xlabel("Time [s]")
plt.ylabel("[N]")
plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()
os.makedirs('images', exist_ok=True)
plt.savefig(os.path.join('images', 'controller_evaluation.png'))
print("Evaluation plot saved to images/controller_evaluation.png")
plt.show()
