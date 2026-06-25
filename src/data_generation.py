"""
Dataset generation for the Quarter-Car Model.
Includes standard data, robust randomized episodes, and unseen road profiles.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
from src.config import A, B, B_w, TIME_PER_EPISODE, DT, NUM_EPISODES

# ==========================================
# 1. INPUT FUNCTIONS
# ==========================================
def basic_road_profile(t):
    """Simple bump followed by rough road."""
    if 1.0 < t < 1.5:
        return 0.1
    else:
        return 0.05 * np.sin(2 * np.pi * 5 * t)

def basic_control_input(t):
    """Small active force oscillating."""
    return 10.0 * np.sin(2 * np.pi * 0.5 * t)

def get_random_road_profile(t_max, dt, profile_type):
    """Generates a specific random road profile function for an episode."""
    t_vals = np.arange(0, t_max, dt)
    if profile_type == 'bump':
        bump_time = np.random.uniform(2, 8)
        w_vals = np.exp(-50*(t_vals - bump_time)**2) * 0.5 
    elif profile_type == 'rough':
        w_vals = np.random.normal(0, 0.02, size=len(t_vals))
    elif profile_type == 'sine':
        freq = np.random.uniform(1, 8) 
        w_vals = 0.05 * np.sin(2 * np.pi * freq * t_vals)
    else:
        w_vals = np.zeros_like(t_vals)
    return t_vals, w_vals

def get_random_control_input(t_vals):
    """Generates random control inputs to excite the system with actuator constraints."""
    MAX_FORCE = 2500.0
    f1 = np.random.uniform(0.1, 2.0)
    f2 = np.random.uniform(0.1, 2.0)
    u_vals = 1000.0 * np.sin(2*np.pi*f1*t_vals) + 2000.0 * np.cos(2*np.pi*f2*t_vals)
    u_vals = np.clip(u_vals, -MAX_FORCE, MAX_FORCE)
    return u_vals

def generate_unseen_profiles(t_vals):
    """Generates complex, unseen structural profiles."""
    profiles = {}
    
    # Profile 1: Smooth Pothole
    profiles['pothole'] = -1.2 * 10 * np.exp(-100 * (t_vals - 5.0)**2)
    
    # Profile 2: Trapezoidal Step
    trap_step = np.zeros_like(t_vals)
    rise_mask = (t_vals >= 4.5) & (t_vals <= 4.7)
    flat_mask = (t_vals >= 4.7) & (t_vals <= 5.2)
    fall_mask = (t_vals >= 5.2) & (t_vals <= 5.4)
    trap_step[rise_mask] = np.linspace(0.0, 0.8, np.sum(rise_mask))
    trap_step[flat_mask] = 0.8
    trap_step[fall_mask] = np.linspace(0.8, 0.0, np.sum(fall_mask))
    profiles['trapezoidal_step'] = trap_step
    
    # Profile 3: Trapezoidal Pothole
    trap_pothole = np.zeros_like(t_vals)
    rise_mask = (t_vals >= 5.5) & (t_vals <= 5.7)
    flat_mask = (t_vals >= 5.7) & (t_vals <= 6.7)
    fall_mask = (t_vals >= 6.7) & (t_vals <= 6.9)
    trap_pothole[rise_mask] = np.linspace(0.0, -1.5, np.sum(rise_mask))
    trap_pothole[flat_mask] = -1.5
    trap_pothole[fall_mask] = np.linspace(-1.5, 0.0, np.sum(fall_mask))
    profiles['trapezoidal_pothole'] = trap_pothole
    
    # Profile 4: Complex Step Pair
    step_pair = np.zeros_like(t_vals)
    step1_mask = (t_vals >= 4.0) & (t_vals <= 5.0)
    step_pair[step1_mask] = -0.8
    jump = 1.5 * np.exp(-100 * (t_vals - 6.0)**2)
    profiles['complex_step_pair'] = step_pair + jump
    
    return profiles

# ==========================================
# 2. GENERATION WORKFLOWS
# ==========================================
def generate_basic_data(filename='quarter_car_data.csv'):
    """Generates standard 5-second ground truth data."""
    def system_dynamics(x, t):
        u = basic_control_input(t)
        w = basic_road_profile(t)
        x_dot = A @ x + (B * u).flatten() + (B_w * w).flatten()
        return x_dot

    t = np.linspace(0, 5, 500)
    x0 = [0, 0, 0, 0]
    x_sol = odeint(system_dynamics, x0, t)

    u_data = [basic_control_input(ti) for ti in t]
    w_data = [basic_road_profile(ti) for ti in t]

    df = pd.DataFrame(x_sol, columns=['x1', 'x2', 'x3', 'x4'])
    df['t'] = t
    df['u'] = u_data
    df['w'] = w_data
    df.to_csv(filename, index=False)
    print(f"Basic data saved to {filename}")

def generate_robust_episodes(filename='quarter_car_robust_with_actuator.csv'):
    """Generates robust dataset with randomized multi-episode scenarios."""
    all_data = []
    for episode in range(NUM_EPISODES):
        x0 = [np.random.uniform(-0.02, 0.02), np.random.uniform(-0.1, 0.1), 0, 0]
        road_type = np.random.choice(['bump', 'rough', 'sine'])
        t_eval, w_profile = get_random_road_profile(TIME_PER_EPISODE, DT, road_type)
        u_profile = get_random_control_input(t_eval)
        
        def system_dynamics(x, t):
            idx = int(t / DT) 
            if idx >= len(t_eval): idx = len(t_eval) - 1
            u = u_profile[idx]
            w = w_profile[idx]
            return A @ x + (B * u).flatten() + (B_w * w).flatten()

        x_sol = odeint(system_dynamics, x0, t_eval)
        
        for i in range(len(t_eval)):
            all_data.append({
                'episode_id': episode, 'time': t_eval[i],
                'x1': x_sol[i, 0], 'x2': x_sol[i, 1], 'x3': x_sol[i, 2], 'x4': x_sol[i, 3],
                'u': u_profile[i], 'w': w_profile[i], 'road_type': road_type
            })

    pd.DataFrame(all_data).to_csv(filename, index=False)
    print(f"Robust data ({NUM_EPISODES} episodes) saved to {filename}")

def generate_unseen_data(filename='unseen_road_test_data_increased_amp.csv'):
    """Generates unseen test sets to test the robustness of the trained model."""
    t_eval = np.arange(0, TIME_PER_EPISODE, DT)
    profiles = generate_unseen_profiles(t_eval)
    all_data = []
    episode_id = 0

    for road_type, w_profile in profiles.items():
        def passive_dynamics(t, x):
            w = np.interp(t, t_eval, w_profile)
            return A @ x + (B_w * w).flatten()
        
        sol = solve_ivp(passive_dynamics, [t_eval[0], t_eval[-1]], np.zeros(4), t_eval=t_eval, method='RK45')
        x_sol = sol.y.T
        
        for i in range(len(t_eval)):
            all_data.append({
                'episode_id': episode_id, 'time': t_eval[i],
                'x1': x_sol[i, 0], 'x2': x_sol[i, 1], 'x3': x_sol[i, 2], 'x4': x_sol[i, 3],
                'u': 0.0, 'w': w_profile[i], 'road_type': road_type
            })
        episode_id += 1

    pd.DataFrame(all_data).to_csv(filename, index=False)
    print(f"Unseen data saved to {filename}")

if __name__ == "__main__":
    generate_basic_data()
    generate_robust_episodes()
    generate_unseen_data()
