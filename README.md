# AI-Driven Active Suspension Control: Digital Twin & RL Controller

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![SciPy](https://img.shields.io/badge/SciPy-Integration-yellow.svg)

## 📌 Project Overview
This project is an advanced exploration into combining **Physics-Informed Neural Networks (PINNs)** with **Reinforcement Learning (RL) principles** to model and control a vehicle's suspension system. 

It is divided into two primary phases:
1. **The AI World (Digital Twin):** Training a Physics-Informed Neural Network to act as a highly accurate, non-linear surrogate model of a mathematical 2-DOF Quarter-Car system.
2. **The AI Brain (Neural Controller):** Training a Neural Network controller using Backpropagation Through Time (BPTT) and an LQR-style cost function inside the frozen Digital Twin environment to achieve state-of-the-art active suspension ride comfort.

## ⚙️ System Architecture & Vehicle Dynamics
The core physical system is a 2-Degree-of-Freedom (2-DOF) mechanical Quarter-Car Model. The physical parameters considered include:
* **Sprung Mass ($m_s$):** Representing the chassis and passenger load (250 kg).
* **Unsprung Mass ($m_u$):** Representing the tire and wheel assembly (35 kg).
* **Suspension System:** Primary spring stiffness ($k_s$) and damping coefficient ($b_s$).
* **Tire Dynamics:** Accounting for tire stiffness ($k_u$) interacting directly with the road surface.
* **Actuator ($u$):** An active force actuator capable of applying up to $\pm 2500$ N of force between the sprung and unsprung masses.

## 🧠 Part 1: Physics-Informed Digital Twin
Instead of relying strictly on traditional ODE solvers for control design, we train a `QuarterCarNet` (a deep neural network) to predict the state derivatives. By incorporating the physical state-space matrices into the loss function (PINN architecture), the network learns to simulate the vehicle's transient response across extreme road profiles (e.g., sine waves, bumps, potholes) with near-perfect accuracy.

## 🎮 Part 2: Active Neural Controller (BPTT & LQR)
Once the Digital Twin is highly accurate, its weights are frozen. We then introduce a `ControllerNet`, which observes the car's state and outputs an actuator force ($u$). 
* The controller is trained via **Backpropagation Through Time (BPTT)**. It "drives" through the digital twin environment for 1-second rollouts.
* We utilize a **Differentiable Runge-Kutta 4 (RK4)** integration scheme during training.
* The cost function mimics a normalized **Linear Quadratic Regulator (LQR)**, penalizing sprung mass acceleration (ride comfort), suspension travel limits (safety), tire deflection (handling), and actuator effort (energy).

## 🗂️ Project Structure

```text
NNCD/
├── src/
│   ├── config.py             # Physical parameters and system matrices
│   ├── data_generation.py    # Generates ground-truth, robust, and unseen test datasets
│   ├── model.py              # PyTorch definition of the QuarterCarNet
│   ├── train.py              # Training loop for the PINN Digital Twin
│   ├── evaluate.py           # Evaluation of the Digital Twin against ground truth
│   ├── train_controller.py   # BPTT training of the Neural Controller using the Digital Twin
│   └── evaluate_controller.py# Evaluation of the Active Neural Controller vs Passive dynamics
├── datasets/                 # Note: Generated CSV data is stored here
├── trained_models/           # Saved weights for the Digital Twin and data scalers
├── trained_controller_models/# Saved weights for the RL Neural Controller
├── NNCD_project_presentation.pptx # Project presentation deck
├── images/                   # Output plots and comparison visuals
├── demo_notebook.ipynb       # High-level presentation and workflow notebook
├── requirements.txt          # Python dependencies
└── README.md
```

## 🛠️ Getting Started

### 1. Installation
Clone the repository and set up a virtual environment:

```bash
git clone <your-repo-url>
cd NNCD
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generating Data
The models require interaction data. Generate the datasets (this creates standard, robust, and test `.csv` files locally in the `datasets/` folder):

```bash
python3 src/data_generation.py
```

### 3. Training the Digital Twin
Train the `QuarterCarNet` surrogate model:

```bash
python3 src/train.py
```

### 4. Training the Neural Controller
Train the active suspension controller inside the frozen Digital Twin using BPTT:

```bash
python3 src/train_controller.py
```
*This will output a `controller_stable_lqr.pth` file and a training curve plot.*

### 5. Evaluation & Results
Test the fully trained Active Neural Controller against the passive physical system on completely unseen road profiles:

```bash
python3 src/evaluate_controller.py
```
*This simulates the system using `scipy.integrate.solve_ivp` with RK45, ensuring no data points are skipped, and generates a detailed performance comparison graph in the `images/` directory.*

## 📊 Impact & Results

The Neural Controller achieves a substantial reduction in sprung mass acceleration (improving ride comfort) while respecting physical actuator constraints and maintaining safe suspension and tire deflection limits. By leveraging the differentiable nature of the PINN Digital Twin, the controller discovers highly optimal damping strategies that traditional PID or passive systems cannot match.

> **Note:** For a complete visual walkthrough and presentation of the methodology, please review `demo_notebook.ipynb` and the accompanying `NNCD_project_presentation.pptx` file.
