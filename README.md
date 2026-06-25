# Neural Network Surrogate Modeling for Quarter-Car Suspension Dynamics

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

## 📌 Project Overview
This project applies deep learning to model and predict the dynamic behavior of a vehicle's suspension system. By training a **Physics-Informed Neural Network (PINN)** to mimic a mathematical **Quarter-Car Model**, the system effectively learns the complex non-linear dynamics of vehicle ride comfort and road handling. 

Once the network has successfully learned the baseline system (guided by both empirical data and physical state-space equations), it is deployed to predict the vehicle's transient response across various deterministic and unseen road profiles (e.g., sine waves, bumps, potholes, and step inputs) over time.

## ⚙️ System Architecture & Vehicle Dynamics
The neural network acts as a data-driven surrogate for a standard 2-Degree-of-Freedom (2-DOF) mechanical system. The physical parameters considered include:
* **Sprung Mass ($m_s$):** Representing the chassis and passenger load (250 kg).
* **Unsprung Mass ($m_u$):** Representing the tire and wheel assembly (35 kg).
* **Suspension System:** Primary spring stiffness ($k_s$) and damping coefficient ($b_s$).
* **Tire Dynamics:** Accounting for tire stiffness ($k_u$) interacting directly with the road surface.

## 🗂️ Project Structure

The project has been refactored for modularity and ease of use:

```text
NNCD/
├── src/
│   ├── config.py             # Physical parameters and system matrices
│   ├── data_generation.py    # Generates ground-truth, robust, and unseen test datasets
│   ├── model.py              # PyTorch definition of the QuarterCarNet
│   ├── train.py              # Training loop with Physics-Informed (PINN) loss
│   └── evaluate.py           # Evaluation logic to simulate and plot against ground truth
├── images/                   # Output plots and comparison visuals
├── project-1.ipynb           # High-level presentation and workflow notebook
├── requirements.txt          # Python dependencies
└── README.md
```

## 🛠️ Getting Started

### 1. Installation
Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone <your-repo-url>
cd NNCD

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generating Data
The neural network requires data to train on. Generate the datasets (this creates standard, robust, and test `.csv` files locally):

```bash
python3 src/data_generation.py
```

### 3. Training the Model
Train the `QuarterCarNet` using the Physics-Informed Neural Network approach. 

```bash
python3 src/train.py
```
*This will output a `model_pinn.pth` file and a training curve plot.*

### 4. Evaluation
Test the trained surrogate model on unseen road profiles:

```bash
python3 src/evaluate.py
```
*This will generate an `evaluation.png` plot comparing the Neural Network's predictions against the true physics engine.*

### Jupyter Notebook
Alternatively, you can walk through the entire pipeline interactively using the clean presentation notebook:
```bash
jupyter notebook project-1.ipynb
```

## 📊 Results

The model performs exceptionally well even on previously unseen datasets, simulating both suspension deflection and body velocity to near-perfect accuracy when compared with the ground truth mathematical model.

![Result Graph](images/unseen_road_doublebump_math_model_test.png)
