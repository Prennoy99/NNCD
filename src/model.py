import torch.nn as nn

class QuarterCarNet(nn.Module):
    """
    Neural Network acting as a surrogate for the 2-DOF Quarter-Car Model.
    Inputs (6 features): 4 states + 1 control input (u) + 1 disturbance input (w)
    Outputs (4 features): 4 state derivatives (x_dot)
    """
    def __init__(self):
        super(QuarterCarNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),       # Tanh is smooth, ideal for modeling physical continuous systems
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4) # Predicts x_dot (the derivative of the state)
        )

    def forward(self, x):
        return self.net(x)
