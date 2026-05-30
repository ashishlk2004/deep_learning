import torch
import torch.nn as nn

# Normalisation statistics from training set (mass and area columns are *post-log*).
_X_MEAN = torch.tensor([
    398.27752685546875,
    3.390331983566284,
    -1.0406532287597656,
    0.025237012654542923,
    157.89096069335938,
    2.2450106143951416,
])
_X_STD = torch.tensor([
    112.34913635253906,
    1.7450282573699951,
    1.7732372283935547,
    0.014580701477825642,
    52.46287155151367,
    0.4225120544433594,
])
_LY_MEAN = 5.877403736114502
_LY_STD = 1.6632566452026367

class DecayPredictionNetwork(nn.Module):
    def __init__(self, in_dim=6, hidden=128, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def predict(parameters):
    device = parameters.device
    parameters = parameters.float()

    model = DecayPredictionNetwork().to(device)
    model.load_state_dict(torch.load("weights_q4.pkl", map_location=device))
    model.eval()

    x_mean = _X_MEAN.to(device)
    x_std = _X_STD.to(device)

    # log-transform mass and cross-sectional area (columns 1 and 2)
    x_proc = parameters.clone()
    x_proc[:, 1] = torch.log(x_proc[:, 1])
    x_proc[:, 2] = torch.log(x_proc[:, 2])

    x_norm = (x_proc - x_mean) / x_std

    with torch.no_grad():
        y_log_norm = model(x_norm)

    y_log = y_log_norm * _LY_STD + _LY_MEAN
    y_days = torch.exp(y_log)
    return y_days
