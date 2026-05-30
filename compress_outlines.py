import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv1d(256, 256, 3, padding=1)
        self.fc = nn.Linear(256 * 2, latent_dim)

    def forward(self, x, lengths):
        # x: (B, 200, 2), lengths: (B,)
        B, P, _ = x.shape
        x = x.transpose(1, 2)  # (B, 2, 200)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # (B, 256, 200)
        idx_p = torch.arange(P, device=x.device).unsqueeze(0).expand(B, P)
        mask = (idx_p < lengths.unsqueeze(1)).float()  # (B, P)
        mask_e = mask.unsqueeze(1)  # (B, 1, P)
        x_for_max = x.masked_fill(mask_e == 0, float("-inf"))
        max_pool = x_for_max.max(dim=2).values
        sum_pool = (x * mask_e).sum(dim=2)
        mean_pool = sum_pool / lengths.unsqueeze(1).float().clamp_min(1.0)
        z = torch.cat([max_pool, mean_pool], dim=1)
        return self.fc(z)


class Decoder(nn.Module):
    def __init__(self, latent_dim=32, n_points=200):
        super().__init__()
        self.n_points = n_points
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, n_points * 2)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        out = self.fc4(h).view(-1, self.n_points, 2)
        return torch.tanh(out)


class AE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc = Encoder(latent_dim)
        self.dec = Decoder(latent_dim)


# Module-level cache so repeated encode/decode calls share one model instance.
_MODEL = None
_MODEL_DEVICE = None


def _get_model(device):
    global _MODEL, _MODEL_DEVICE
    if _MODEL is None or _MODEL_DEVICE != device:
        m = AE().to(device)
        m.load_state_dict(torch.load("weights_q6.pkl", map_location=device))
        m.eval()
        _MODEL = m
        _MODEL_DEVICE = device
    return _MODEL


def encode(points, lengths):
    device = points.device
    model = _get_model(device)
    points = points.float()
    lengths = lengths.to(device).long()
    with torch.no_grad():
        return model.enc(points, lengths)


def decode(latents):
    device = latents.device
    model = _get_model(device)
    latents = latents.float()
    with torch.no_grad():
        return model.dec(latents)
