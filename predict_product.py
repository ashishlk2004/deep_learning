import torch
import torch.nn as nn
import torch.nn.functional as F

# Class index -> product (in the same order as the training-time class mapping)
_PRODUCTS = [6, 8, 10, 12, 15, 18, 20, 24, 30, 36, 40, 48, 60, 72, 90, 120]
_TRAIN_RES = 64  # the resolution used at training time

class DiceProductCNN(nn.Module):
    def __init__(self, n_classes=16):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.b1 = block(3, 32)
        self.b2 = block(32, 64)
        self.b3 = block(64, 128)
        self.b4 = block(128, 192)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(192, n_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


def predict(images):
    device = images.device
    images = images.float()

    # Resize from input resolution (assessment spec is 128x128) to the training resolution.
    if images.shape[-1] != _TRAIN_RES or images.shape[-2] != _TRAIN_RES:
        images = F.interpolate(
            images, size=(_TRAIN_RES, _TRAIN_RES), mode="bilinear", align_corners=False
        )

    model = DiceProductCNN(n_classes=len(_PRODUCTS)).to(device)
    model.load_state_dict(torch.load("weights_q5.pkl", map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(images)
        idx = logits.argmax(dim=1)

    products = torch.tensor(_PRODUCTS, device=device, dtype=torch.long)[idx]
    return products.unsqueeze(1)
