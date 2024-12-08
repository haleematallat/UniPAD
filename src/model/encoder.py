import torch.nn as nn

class LiDAREncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super(LiDAREncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ImageEncoder(nn.Module):
    def __init__(self, backbone):
        super(ImageEncoder, self).__init__()
        self.backbone = backbone  # Use a pre-trained CNN, e.g., ResNet.

    def forward(self, x):
        return self.backbone(x)
