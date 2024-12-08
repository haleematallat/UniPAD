import torch.nn as nn
from .encoder import LiDAREncoder, ImageEncoder
from .decoder import NeuralRenderingDecoder

class UniPAD(nn.Module):
    def __init__(self, image_backbone):
        super(UniPAD, self).__init__()
        self.lidar_encoder = LiDAREncoder()
        self.image_encoder = ImageEncoder(image_backbone)
        self.decoder = NeuralRenderingDecoder()

    def forward(self, lidar, images):
        lidar_features = self.lidar_encoder(lidar)
        image_features = self.image_encoder(images)
        combined_features = lidar_features + image_features  # Simple fusion
        output = self.decoder(combined_features)
        return output
