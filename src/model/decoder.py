import torch.nn as nn

class NeuralRenderingDecoder(nn.Module):
    def __init__(self, feature_dim=64, output_dim=3):
        super(NeuralRenderingDecoder, self).__init__()
        self.render_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # Output RGB or depth
        )

    def forward(self, features):
        return self.render_mlp(features)
