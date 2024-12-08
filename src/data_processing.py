import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NuScenesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = self.load_samples()

    def load_samples(self):
        # Load file paths for LiDAR and images
        lidar_files = sorted(os.listdir(os.path.join(self.data_dir, 'lidar')))
        image_files = sorted(os.listdir(os.path.join(self.data_dir, 'images')))
        return list(zip(lidar_files, image_files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lidar_file, image_file = self.samples[idx]
        lidar = np.load(os.path.join(self.data_dir, 'lidar', lidar_file))
        image = np.load(os.path.join(self.data_dir, 'images', image_file))
        return torch.tensor(lidar, dtype=torch.float32), torch.tensor(image, dtype=torch.float32)

def load_dataset(data_dir, batch_size=4):
    dataset = NuScenesDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
