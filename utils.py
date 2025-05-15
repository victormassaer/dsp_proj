import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import random

class NoisyDataset(Dataset):
    def __init__(self, root_dir, noise_std=0.1, random_noise=False, noise_std_options=None):
        """
        Parameters:
        - root_dir: pad naar je afbeeldingsmap
        - noise_std: standaard std wanneer random_noise=False
        - random_noise: of er random gekozen moet worden uit noise_std_options
        - noise_std_options: lijst van mogelijke std's bij random_noise (default: [0.02, 0.05, 0.1, 0.2])
        """
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        self.noise_std = noise_std
        self.random_noise = random_noise
        self.noise_std_options = noise_std_options or [0.02, 0.05, 0.1, 0.2]

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        clean = Image.open(self.image_paths[idx]).convert('RGB')
        clean = self.transform(clean)

        # Bepaal de ruissterkte
        std = random.choice(self.noise_std_options) if self.random_noise else self.noise_std
        noise = torch.randn_like(clean) * std
        noisy = clean + noise
        noisy = torch.clamp(noisy, 0., 1.)

        return noisy, clean

    def __len__(self):
        return len(self.image_paths)
