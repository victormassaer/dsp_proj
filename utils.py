import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import random

class NoisyDataset(Dataset):
    def __init__(self, root_dir, noise_std=0.1):
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        self.noise_std = noise_std
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        clean = Image.open(self.image_paths[idx]).convert('RGB')
        clean = self.transform(clean)
        noise = torch.randn_like(clean) * self.noise_std
        noisy = clean + noise
        noisy = torch.clamp(noisy, 0., 1.)
        return noisy, clean

    def __len__(self):
        return len(self.image_paths)