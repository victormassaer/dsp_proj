import torch
from model import DenoiseCNN
from utils import NoisyDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import cv2

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NoisyDataset('datasets/test', noise_std=0.1)
    loader = DataLoader(dataset, batch_size=1)

    model = DenoiseCNN().to(device)
    model.load_state_dict(torch.load('denoiser.pth', map_location=device))
    model.eval()

    psnr_cnn, ssim_cnn = [], []
    psnr_gauss, ssim_gauss = [], []

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        with torch.no_grad():
            out = model(noisy)

        img_clean = clean.cpu().numpy()[0, 0]
        img_cnn = out.cpu().numpy()[0, 0]
        img_noisy = noisy.cpu().numpy()[0, 0]

        cnn_psnr = peak_signal_noise_ratio(img_clean, img_cnn, data_range=1)
        cnn_ssim = structural_similarity(img_clean, img_cnn, data_range=1)

        gauss_img = cv2.GaussianBlur(img_noisy, (3, 3), 1)
        g_psnr = peak_signal_noise_ratio(img_clean, gauss_img, data_range=1)
        g_ssim = structural_similarity(img_clean, gauss_img, data_range=1)

        psnr_cnn.append(cnn_psnr)
        ssim_cnn.append(cnn_ssim)
        psnr_gauss.append(g_psnr)
        ssim_gauss.append(g_ssim)

    print(f"Avg CNN PSNR: {np.mean(psnr_cnn):.2f}, SSIM: {np.mean(ssim_cnn):.3f}")
    print(f"Avg Gaussian PSNR: {np.mean(psnr_gauss):.2f}, SSIM: {np.mean(ssim_gauss):.3f}")

if __name__ == "__main__":
    evaluate_model()