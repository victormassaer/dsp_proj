import torch
from model import DenoiseCNN
from utils import NoisyDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import cv2
import plotly.graph_objects as go
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


NOISE_STD = 0.2
TEST_PATH = 'datasets/test'
MODEL_PATH = 'denoiser.pth'
LOSS_PLOT_PATH = 'test_loss_plot.png'
NUM_EXAMPLES_TO_SHOW = 5
OUTPUT_COMPARE_PATH = "image_comparisons.png"


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NoisyDataset(TEST_PATH, noise_std=NOISE_STD)
    loader = DataLoader(dataset, batch_size=1)

    model = DenoiseCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    psnr_cnn, ssim_cnn = [], []
    psnr_gauss, ssim_gauss = [], []
    mse_losses = []

    for i, (noisy, clean) in enumerate(loader):
        noisy, clean = noisy.to(device), clean.to(device)
        with torch.no_grad():
            out = model(noisy)
            loss = F.mse_loss(out, clean)
            mse_losses.append(loss.item())

        # Compute quality metrics
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


    # Print average results
    print(f"Avg CNN PSNR: {np.mean(psnr_cnn):.2f}, SSIM: {np.mean(ssim_cnn):.3f}")
    print(f"Avg Gaussian PSNR: {np.mean(psnr_gauss):.2f}, SSIM: {np.mean(ssim_gauss):.3f}")

    # Plot per-image test loss
    plot_test_loss(mse_losses)
    show_image_comparisons(model, dataset, device)

def plot_test_loss(losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(losses) + 1)),
        y=losses,
        mode='lines+markers',
        name='Test MSE Loss'
    ))
    fig.update_layout(
        title='MSE Loss per Test Image',
        xaxis_title='Image Index',
        yaxis_title='MSE Loss',
        template='plotly_white'
    )
    fig.write_image(LOSS_PLOT_PATH, width=800, height=600, scale=2)

def show_image_comparisons(model, dataset, device):
    model.eval()
    fig, axes = plt.subplots(NUM_EXAMPLES_TO_SHOW, 4, figsize=(12, 3 * NUM_EXAMPLES_TO_SHOW))
    titles = ["Original", "Noisy", "CNN Denoised", "Gaussian Filtered"]

    for i in range(NUM_EXAMPLES_TO_SHOW):
        clean, noisy = dataset[i][1].unsqueeze(0), dataset[i][0].unsqueeze(0)  # (1, 1, H, W)
        clean, noisy = clean.to(device), noisy.to(device)

        with torch.no_grad():
            out = model(noisy)

        # Convert to numpy for plotting
        img_clean = clean.cpu().numpy()[0, 0]
        img_noisy = noisy.cpu().numpy()[0, 0]
        img_cnn = out.cpu().numpy()[0, 0]
        img_gauss = cv2.GaussianBlur(img_noisy, (3, 3), 1)

        images = [img_clean, img_noisy, img_cnn, img_gauss]

        for j in range(4):
            ax = axes[i, j]
            ax.imshow(images[j], cmap='gray', vmin=0, vmax=1)
            ax.set_title(titles[j])
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_COMPARE_PATH)
    plt.show()

if __name__ == "__main__":
    evaluate_model()