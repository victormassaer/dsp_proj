"""
- load the dataset
- add noise
- train CNN model
- save trained model
"""

import torch
from torch.utils.data import DataLoader
from model import get_model
from utils import NoisyDataset
import torch.nn.functional as F
from torch import optim
import plotly.graph_objects as go
from tqdm import tqdm
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import random
from datetime import datetime
import pytz
tz = pytz.timezone('Europe/Brussels')

# hyperparameters for model training
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NOISE_STD = 0.2
TRAIN_PATH = 'datasets/train'
VAL_PATH = 'datasets/validate'
MODEL_SAVE_PATH = 'denoiser.pth'
LOSS_PLOT_PATH_TRAIN = 'training_loss_plot.png'

# change to model_type = "convnet" for convnet
# change to model_type = "unet" for mini-unet
# change to model_type = "gan" for GAN-net
MODEL_TYPE = "gan"




def train_model(model_type="unet"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    print(torch.cuda.is_available())

    train_data = NoisyDataset(TRAIN_PATH, noise_std=NOISE_STD)
    val_data = NoisyDataset(VAL_PATH, noise_std=NOISE_STD)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = get_model(model_type=model_type).to(device)

    if model_type == "gan":
        optimizer_G = optim.Adam(model.generator.parameters(), lr=LEARNING_RATE)  # generator
        optimizer_D = optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE)  # discriminator

        train_losses = []
        val_losses = []

        for epoch in tqdm(range(NUM_EPOCHS), desc="training GAN"):
            model.train()
            g_epoch_loss, d_epoch_loss = 0, 0

            for noisy, clean in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)

                # discriminator
                optimizer_D.zero_grad()
                fake_img = model.generator(noisy).detach()
                real_pred = model.discriminator(clean)
                fake_pred = model.discriminator(fake_img)
                d_loss = model.discriminator_loss(real_pred, fake_pred)
                d_loss.backward()
                optimizer_D.step()

                # generator
                optimizer_G.zero_grad()
                fake_img = model.generator(noisy)
                fake_pred = model.discriminator(fake_img)
                g_loss = model.generator_loss(fake_pred, fake_img, clean, epoch=epoch)
                g_loss.backward()
                optimizer_G.step()

                g_epoch_loss += g_loss.item()
                d_epoch_loss += d_loss.item()

            train_losses.append(g_epoch_loss / len(train_loader))
            val_losses.append(d_epoch_loss / len(train_loader))
            print(f"Epoch {epoch+1}: G loss: {g_epoch_loss:.4f} | D loss: {d_epoch_loss:.4f}")

        torch.save(model.generator.state_dict(), MODEL_SAVE_PATH)

    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_losses = []
        val_losses = []

        for epoch in tqdm(range(NUM_EPOCHS), desc="training process"):
            model.train()
            total_train_loss = 0
            for noisy, clean in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out = model(noisy)
                loss = F.mse_loss(out, clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- Validation ---
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    out = model(noisy)
                    val_loss = F.mse_loss(out, clean)
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    plot_loss(train_losses, val_losses)

def plot_loss(train_losses, val_losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(train_losses)+1)),
        y=train_losses,
        mode='lines+markers',
        name='Training Loss'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(val_losses)+1)),
        y=val_losses,
        mode='lines+markers',
        name='Validation Loss'
    ))
    fig.update_layout(
        title='Training & Validation Loss Over Epochs',
        xaxis_title='Epoch',
        yaxis_title='MSE Loss',
        template='plotly_white'
    )
    fig.write_image(LOSS_PLOT_PATH_TRAIN, width=800, height=600, scale=2)
    print(f"Plot saved as {LOSS_PLOT_PATH_TRAIN}")



if __name__ == "__main__":
    print("Started at:", datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S"))
    train_model(model_type=MODEL_TYPE)  # convnet, unet, gan
    print("Ended at:", datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S"))