import torch
from torch.utils.data import DataLoader
from model import DenoiseCNN
from utils import NoisyDataset
import torch.nn.functional as F
from torch import optim
import plotly.graph_objects as go
from tqdm import tqdm


NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NOISE_STD = 0.1
TRAIN_PATH = 'datasets/train'
VAL_PATH = 'datasets/val'
MODEL_SAVE_PATH = 'denoiser.pth'
LOSS_PLOT_PATH = 'training_loss_plot.png'


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = NoisyDataset(TRAIN_PATH, noise_std=NOISE_STD)
    val_data = NoisyDataset(VAL_PATH, noise_std=NOISE_STD)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = DenoiseCNN().to(device)
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
    fig.write_image(LOSS_PLOT_PATH, width=800, height=600, scale=2)
    print(f"Plot saved as {LOSS_PLOT_PATH}")

if __name__ == "__main__":
    train_model()