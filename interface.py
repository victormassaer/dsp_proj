# gui.py

import dash
from dash import html, dcc, Input, Output, State
import torch
from PIL import Image
import base64
import io
import numpy as np
import plotly.express as px

from model import get_model
from evaluate import predict_single_image

MODEL_SAVE_PATH = 'denoiser.pth'

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

# Setup Dash app
app = dash.Dash(__name__)
app.title = "Image Denoising Interface"

app.layout = html.Div([
    html.H2("Image Denoising with Your CNN", style={"textAlign": "center"}),
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Drag and Drop or ', html.A('Select Image')]),
        style={
            'width': '50%', 'height': '100px', 'lineHeight': '100px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '10px', 'textAlign': 'center', 'margin': 'auto'
        },
        accept='image/*'
    ),
    html.Div(id='image-display'),
    html.Div(id='metrics-display', style={'textAlign': 'center', 'marginTop': '20px'})
])

@app.callback(
    [Output('image-display', 'children'),
     Output('metrics-display', 'children')],
    Input('upload-image', 'contents')
)
def handle_image(contents):
    if contents is None:
        return dash.no_update, dash.no_update

    content_type, content_string = contents.split(',')
    image_bytes = base64.b64decode(content_string)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('L')
    pil_img = pil_img.resize((240, 160))  # match your expected dimensions

    # Convert image to tensor
    img_array = np.array(pil_img) / 255.0
    tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Denoise using evaluate_model
    denoised_tensor, psnr, ssim = predict_single_image(model, tensor)

    # Back to numpy
    noisy_img = tensor.squeeze().cpu().numpy()
    denoised_img = denoised_tensor.squeeze().cpu().numpy()

    fig1 = px.imshow(noisy_img, color_continuous_scale='gray', title='Noisy Input')
    fig2 = px.imshow(denoised_img, color_continuous_scale='gray', title='Denoised Output')

    return [
        html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
        ])
    ], f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}" if psnr is not None and ssim is not None else "No ground truth available for PSNR/SSIM"


if __name__ == '__main__':
    app.run(debug=True)
