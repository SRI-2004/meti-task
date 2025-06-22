import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Layers helper from your training script
class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

# Updated Generator model to match your new trainer.py
class Generator(nn.Module):
    """G(z, y) → 28×28 grayscale image conditioned on digit *y* (one‑hot).
    """
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim

        self.project = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample14 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample28 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.to_img = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # (B, L), (B,)
        y_onehot = F.one_hot(y, 10).float()
        x = torch.cat([z, y_onehot], dim=1)
        x = self.project(x).view(-1, 256, 7, 7)
        x = self.upsample14(x)
        x = self.upsample28(x)
        return self.to_img(x)

@st.cache_resource
def load_model():
    """Loads the pre-trained generator model and caches it."""
    # --- IMPORTANT ---
    # This path must be relative to the root of your repository
    # to work on Render.
    model_path = 'G_epoch_29.pt'
    # --- IMPORTANT ---

    try:
        model = Generator()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure the file exists at the specified path.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the trained generator model
generator = load_model()


st.title("Handwritten Digit Image Generator")

st.write("Generate synthetic MNIST-like images using your trained model.")


# User input
digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate Images"):
    if generator:
        st.write(f"Generated images of digit {digit}")

        # Generate 5 images
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                with torch.no_grad():
                    # Generate a random noise vector
                    z = torch.randn(1, 100)
                    label = torch.LongTensor([digit])
                    
                    # Generate an image
                    # The output shape will be (1, 1, 28, 28), so we squeeze it
                    generated_image = generator(z, label).squeeze().numpy()
                    
                    # Post-process the image
                    # The generator outputs values in [-1, 1], so we scale it to [0, 255]
                    generated_image = (generated_image * 0.5 + 0.5) * 255
                    generated_image = generated_image.astype(np.uint8)

                    st.image(Image.fromarray(generated_image), caption=f"Sample {i+1}", use_container_width=True)
    else:
        st.warning("Cannot generate images because the model could not be loaded.") 