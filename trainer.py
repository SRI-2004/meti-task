#!/usr/bin/env python
"""
AC‑GAN for MNIST (Colab‑ready)
==============================
*Fixes the label‑mismatch issue* by adding an **auxiliary‑classifier head** to the
Discriminator. The losses now enforce that generated digits must be recognised
as the *requested* class.

Key changes
------------
1. **Discriminator** returns `(adv_prob, class_logits)`.
2. **Losses** combine BCE (real/fake) and Cross‑Entropy (class).
3. Training loop updated accordingly.

Usage
-----
```python
!pip install --quiet torch torchvision tqdm einops
from mnist_acgan import train, generate_digit
G = train()
```
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm

# ─────────────────────────────
# Hyper‑parameters
# ─────────────────────────────
EPOCHS       = 30
BATCH_SIZE   = 128
LR           = 2e-4
LATENT_DIM   = 100
EMBED_DIM    = 50
NUM_CLASSES  = 10
DATA_DIR     = "./data"
SAMPLE_DIR   = "./samples"
CKPT_DIR     = "./checkpoints"
USE_CPU_ONLY = False

# ─────────────────────────────
# Layers helper
# ─────────────────────────────
class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

# ─────────────────────────────
# Generator (unchanged vs previous)
# ─────────────────────────────
class Generator(nn.Module):
    """G(z, y) → 28×28 grayscale image conditioned on digit *y*."""
    def __init__(self, latent_dim=LATENT_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, embed_dim)

        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
        )
        self.upsample14 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ConvBNReLU(128, 128),
        )
        self.upsample28 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ConvBNReLU(64, 64),
        )
        self.to_img = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, y):
        x = torch.cat([z, self.label_emb(y)], dim=1)
        x = self.project(x).view(-1, 256, 7, 7)
        x = self.upsample14(x)
        x = self.upsample28(x)
        return self.to_img(x)

# ─────────────────────────────
# Discriminator with auxiliary classifier
# ─────────────────────────────
class Discriminator(nn.Module):
    """Returns (adv_prob, class_logits)."""
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),  # (B, 128*7*7)
        )
        in_dim = 128 * 7 * 7
        self.adv_head   = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        self.class_head = nn.Linear(in_dim, NUM_CLASSES)  # logits

    def forward(self, x):
        feat = self.feat(x)
        return self.adv_head(feat), self.class_head(feat)

# ─────────────────────────────
# Helper: generate n images for Streamlit
# ─────────────────────────────
@torch.no_grad()
def generate_digit(generator, digit, n_imgs=5, device="cpu"):
    generator.eval()
    z = torch.randn(n_imgs, LATENT_DIM, device=device)
    y = torch.full((n_imgs,), digit, dtype=torch.long, device=device)
    imgs = generator(z, y).cpu()
    return imgs

# ─────────────────────────────
# Training routine
# ─────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU_ONLY else "cpu")
    print("Device:", device)

    Path(SAMPLE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

    ds = datasets.MNIST(DATA_DIR, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    G, D = Generator().to(device), Discriminator().to(device)
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=LR * 0.5, betas=(0.0, 0.99))

    fixed_z   = torch.randn(NUM_CLASSES * 10, LATENT_DIM, device=device)
    fixed_lbl = torch.arange(NUM_CLASSES, device=device).repeat_interleave(10)

    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        pbar = tqdm(dl, leave=False, ncols=100, desc=f"Epoch {epoch}/{EPOCHS}")

        for real, labels in pbar:
            real, labels = real.to(device), labels.to(device)
            bs = real.size(0)
            valid = torch.ones(bs, 1, device=device)
            fake  = torch.zeros(bs, 1, device=device)

            # ---------------- D step ----------------
            opt_D.zero_grad(set_to_none=True)
            adv_real, cls_real = D(real)
            loss_real = bce(adv_real, valid * 0.9) + ce(cls_real, labels)

            z = torch.randn(bs, LATENT_DIM, device=device)
            fake_labels = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            fake_imgs = G(z, fake_labels).detach()
            adv_fake, cls_fake = D(fake_imgs)
            loss_fake = bce(adv_fake, fake) + ce(cls_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # ---------------- G step ----------------
            opt_G.zero_grad(set_to_none=True)
            z = torch.randn(bs, LATENT_DIM, device=device)
            gen_labels = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            gen_imgs = G(z, gen_labels)
            adv_gen, cls_gen = D(gen_imgs)
            loss_G = bce(adv_gen, valid) + ce(cls_gen, gen_labels)
            loss_G.backward()
            opt_G.step()

            pbar.set_postfix(l_D=loss_D.item(), l_G=loss_G.item())

        # snapshot grid
        with torch.no_grad():
            grid = G(fixed_z, fixed_lbl).cpu()
            utils.save_image(grid, f"{SAMPLE_DIR}/epoch_{epoch:03d}.png", nrow=10,
                             normalize=True, value_range=(-1, 1))
            torch.save(G.state_dict(), f"{CKPT_DIR}/G_epoch_{epoch}.pt")
        print(f"✓ Epoch {epoch} — loss_D {loss_D.item():.3f} | loss_G {loss_G.item():.3f}")

    return G.cpu()



if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available() and not USE_CPU_ONLY:
        torch.cuda.manual_seed_all(42)
    train()
