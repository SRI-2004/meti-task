#!/usr/bin/env python
"""
Improved AC‑GAN for MNIST
========================
* Adds stronger label conditioning, better regularisation, and evaluation hooks *

Key upgrades vs. the original script
------------------------------------
1. One‑hot concatenation instead of embedding for the Generator.
2. Tunable classifier‑loss weight (`LAMBDA_CLS`).
3. Spectral Normalisation on *all* D layers.
4. LeakyReLU activations for G (less dead‑ReLU).
5. Exponential Moving Average (EMA) copy of G for clean samples.
6. On‑the‑fly classifier‑accuracy metric on generated images.

Run:
```
$ python mnist_acgan_mod.py
```
The script will create `./samples/` and `./checkpoints/`.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import pack, unpack  # convenience for flatten/reshape
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm

# ─────────────────────────────
# Hyper‑parameters
# ─────────────────────────────
EPOCHS        = 30
BATCH_SIZE    = 128
LR            = 2e-4
LATENT_DIM    = 100
NUM_CLASSES   = 10
LAMBDA_CLS    = 1.0     # weight of cross‑entropy in all losses
LAMBDA_FM     = 0.0     # set >0 to enable feature‑matching
EMA_BETA      = 0.999   # set None to disable EMA
DATA_DIR      = "./data"
SAMPLE_DIR    = "./samples"
CKPT_DIR      = "./checkpoints"
USE_CPU_ONLY  = False

# ─────────────────────────────
# Layers helper
# ─────────────────────────────
class ConvSNLeaky(nn.Sequential):
    """Spectral‑Norm Conv + LeakyReLU"""

    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.utils.spectral_norm(nn.Conv2d(cin, cout, k, s, p)),
            nn.LeakyReLU(0.2, inplace=True),
        )


# ─────────────────────────────
# Generator
# ─────────────────────────────
class Generator(nn.Module):
    """G(z, y) → 28×28 grayscale image conditioned on digit *y* (one‑hot).
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.project = nn.Sequential(
            nn.Linear(latent_dim + NUM_CLASSES, 256 * 7 * 7),
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
        y_onehot = F.one_hot(y, NUM_CLASSES).float()
        x = torch.cat([z, y_onehot], dim=1)
        x = self.project(x).view(-1, 256, 7, 7)
        x = self.upsample14(x)
        x = self.upsample28(x)
        return self.to_img(x)


# ─────────────────────────────
# Discriminator with auxiliary classifier
# ─────────────────────────────
class Discriminator(nn.Module):
    """Returns (adv_prob, class_logits, feature_map). feature_map is optional
    output used for feature‑matching."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ConvSNLeaky(1, 64, 4, 2, 1),
            ConvSNLeaky(64, 128, 4, 2, 1),
        )
        self.flatten = nn.Flatten()
        self.in_dim = 128 * 7 * 7
        self.adv_head = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.in_dim, 1)),
            nn.Sigmoid(),
        )
        self.cls_head = nn.utils.spectral_norm(nn.Linear(self.in_dim, NUM_CLASSES))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        feat_f, _ = pack([feat], 'b *')  # flatten but keep grad ok
        adv = self.adv_head(feat_f)
        cls = self.cls_head(feat_f)
        return adv, cls, feat_f


# ─────────────────────────────
# Helper: generate n images
# ─────────────────────────────
@torch.no_grad()
def generate_digit(generator: nn.Module, digit: int, n_imgs: int = 5, device: str = "cpu") -> torch.Tensor:
    generator.eval()
    z = torch.randn(n_imgs, LATENT_DIM, device=device)
    y = torch.full((n_imgs,), digit, dtype=torch.long, device=device)
    imgs = generator(z, y).cpu()
    return imgs


# ─────────────────────────────
# Training routine
# ─────────────────────────────

def train() -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU_ONLY else "cpu")
    print("Device:", device)

    Path(SAMPLE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

    ds = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    G, D = Generator().to(device), Discriminator().to(device)
    if EMA_BETA is not None:
        G_ema = deepcopy(G).eval()  # EMA copy; used only for sampling
    else:
        G_ema = None

    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=LR * 0.5, betas=(0.0, 0.99))

    fixed_z = torch.randn(NUM_CLASSES * 10, LATENT_DIM, device=device)
    fixed_lbl = torch.arange(NUM_CLASSES, device=device).repeat_interleave(10)

    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        pbar = tqdm(dl, leave=False, ncols=100, desc=f"Epoch {epoch}/{EPOCHS}")
        clf_acc_meter = 0.0
        n_batches = 0

        for real, labels in pbar:
            real, labels = real.to(device), labels.to(device)
            bs = real.size(0)
            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)

            # ---------------- D step ----------------
            opt_D.zero_grad(set_to_none=True)
            adv_real, cls_real, feat_real = D(real)
            loss_real = bce(adv_real, valid * 0.9) + LAMBDA_CLS * ce(cls_real, labels)

            z = torch.randn(bs, LATENT_DIM, device=device)
            fake_labels = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            fake_imgs = G(z, fake_labels).detach()
            adv_fake, cls_fake, feat_fake = D(fake_imgs)
            loss_fake = bce(adv_fake, fake) + LAMBDA_CLS * ce(cls_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # ---------------- G step ----------------
            opt_G.zero_grad(set_to_none=True)
            z = torch.randn(bs, LATENT_DIM, device=device)
            gen_labels = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            gen_imgs = G(z, gen_labels)
            adv_gen, cls_gen, feat_gen = D(gen_imgs)
            loss_G = bce(adv_gen, valid) + LAMBDA_CLS * ce(cls_gen, gen_labels)
            if LAMBDA_FM > 0:
                loss_fm = F.l1_loss(feat_gen.mean(0), feat_real.detach().mean(0))
                loss_G = loss_G + LAMBDA_FM * loss_fm
            loss_G.backward()
            opt_G.step()

            # EMA update
            if G_ema is not None:
                with torch.no_grad():
                    for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                        p_ema.data.mul_(EMA_BETA).add_(p.data, alpha=1.0 - EMA_BETA)

            # track classifier accuracy on fakes
            clf_acc = (cls_gen.argmax(1) == gen_labels).float().mean()
            clf_acc_meter += clf_acc.item()
            n_batches += 1

            pbar.set_postfix(l_D=loss_D.item(), l_G=loss_G.item(), acc=f"{clf_acc.item()*100:4.1f}%")

        # Snapshot grid from EMA G if available, else current G
        G_vis = G_ema if G_ema is not None else G
        with torch.no_grad():
            grid = G_vis(fixed_z, fixed_lbl).cpu()
        utils.save_image(grid, f"{SAMPLE_DIR}/epoch_{epoch:03d}.png", nrow=10,
                         normalize=True, value_range=(-1, 1))
        torch.save(G.state_dict(), f"{CKPT_DIR}/G_epoch_{epoch}.pt")
        print(f"✓ Epoch {epoch} — D {loss_D.item():.3f} | G {loss_G.item():.3f} | Fake‑cls‑acc {(clf_acc_meter/n_batches)*100:.2f}%")

    return (G_ema or G).cpu()


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available() and not USE_CPU_ONLY:
        torch.cuda.manual_seed_all(42)
    train()
