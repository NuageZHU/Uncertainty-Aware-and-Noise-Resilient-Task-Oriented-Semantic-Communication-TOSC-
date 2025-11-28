"""
Quick Test Script for Quantization Experiment
----------------------------------------------

This is a simplified version to quickly test if everything works.
It uses only the existing 'people2.png' image and a smaller parameter grid.

Usage:
    python experiments/exp_quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from diffusers import AutoencoderKL
import open_clip

from semcom_utils import (
    vae_encode, vae_decode, channel,
    clip_img_embed, clip_tensor_embed, cosine_sim
)


# Quick test parameters
N_BITS_LIST = [2, 4, 6, 8]  # Fewer bits for quick test
SIGMA_LIST = [0.0, 0.1, 0.2]  # Fewer noise levels
IMG_PATH = "people2.png"  # Use the existing example image
SIZE_VAE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("Loading models...")

# Load models
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
sf = getattr(vae.config, "scaling_factor", 1.0)

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai",
    device=DEVICE
)
clip_model.eval()

pre_vae = transforms.Compose([
    transforms.Resize((SIZE_VAE, SIZE_VAE), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

print("Models loaded!\n")

# Load image once
from PIL import Image
pil = Image.open(IMG_PATH).convert("RGB")
x_in = pre_vae(pil).unsqueeze(0).to(DEVICE)

# Compute reference embedding
emb_ref = clip_img_embed(clip_model, clip_preprocess, pil, DEVICE)

# Encode once (reuse for all experiments)
z0 = vae_encode(vae, x_in, sf)
x_hat0 = vae_decode(vae, z0, sf)
emb_loc = clip_tensor_embed(clip_model, clip_preprocess, x_hat0[0], DEVICE)
sim_local = cosine_sim(emb_ref, emb_loc)

print(f"Image: {IMG_PATH}")
print(f"Local reconstruction similarity: {sim_local:.4f}\n")

# Run experiments
results = []
total = len(N_BITS_LIST) * len(SIGMA_LIST)

print(f"Running {total} experiments...")
with tqdm(total=total) as pbar:
    for n_bits in N_BITS_LIST:
        for sigma in SIGMA_LIST:
            # Apply channel
            z_tx = channel(z0, sigma=sigma, n_bits=n_bits, p_drop=0.0)
            x_hat_rx = vae_decode(vae, z_tx, sf)
            emb_rx = clip_tensor_embed(clip_model, clip_preprocess, x_hat_rx[0], DEVICE)
            sim_rx = cosine_sim(emb_ref, emb_rx)
            
            results.append({
                "n_bits": n_bits,
                "sigma": sigma,
                "sim_local": float(sim_local),
                "sim_rx": float(sim_rx),
                "degradation": float(sim_local - sim_rx)
            })
            
            pbar.update(1)

# Create DataFrame and save
df = pd.DataFrame(results)
output_file = Path("results") / "quick_test_results.csv"
output_file.parent.mkdir(exist_ok=True)
df.to_csv(output_file, index=False)

print(f"\n✅ Results saved to: {output_file}\n")

# Print results table
print("="*70)
print("Results:")
print("="*70)
print(df.to_string(index=False))
print()

# Print insights
print("="*70)
print("Key Insights:")
print("="*70)
best = df.loc[df["sim_rx"].idxmax()]
worst = df.loc[df["sim_rx"].idxmin()]

print(f"Best performance: {best['n_bits']}-bit, σ={best['sigma']:.2f} → sim={best['sim_rx']:.4f}")
print(f"Worst performance: {worst['n_bits']}-bit, σ={worst['sigma']:.2f} → sim={worst['sim_rx']:.4f}")
print(f"Max degradation: {df['degradation'].max():.4f}")
print(f"Min degradation: {df['degradation'].min():.4f}")
