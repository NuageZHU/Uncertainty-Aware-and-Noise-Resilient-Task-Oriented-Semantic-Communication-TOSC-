"""
Experiment: Quantization and Noise Impact on Semantic Fidelity
---------------------------------------------------------------

This script systematically evaluates how different quantization bit depths (n_bits)
and channel noise levels (sigma) affect semantic similarity (CLIP score).

The goal is to generate Rate-Semantic Distortion curves for analysis.

Usage:
    python experiments/exp_quantization_noise.py
    
Output:
    - results/quantization_noise_results.csv: Full experimental data
    - results/plots/*.png: Visualization plots
"""

import sys
import os
# Add parent directory to path to import semcom modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from diffusers import AutoencoderKL
import open_clip

# Import our utility functions
from semcom_utils import (
    vae_encode, vae_decode, channel,
    clip_img_embed, clip_tensor_embed, cosine_sim
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Experimental parameters
N_BITS_LIST = [2, 4, 6, 8, 12, 16]  # Quantization bit depths to test
SIGMA_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]  # Noise levels to test
TAU = 0.02  # Uncertainty threshold (for transmission gate)
DROPOUT_P = 0.0  # Fixed dropout probability

# Paths
DATA_DIR = Path("data/all_images")  # Directory containing input images
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
CSV_PATH = RESULTS_DIR / "quantization_noise_results.csv"

# Sampling configuration
MAX_IMAGES = 20  # Set to None to use all images, or a number (e.g., 20) to limit
RANDOM_SAMPLE = True  # If True, randomly sample MAX_IMAGES; if False, take first MAX_IMAGES
RANDOM_SEED = 42  # For reproducible sampling

# Model parameters
SIZE_VAE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ----------------------------------------------------------------------
# Initialize Models (once, globally)
# ----------------------------------------------------------------------

print("Loading models...")
print(f"Using device: {DEVICE}")

# Load VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
sf = getattr(vae.config, "scaling_factor", 1.0)

# Load CLIP
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai",
    device=DEVICE
)
clip_model.eval()

# VAE preprocessing
pre_vae = transforms.Compose([
    transforms.Resize((SIZE_VAE, SIZE_VAE), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

print("Models loaded successfully!\n")


# ----------------------------------------------------------------------
# Core Experiment Function
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate_single_image(img_path, n_bits, sigma, tau=0.02):
    """
    Evaluate a single image with given channel parameters.
    
    Args:
        img_path: Path to input image
        n_bits: Quantization bit depth (None for no quantization)
        sigma: Noise standard deviation
        tau: Uncertainty threshold
    
    Returns:
        dict with metrics
    """
    # Load and preprocess
    pil = Image.open(img_path).convert("RGB")
    x_in = pre_vae(pil).unsqueeze(0).to(DEVICE)
    
    # Encode and decode locally (no channel)
    z0 = vae_encode(vae, x_in, sf)
    x_hat0 = vae_decode(vae, z0, sf)
    
    # Compute CLIP similarity for local reconstruction
    emb_ref = clip_img_embed(clip_model, clip_preprocess, pil, DEVICE)
    emb_loc = clip_tensor_embed(clip_model, clip_preprocess, x_hat0[0], DEVICE)
    sim_local = cosine_sim(emb_ref, emb_loc)
    
    # Uncertainty and transmission decision
    uncertainty = 1.0 - sim_local
    transmit = uncertainty > tau
    
    # Apply channel and compute post-channel similarity
    z_tx = channel(z0, sigma=sigma, n_bits=n_bits, p_drop=DROPOUT_P)
    x_hat_rx = vae_decode(vae, z_tx, sf)
    emb_rx = clip_tensor_embed(clip_model, clip_preprocess, x_hat_rx[0], DEVICE)
    sim_rx = cosine_sim(emb_ref, emb_rx)
    
    # Calculate semantic degradation
    semantic_degradation = sim_local - sim_rx
    
    return {
        "sim_local": float(sim_local),
        "sim_rx": float(sim_rx),
        "uncertainty": float(uncertainty),
        "transmit": bool(transmit),
        "semantic_degradation": float(semantic_degradation),
    }

def run_experiment():
    """
    Run the full experiment across all images and parameter combinations.
    """
    # Find all images in data directory
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(DATA_DIR.glob(f"*{ext}")))
    
    # Remove duplicates (Windows is case-insensitive, may find same file twice)
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"⚠️  No images found in {DATA_DIR}")
        print(f"Please add some images to the '{DATA_DIR}' directory.")
        return None
    
    print(f"Found {len(image_files)} images in {DATA_DIR}")
    
    # Apply sampling if configured
    if MAX_IMAGES is not None and MAX_IMAGES < len(image_files):
        if RANDOM_SAMPLE:
            np.random.seed(RANDOM_SEED)
            image_files = list(np.random.choice(image_files, MAX_IMAGES, replace=False))
            print(f"Randomly sampled {MAX_IMAGES} images (seed={RANDOM_SEED})")
        else:
            image_files = image_files[:MAX_IMAGES]
            print(f"Using first {MAX_IMAGES} images")
    else:
        print(f"Using all {len(image_files)} images")
    
    print(f"Testing {len(N_BITS_LIST)} quantization levels × {len(SIGMA_LIST)} noise levels")
    print(f"Total experiments: {len(image_files) * len(N_BITS_LIST) * len(SIGMA_LIST)}")
    print(f"Estimated time: {len(image_files) * len(N_BITS_LIST) * len(SIGMA_LIST) * 2 / 60:.1f} minutes\n")
    
    # Storage for results
    results = []
    
    # Iterate through all combinations
    total_iterations = len(image_files) * len(N_BITS_LIST) * len(SIGMA_LIST)
    
    with tqdm(total=total_iterations, desc="Running experiments") as pbar:
        for img_path in image_files:
            img_name = img_path.name
            
            for n_bits in N_BITS_LIST:
                for sigma in SIGMA_LIST:
                    try:
                        # Run evaluation
                        metrics = evaluate_single_image(img_path, n_bits, sigma, TAU)
                        
                        # Store results
                        results.append({
                            "img_name": img_name,
                            "n_bits": n_bits,
                            "sigma": sigma,
                            **metrics
                        })
                        
                    except Exception as e:
                        print(f"\n⚠️  Error processing {img_name} (n_bits={n_bits}, sigma={sigma}): {e}")
                    
                    pbar.update(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n✅ Results saved to: {CSV_PATH}")
    print(f"   Total records: {len(df)}")
    
    return df


# ----------------------------------------------------------------------
# Visualization Functions
# ----------------------------------------------------------------------

def plot_results(df):
    """
    Generate visualization plots from experimental data.
    """
    if df is None or len(df) == 0:
        print("No data to plot.")
        return
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    
    # Calculate average metrics across all images
    df_avg = df.groupby(["n_bits", "sigma"]).agg({
        "sim_local": "mean",
        "sim_rx": "mean",
        "uncertainty": "mean",
        "semantic_degradation": "mean",
    }).reset_index()
    
    # Plot 1: Rate-Distortion Curve (fixed sigma)
    print("\nGenerating plots...")
    
    plt.figure(figsize=(12, 8))
    for sigma in sorted(df_avg["sigma"].unique()):
        data = df_avg[df_avg["sigma"] == sigma]
        plt.plot(data["n_bits"], data["sim_rx"], marker="o", label=f"σ={sigma:.2f}", linewidth=2)
    
    plt.xlabel("Quantization Bits (n_bits)", fontsize=12)
    plt.ylabel("CLIP Semantic Similarity (sim_rx)", fontsize=12)
    plt.title("Rate-Semantic Fidelity: Impact of Quantization", fontsize=14, fontweight="bold")
    plt.legend(title="Noise Level", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rate_distortion_quantization.png", dpi=300)
    print(f"  ✓ Saved: {PLOTS_DIR / 'rate_distortion_quantization.png'}")
    plt.close()
    
    # Plot 2: Noise Robustness (fixed n_bits)
    plt.figure(figsize=(12, 8))
    for n_bits in sorted(df_avg["n_bits"].unique()):
        data = df_avg[df_avg["n_bits"] == n_bits]
        plt.plot(data["sigma"], data["sim_rx"], marker="s", label=f"{n_bits}-bit", linewidth=2)
    
    plt.xlabel("Channel Noise (σ)", fontsize=12)
    plt.ylabel("CLIP Semantic Similarity (sim_rx)", fontsize=12)
    plt.title("Noise Robustness: Impact of Channel Degradation", fontsize=14, fontweight="bold")
    plt.legend(title="Quantization", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "noise_robustness.png", dpi=300)
    print(f"  ✓ Saved: {PLOTS_DIR / 'noise_robustness.png'}")
    plt.close()
    
    # Plot 3: Heatmap of semantic degradation
    pivot = df_avg.pivot(index="sigma", columns="n_bits", values="semantic_degradation")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", cbar_kws={"label": "Degradation"})
    plt.xlabel("Quantization Bits (n_bits)", fontsize=12)
    plt.ylabel("Channel Noise (σ)", fontsize=12)
    plt.title("Semantic Degradation Heatmap\n(sim_local - sim_rx)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "degradation_heatmap.png", dpi=300)
    print(f"  ✓ Saved: {PLOTS_DIR / 'degradation_heatmap.png'}")
    plt.close()
    
    # Plot 4: Combined 3D view (if possible)
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(df_avg["n_bits"], df_avg["sigma"], df_avg["sim_rx"],
                           c=df_avg["sim_rx"], cmap="viridis", s=100, alpha=0.6)
        
        ax.set_xlabel("Quantization Bits", fontsize=10)
        ax.set_ylabel("Noise σ", fontsize=10)
        ax.set_zlabel("Similarity", fontsize=10)
        ax.set_title("3D Rate-Noise-Distortion Space", fontsize=12, fontweight="bold")
        
        fig.colorbar(scatter, ax=ax, label="CLIP Similarity", shrink=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "3d_rate_noise_distortion.png", dpi=300)
        print(f"  ✓ Saved: {PLOTS_DIR / '3d_rate_noise_distortion.png'}")
        plt.close()
    except Exception as e:
        print(f"  ⚠️  Couldn't generate 3D plot: {e}")
    
    print("\n✅ All plots generated successfully!")


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("="*70)
    print("Quantization & Noise Impact Experiment")
    print("="*70)
    print()
    
    # Check if data directory exists and has images
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"Created data directory: {DATA_DIR}")
        print(f"⚠️  Please add images to '{DATA_DIR}/' before running the experiment.")
        print("   You can copy your 'people2.png' or other images there.")
        sys.exit(0)
    
    # Run experiment
    df = run_experiment()
    
    # Generate plots
    if df is not None:
        plot_results(df)
        
        # Print summary statistics
        print("\n" + "="*70)
        print("Summary Statistics")
        print("="*70)
        print(df.groupby(["n_bits", "sigma"])[["sim_rx", "semantic_degradation"]].mean())
        print()
        
        print("✅ Experiment completed successfully!")
        print(f"   Results: {CSV_PATH}")
        print(f"   Plots: {PLOTS_DIR}/")
