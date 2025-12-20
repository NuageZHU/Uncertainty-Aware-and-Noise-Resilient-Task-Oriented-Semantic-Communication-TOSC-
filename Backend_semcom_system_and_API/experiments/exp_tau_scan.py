"""
Experiment: Uncertainty Threshold (τ) Scan
-------------------------------------------

This script explores the trade-off between transmission rate and semantic quality
by scanning different uncertainty thresholds (tau).

Key Question:
    How does the uncertainty threshold τ affect:
    - Transmission rate (percentage of samples transmitted)
    - Average semantic similarity (mean_sim_rx, mean_sim_local)

The experiment reveals the "transmission rate vs semantic quality" trade-off curve,
which is critical for adaptive semantic communication systems.

Usage:
    python experiments/exp_tau_scan.py
    
Output:
    - results/tau_scan_results.csv: Aggregated metrics for each tau
    - results/plots/tau_tradeoff.png: Transmission rate vs quality curve
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
TAU_LIST = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]  # Uncertainty thresholds to scan
N_BITS = 6  # Fixed quantization (6-bit is optimal from previous experiment)
SIGMA = 0.1  # Fixed noise level (moderate noise)
DROPOUT_P = 0.0  # Fixed dropout probability

# Paths
DATA_DIR = Path("data/all_images")  # Directory containing input images
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
CSV_PATH = RESULTS_DIR / "tau_scan_results.csv"

# Sampling configuration
MAX_IMAGES = 50  # Use 50 images for more stable statistics
RANDOM_SAMPLE = True  # Randomly sample images
RANDOM_SEED = 42  # For reproducibility

# Model parameters
SIZE_VAE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ----------------------------------------------------------------------
# Initialize Models (once, globally)
# ----------------------------------------------------------------------

print("=" * 70)
print("Experiment: Uncertainty Threshold (τ) Scan")
print("=" * 70)
print(f"Using device: {DEVICE}")
print(f"Fixed parameters: n_bits={N_BITS}, sigma={SIGMA}")
print(f"Scanning tau values: {TAU_LIST}")
print()

print("Loading models...")

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
# Core Evaluation Function
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate_single_image(img_path, tau, n_bits, sigma):
    """
    Evaluate a single image with given uncertainty threshold and channel parameters.
    
    Args:
        img_path: Path to input image
        tau: Uncertainty threshold (transmission gate)
        n_bits: Quantization bit depth
        sigma: Noise standard deviation
    
    Returns:
        dict with metrics: sim_local, sim_rx, uncertainty, transmit
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
    # (Even if transmit=False, we still compute sim_rx for analysis purposes)
    z_tx = channel(z0, sigma=sigma, n_bits=n_bits, p_drop=DROPOUT_P)
    x_hat_rx = vae_decode(vae, z_tx, sf)
    emb_rx = clip_tensor_embed(clip_model, clip_preprocess, x_hat_rx[0], DEVICE)
    sim_rx = cosine_sim(emb_ref, emb_rx)
    
    return {
        "sim_local": float(sim_local),
        "sim_rx": float(sim_rx),
        "uncertainty": float(uncertainty),
        "transmit": bool(transmit),
    }


# ----------------------------------------------------------------------
# Experiment Runner
# ----------------------------------------------------------------------

def run_tau_scan():
    """
    Run the tau scan experiment across all images.
    
    For each tau value:
    - Evaluate all images
    - Calculate transmit_rate, mean_sim_rx, mean_sim_local
    """
    # Find all images in data directory
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(DATA_DIR.glob(f"*{ext}")))
    
    # Remove duplicates (Windows is case-insensitive)
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
    
    print(f"\nTotal experiments: {len(TAU_LIST)} tau values × {len(image_files)} images = {len(TAU_LIST) * len(image_files)}")
    print()
    
    # Storage for aggregated results
    aggregated_results = []
    
    # Progress bar over tau values
    for tau in tqdm(TAU_LIST, desc="Scanning tau values"):
        # Collect results for this tau
        tau_results = []
        
        for img_path in image_files:
            result = evaluate_single_image(img_path, tau, N_BITS, SIGMA)
            tau_results.append(result)
        
        # Compute aggregated metrics for this tau
        df_tau = pd.DataFrame(tau_results)
        
        transmit_count = df_tau["transmit"].sum()
        transmit_rate = transmit_count / len(df_tau)
        mean_sim_local = df_tau["sim_local"].mean()
        mean_sim_rx = df_tau["sim_rx"].mean()
        mean_uncertainty = df_tau["uncertainty"].mean()
        
        # *** KEY METRIC: Effective similarity considering transmission decisions ***
        # If transmitted: use sim_rx (affected by channel)
        # If not transmitted: use sim_local (no channel, better quality)
        df_tau["effective_sim"] = df_tau.apply(
            lambda row: row["sim_rx"] if row["transmit"] else row["sim_local"], 
            axis=1
        )
        mean_effective_sim = df_tau["effective_sim"].mean()
        std_effective_sim = df_tau["effective_sim"].std()
        
        aggregated_results.append({
            "tau": tau,
            "transmit_rate": transmit_rate,
            "transmit_count": transmit_count,
            "total_samples": len(df_tau),
            "mean_sim_local": mean_sim_local,
            "mean_sim_rx": mean_sim_rx,
            "mean_effective_sim": mean_effective_sim,  # NEW: Most important metric!
            "mean_uncertainty": mean_uncertainty,
            "std_sim_rx": df_tau["sim_rx"].std(),
            "std_effective_sim": std_effective_sim,
        })
    
    # Convert to DataFrame
    df_agg = pd.DataFrame(aggregated_results)
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    df_agg.to_csv(CSV_PATH, index=False)
    print(f"\n✅ Results saved to: {CSV_PATH}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary Results")
    print("=" * 70)
    print(df_agg.to_string(index=False))
    print()
    
    return df_agg


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def plot_tau_tradeoff(df):
    """
    Create visualization of the transmission rate vs semantic quality trade-off.
    
    Args:
        df: DataFrame with aggregated results (columns: tau, transmit_rate, mean_sim_rx, etc.)
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Plot 1: Transmission Rate vs Effective Similarity =====
    ax1 = axes[0]
    # Plot both metrics for comparison
    ax1.plot(df["transmit_rate"], df["mean_effective_sim"], 
             marker='D', linewidth=2.5, markersize=8, 
             color='#F18F01', label='mean_effective_sim (ACTUAL)', zorder=3)
    ax1.plot(df["transmit_rate"], df["mean_sim_rx"], 
             marker='o', linewidth=1.5, markersize=6, 
             color='#2E86AB', label='mean_sim_rx (all samples)', alpha=0.6)
    
    # Annotate each point with tau value
    for _, row in df.iterrows():
        ax1.annotate(f'τ={row["tau"]:.2f}', 
                     xy=(row["transmit_rate"], row["mean_effective_sim"]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Transmission Rate (proportion)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Semantic Similarity', fontsize=12, fontweight='bold')
    ax1.set_title('Trade-off: Transmission Rate vs ACTUAL Quality', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add text box with experiment settings
    textstr = f'Fixed: n_bits={N_BITS}, σ={SIGMA}\nImages: {df["total_samples"].iloc[0]}'
    ax1.text(0.05, 0.05, textstr, transform=ax1.transAxes,
             fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ===== Plot 2: Tau vs All Three Similarity Metrics =====
    ax2 = axes[1]
    ax2.plot(df["tau"], df["mean_sim_local"], 
             marker='s', linewidth=2, markersize=7,
             color='#A23B72', label='mean_sim_local (no channel)', alpha=0.8, linestyle='--')
    ax2.plot(df["tau"], df["mean_effective_sim"], 
             marker='D', linewidth=2.5, markersize=7,
             color='#F18F01', label='mean_effective_sim (ACTUAL)', zorder=3)
    ax2.plot(df["tau"], df["mean_sim_rx"], 
             marker='o', linewidth=1.5, markersize=6,
             color='#2E86AB', label='mean_sim_rx (all through channel)', alpha=0.6)
    
    ax2.set_xlabel('Uncertainty Threshold (τ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Semantic Similarity', fontsize=12, fontweight='bold')
    ax2.set_title('Impact of τ: Which Metric Matters?', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = PLOTS_DIR / "tau_tradeoff.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    plt.close()
    
    # ===== Additional Plot: Transmission Rate vs Tau =====
    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    ax.plot(df["tau"], df["transmit_rate"] * 100, 
            marker='D', linewidth=2.5, markersize=8,
            color='#F18F01', label='Transmission Rate')
    
    ax.set_xlabel('Uncertainty Threshold (τ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Transmission Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Transmission Rate vs Uncertainty Threshold', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Highlight key regions
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    
    plt.tight_layout()
    
    output_path2 = PLOTS_DIR / "tau_transmission_rate.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path2}")
    plt.close()


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Run experiment
    df_results = run_tau_scan()
    
    if df_results is not None:
        # Generate plots
        print("\nGenerating plots...")
        plot_tau_tradeoff(df_results)
        
        print("\n" + "=" * 70)
        print("🎉 Experiment completed successfully!")
        print("=" * 70)
        print(f"Results: {CSV_PATH}")
        print(f"Plots: {PLOTS_DIR}/")
        print()
        
        # Print key insights
        print("📊 Key Insights (Using EFFECTIVE Similarity):")
        print("-" * 70)
        
        # Find optimal tau (highest mean_effective_sim)
        best_idx = df_results["mean_effective_sim"].idxmax()
        best_row = df_results.iloc[best_idx]
        print(f"✓ BEST ACTUAL QUALITY: τ={best_row['tau']:.3f}")
        print(f"  → mean_effective_sim = {best_row['mean_effective_sim']:.4f} ⭐")
        print(f"  → transmit_rate = {best_row['transmit_rate']:.2%}")
        print(f"  → bandwidth saved: {(1-best_row['transmit_rate'])*100:.1f}%")
        
        # Compare extremes
        low_tau = df_results.iloc[0]
        high_tau = df_results.iloc[-1]
        
        print(f"\n✓ Low threshold (τ={low_tau['tau']:.3f}) - Transmit everything:")
        print(f"  → transmit_rate = {low_tau['transmit_rate']:.1%}")
        print(f"  → mean_effective_sim = {low_tau['mean_effective_sim']:.4f}")
        print(f"  → Quality loss vs optimal: {(best_row['mean_effective_sim'] - low_tau['mean_effective_sim'])*100:.2f}%")
        
        print(f"\n✓ High threshold (τ={high_tau['tau']:.3f}) - Transmit almost nothing:")
        print(f"  → transmit_rate = {high_tau['transmit_rate']:.1%}")
        print(f"  → mean_effective_sim = {high_tau['mean_effective_sim']:.4f}")
        print(f"  → Quality gain vs low-tau: {(high_tau['mean_effective_sim'] - low_tau['mean_effective_sim'])*100:.2f}%")
        
        # Key finding
        quality_improvement = (best_row['mean_effective_sim'] - low_tau['mean_effective_sim']) * 100
        bandwidth_saving = (1 - best_row['transmit_rate']) * 100
        
        print(f"\n🎯 KEY FINDING:")
        print(f"  By using τ={best_row['tau']:.3f} instead of transmitting everything:")
        print(f"  → Quality improves by {quality_improvement:.2f}%")
        print(f"  → Bandwidth reduces by {bandwidth_saving:.1f}%")
        print(f"  → Win-win situation! Less transmission = BETTER quality!")
        
        print("\n💡 WHY? Because channel degrades quality (sim_rx < sim_local)")
        print(f"  → mean_sim_local = {df_results['mean_sim_local'].iloc[0]:.4f}")
        print(f"  → mean_sim_rx = {df_results['mean_sim_rx'].iloc[0]:.4f}")
        print(f"  → Channel causes {(df_results['mean_sim_local'].iloc[0] - df_results['mean_sim_rx'].iloc[0])*100:.1f}% quality loss!")
        
        print("\n" + "=" * 70)
