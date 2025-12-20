"""
Experiment 3: Sigma × Tau Joint Exploration
--------------------------------------------

This experiment explores the relationship between channel quality (sigma)
and optimal uncertainty threshold (tau) by scanning both parameters.

Key Question:
    How does the optimal tau change with channel quality?
    - Good channel (low sigma): Should we transmit more (low tau)?
    - Bad channel (high sigma): Should we transmit less (high tau)?

Usage:
    python experiments/exp_sigma_tau_joint.py
    
Output:
    - results/sigma_tau_joint_results.csv: Full data for all combinations
    - results/plots/sigma_tau_heatmap.png: Quality heatmap
    - results/plots/optimal_tau_vs_sigma.png: Optimal tau for each sigma
"""

import sys
import os
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

from semcom_utils import (
    vae_encode, vae_decode, channel,
    clip_img_embed, clip_tensor_embed, cosine_sim
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Experimental parameters - JOINT EXPLORATION
SIGMA_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]  # Channel noise levels
TAU_LIST = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]  # Uncertainty thresholds
N_BITS = 6  # Fixed quantization (optimal from Exp 1)
DROPOUT_P = 0.0

# Paths
DATA_DIR = Path("data/all_images")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
CSV_PATH = RESULTS_DIR / "sigma_tau_joint_results.csv"

# Sampling
MAX_IMAGES = 30  # Use 30 images for faster execution (6×7×30 = 1260 evaluations)
RANDOM_SAMPLE = True
RANDOM_SEED = 42

# Model parameters
SIZE_VAE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)


# ----------------------------------------------------------------------
# Initialize Models
# ----------------------------------------------------------------------

print("=" * 70)
print("Experiment 3: Sigma × Tau Joint Exploration")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Sigma values: {SIGMA_LIST}")
print(f"Tau values: {TAU_LIST}")
print(f"Total combinations: {len(SIGMA_LIST)} × {len(TAU_LIST)} = {len(SIGMA_LIST) * len(TAU_LIST)}")
print()

print("Loading models...")
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


# ----------------------------------------------------------------------
# Core Function
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate_single_image(img_path, tau, sigma, n_bits):
    """Evaluate one image with given tau and sigma."""
    pil = Image.open(img_path).convert("RGB")
    x_in = pre_vae(pil).unsqueeze(0).to(DEVICE)
    
    # Local reconstruction
    z0 = vae_encode(vae, x_in, sf)
    x_hat0 = vae_decode(vae, z0, sf)
    
    emb_ref = clip_img_embed(clip_model, clip_preprocess, pil, DEVICE)
    emb_loc = clip_tensor_embed(clip_model, clip_preprocess, x_hat0[0], DEVICE)
    sim_local = cosine_sim(emb_ref, emb_loc)
    
    # Transmission decision
    uncertainty = 1.0 - sim_local
    transmit = uncertainty > tau
    
    # Channel reconstruction
    z_tx = channel(z0, sigma=sigma, n_bits=n_bits, p_drop=DROPOUT_P)
    x_hat_rx = vae_decode(vae, z_tx, sf)
    emb_rx = clip_tensor_embed(clip_model, clip_preprocess, x_hat_rx[0], DEVICE)
    sim_rx = cosine_sim(emb_ref, emb_rx)
    
    # Effective similarity (KEY METRIC)
    effective_sim = sim_rx if transmit else sim_local
    
    return {
        "sim_local": float(sim_local),
        "sim_rx": float(sim_rx),
        "effective_sim": float(effective_sim),
        "uncertainty": float(uncertainty),
        "transmit": bool(transmit),
    }


# ----------------------------------------------------------------------
# Experiment Runner
# ----------------------------------------------------------------------

def run_joint_experiment():
    """Run sigma × tau joint experiment."""
    # Load images
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(DATA_DIR.glob(f"*{ext}")))
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"⚠️  No images found in {DATA_DIR}")
        return None
    
    print(f"Found {len(image_files)} images")
    
    if MAX_IMAGES and MAX_IMAGES < len(image_files):
        if RANDOM_SAMPLE:
            np.random.seed(RANDOM_SEED)
            image_files = list(np.random.choice(image_files, MAX_IMAGES, replace=False))
            print(f"Sampled {MAX_IMAGES} images (seed={RANDOM_SEED})")
        else:
            image_files = image_files[:MAX_IMAGES]
    
    n_images = len(image_files)
    total_evals = len(SIGMA_LIST) * len(TAU_LIST) * n_images
    print(f"\nTotal evaluations: {len(SIGMA_LIST)}×{len(TAU_LIST)}×{n_images} = {total_evals}")
    print(f"Estimated time: ~{total_evals * 2 / 60:.1f} minutes\n")
    
    # Storage
    all_results = []
    aggregated_results = []
    
    # Main loop
    with tqdm(total=len(SIGMA_LIST) * len(TAU_LIST), desc="Scanning (sigma, tau) pairs") as pbar:
        for sigma in SIGMA_LIST:
            for tau in TAU_LIST:
                # Evaluate all images for this (sigma, tau) pair
                pair_results = []
                for img_path in image_files:
                    result = evaluate_single_image(img_path, tau, sigma, N_BITS)
                    result["sigma"] = sigma
                    result["tau"] = tau
                    result["img_name"] = img_path.name
                    pair_results.append(result)
                
                # Aggregate for this pair
                df_pair = pd.DataFrame(pair_results)
                transmit_rate = df_pair["transmit"].sum() / len(df_pair)
                
                aggregated_results.append({
                    "sigma": sigma,
                    "tau": tau,
                    "transmit_rate": transmit_rate,
                    "mean_sim_local": df_pair["sim_local"].mean(),
                    "mean_sim_rx": df_pair["sim_rx"].mean(),
                    "mean_effective_sim": df_pair["effective_sim"].mean(),
                    "std_effective_sim": df_pair["effective_sim"].std(),
                    "n_samples": len(df_pair),
                })
                
                all_results.extend(pair_results)
                pbar.update(1)
    
    # Convert to DataFrames
    df_all = pd.DataFrame(all_results)
    df_agg = pd.DataFrame(aggregated_results)
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    df_agg.to_csv(CSV_PATH, index=False)
    print(f"\n✅ Aggregated results saved to: {CSV_PATH}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Sample Results (first 10 rows)")
    print("=" * 70)
    print(df_agg.head(10).to_string(index=False))
    print("\n...")
    
    return df_agg


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def plot_joint_results(df):
    """Create visualizations for sigma × tau experiment."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    
    # Pivot for heatmap
    pivot_effective = df.pivot(index="tau", columns="sigma", values="mean_effective_sim")
    pivot_transmit = df.pivot(index="tau", columns="sigma", values="transmit_rate")
    
    # ===== Plot 1: Effective Similarity Heatmap =====
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    sns.heatmap(pivot_effective, annot=True, fmt=".3f", cmap="RdYlGn", 
                vmin=0.75, vmax=0.95, ax=ax, cbar_kws={"label": "Mean Effective Sim"})
    
    ax.set_xlabel("Channel Noise (σ)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Uncertainty Threshold (τ)", fontsize=13, fontweight="bold")
    ax.set_title("Quality Heatmap: Effective Similarity for (σ, τ) Combinations", 
                 fontsize=14, fontweight="bold", pad=15)
    
    plt.tight_layout()
    output1 = PLOTS_DIR / "sigma_tau_heatmap_quality.png"
    plt.savefig(output1, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved: {output1}")
    plt.close()
    
    # ===== Plot 2: Transmission Rate Heatmap =====
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    sns.heatmap(pivot_transmit, annot=True, fmt=".2f", cmap="YlOrRd", 
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Transmission Rate"})
    
    ax.set_xlabel("Channel Noise (σ)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Uncertainty Threshold (τ)", fontsize=13, fontweight="bold")
    ax.set_title("Transmission Rate Heatmap for (σ, τ) Combinations", 
                 fontsize=14, fontweight="bold", pad=15)
    
    plt.tight_layout()
    output2 = PLOTS_DIR / "sigma_tau_heatmap_transmit.png"
    plt.savefig(output2, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved: {output2}")
    plt.close()
    
    # ===== Plot 3: Optimal Tau for Each Sigma =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Find optimal tau for each sigma
    optimal_tau_per_sigma = []
    for sigma in SIGMA_LIST:
        df_sigma = df[df["sigma"] == sigma]
        best_idx = df_sigma["mean_effective_sim"].idxmax()
        best_row = df_sigma.loc[best_idx]
        optimal_tau_per_sigma.append({
            "sigma": sigma,
            "optimal_tau": best_row["tau"],
            "max_effective_sim": best_row["mean_effective_sim"],
            "transmit_rate_at_optimal": best_row["transmit_rate"],
        })
    
    df_optimal = pd.DataFrame(optimal_tau_per_sigma)
    
    # Left plot: Optimal tau vs sigma
    ax1 = axes[0]
    ax1.plot(df_optimal["sigma"], df_optimal["optimal_tau"], 
             marker='o', linewidth=2.5, markersize=10, color='#E63946')
    ax1.set_xlabel("Channel Noise (σ)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Optimal Threshold (τ)", fontsize=12, fontweight="bold")
    ax1.set_title("How Optimal τ Changes with Channel Quality", 
                  fontsize=13, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.3)
    
    # Annotate each point
    for _, row in df_optimal.iterrows():
        ax1.annotate(f'τ={row["optimal_tau"]:.2f}', 
                     xy=(row["sigma"], row["optimal_tau"]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Right plot: Quality at optimal tau vs sigma
    ax2 = axes[1]
    ax2.plot(df_optimal["sigma"], df_optimal["max_effective_sim"], 
             marker='D', linewidth=2.5, markersize=10, color='#2A9D8F')
    ax2.set_xlabel("Channel Noise (σ)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Max Effective Similarity", fontsize=12, fontweight="bold")
    ax2.set_title("Best Achievable Quality vs Channel Noise", 
                  fontsize=13, fontweight="bold", pad=15)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output3 = PLOTS_DIR / "optimal_tau_vs_sigma.png"
    plt.savefig(output3, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved: {output3}")
    plt.close()
    
    # ===== Plot 4: Quality vs Tau curves for different Sigma =====
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(SIGMA_LIST)))
    for i, sigma in enumerate(SIGMA_LIST):
        df_sigma = df[df["sigma"] == sigma].sort_values("tau")
        ax.plot(df_sigma["tau"], df_sigma["mean_effective_sim"], 
                marker='o', linewidth=2, markersize=6, 
                color=colors[i], label=f'σ={sigma:.2f}')
    
    ax.set_xlabel("Uncertainty Threshold (τ)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Effective Similarity", fontsize=12, fontweight="bold")
    ax.set_title("Quality vs τ for Different Channel Noise Levels", 
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(title="Channel Noise", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output4 = PLOTS_DIR / "quality_vs_tau_by_sigma.png"
    plt.savefig(output4, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved: {output4}")
    plt.close()
    
    return df_optimal


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    df_results = run_joint_experiment()
    
    if df_results is not None:
        print("\nGenerating plots...")
        df_optimal = plot_joint_results(df_results)
        
        print("\n" + "=" * 70)
        print("🎉 Experiment completed!")
        print("=" * 70)
        print(f"Results: {CSV_PATH}")
        print(f"Plots: {PLOTS_DIR}/")
        
        print("\n📊 Optimal Tau for Each Sigma:")
        print("-" * 70)
        print(df_optimal.to_string(index=False))
        
        print("\n💡 Key Insight:")
        print(f"  As channel gets worse (σ↑), optimal τ should {'increase' if df_optimal['optimal_tau'].iloc[-1] > df_optimal['optimal_tau'].iloc[0] else 'decrease'}.")
        print(f"  σ={df_optimal['sigma'].iloc[0]:.2f}: τ*={df_optimal['optimal_tau'].iloc[0]:.2f}")
        print(f"  σ={df_optimal['sigma'].iloc[-1]:.2f}: τ*={df_optimal['optimal_tau'].iloc[-1]:.2f}")
        print("\n" + "=" * 70)
