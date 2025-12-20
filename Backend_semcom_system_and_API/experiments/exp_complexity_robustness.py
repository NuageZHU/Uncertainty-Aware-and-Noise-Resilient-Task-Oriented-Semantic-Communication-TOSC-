"""
Experiment 4: Semantic Robustness vs Image Complexity

Objective:
    Group images by data-driven statistical properties (not human labels) and study
    how robustness under latent compression + channel noise varies with image complexity.

Grouping strategy (by complexity score):
    - Low complexity: low edge density, simple colors, mostly low-frequency content
    - Medium complexity: moderate textures and color diversity
    - High complexity: high edge density, rich colors, many high-frequency details (e.g., text/texture)

Fixed parameters:
    - n_bits = 6 (best from Experiment 1)
    - sigma = 0.1 (moderate channel quality)
    - tau = 0.05 (moderate transmission threshold)

Outputs:
    - results/complexity_robustness_results.csv
    - results/plots/complexity_robustness_bar.png
    - results/plots/complexity_distribution.png
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision import transforms
from diffusers import AutoencoderKL
import open_clip

from semcom_utils import (
    vae_encode, vae_decode, channel,
    clip_img_embed, clip_tensor_embed, cosine_sim
)

# ============================================================================
# Experiment configuration
# ============================================================================
N_BITS = 6          # Quantization bits (fixed)
SIGMA = 0.1         # Channel noise std (fixed)
TAU = 0.05          # Uncertainty threshold (fixed)
DROPOUT_P = 0.0     # Dropout probability (fixed)
SIZE_VAE = 512      # VAE input size

# Complexity grouping percentiles (based on edge density distribution)
LOW_COMPLEXITY_PERCENTILE = 33   # below 33% → low complexity
HIGH_COMPLEXITY_PERCENTILE = 67  # above 67% → high complexity

# Paths
IMAGE_FOLDER = PROJECT_ROOT / "data" / "all_images"
RESULTS_FOLDER = PROJECT_ROOT / "results"
PLOTS_FOLDER = RESULTS_FOLDER / "plots"

# ============================================================================
# Image complexity metrics
# ============================================================================

def compute_image_complexity(img_path):
    """
    Compute multiple image complexity metrics.

    Returns a dict with:
    {
        'edge_density': edge density in [0, 1]
        'color_variance': average channel std across RGB
        'high_freq_ratio': high-frequency energy proportion
        'complexity_score': aggregated complexity score
    }
    """
    try:
        # Load image via PIL and convert to numpy/OpenCV format
        pil_img = Image.open(img_path).convert("RGB")
        img_rgb = np.array(pil_img)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"\n⚠️  Error loading {img_path.name}: {e}")
        return None
    
    # ===== 1) Edge density =====
    # Using Canny edge detector
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = edges.mean() / 255.0  # normalize to [0, 1]
    
    # ===== 2) Color diversity =====
    # Average std across RGB channels
    color_variance = img_rgb.std(axis=(0, 1)).mean()
    
    # ===== 3) High-frequency energy ratio =====
    # Frequency analysis via FFT
    fft = np.fft.fft2(img_gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    
    # Define high-frequency region (distance from center > radius)
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = min(rows, cols) // 4  # radius for high frequency mask
    
    # Build high-frequency mask
    y, x = np.ogrid[:rows, :cols]
    mask = (x - ccol)**2 + (y - crow)**2 > radius**2
    
    high_freq_energy = magnitude_spectrum[mask].sum()
    total_energy = magnitude_spectrum.sum()
    high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
    
    # ===== 4) Aggregated complexity score =====
    # Weighted sum (edge density has the highest weight)
    complexity_score = (
        0.5 * edge_density +           # edge density (dominant)
        0.3 * (color_variance / 100) + # normalized color variance
        0.2 * high_freq_ratio          # high-frequency ratio
    )
    
    return {
        'edge_density': edge_density,
        'color_variance': color_variance,
        'high_freq_ratio': high_freq_ratio,
        'complexity_score': complexity_score,
    }


def classify_by_complexity(complexity_scores, low_pct, high_pct):
    """
    Compute percentile thresholds for complexity grouping.

    Args:
        complexity_scores: list/array of complexity scores
        low_pct: percentile for low-complexity threshold
        high_pct: percentile for high-complexity threshold

    Returns:
        (low_threshold, high_threshold)
    """
    low_threshold = np.percentile(complexity_scores, low_pct)
    high_threshold = np.percentile(complexity_scores, high_pct)
    return low_threshold, high_threshold


# ============================================================================
# Main experiment
# ============================================================================

def run_complexity_experiment():
    """Run the robustness vs complexity experiment."""
    
    print("=" * 70)
    print("Experiment 4: Image Complexity Robustness Analysis")
    print("=" * 70)
    
    # Ensure output folders exist
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\n[1/5] Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    sf = getattr(vae.config, "scaling_factor", 1.0)
    
    # Load CLIP
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="openai",
        device=device
    )
    clip_model.eval()
    
    print(f"Device: {device}")
    print(f"Fixed parameters: n_bits={N_BITS}, sigma={SIGMA}, tau={TAU}")
    
    # Collect images
    print("\n[2/5] Collecting images...")
    image_paths = []
    # Use a set to avoid duplicates (case-insensitive)
    seen_files = set()
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for path in IMAGE_FOLDER.glob(ext):
            if path.name.lower() not in seen_files:
                image_paths.append(path)
                seen_files.add(path.name.lower())
    
    if len(image_paths) == 0:
        print(f"❌ No images found in {IMAGE_FOLDER}")
        print(f"   Please add some images to the data/ folder.")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Compute complexity for all images
    print("\n[3/5] Computing image complexity metrics...")
    complexity_data = []
    
    for img_path in tqdm(image_paths, desc="Computing complexity"):
        complexity = compute_image_complexity(img_path)
        if complexity is not None:
            complexity_data.append({
                'img_path': img_path,
                'img_name': img_path.name,
                **complexity
            })
    
    if len(complexity_data) == 0:
        print("❌ No valid images found")
        return
    
    df_complexity = pd.DataFrame(complexity_data)
    
    # Group by complexity
    print("\n[4/5] Classifying images by complexity...")
    complexity_scores = df_complexity['complexity_score'].values
    low_thresh, high_thresh = classify_by_complexity(
        complexity_scores, 
        LOW_COMPLEXITY_PERCENTILE, 
        HIGH_COMPLEXITY_PERCENTILE
    )
    
    def assign_complexity_group(score):
        if score < low_thresh:
            return 'low'
        elif score < high_thresh:
            return 'medium'
        else:
            return 'high'
    
    df_complexity['complexity_group'] = df_complexity['complexity_score'].apply(
        assign_complexity_group
    )
    
    # Report grouping
    group_counts = df_complexity['complexity_group'].value_counts()
    print(f"\nComplexity grouping:")
    print(f"  Low complexity:    {group_counts.get('low', 0)} images (score < {low_thresh:.3f})")
    print(f"  Medium complexity: {group_counts.get('medium', 0)} images ({low_thresh:.3f} ≤ score < {high_thresh:.3f})")
    print(f"  High complexity:   {group_counts.get('high', 0)} images (score ≥ {high_thresh:.3f})")
    
    # Run semantic transmission pipeline
    print("\n[5/5] Running semantic transmission experiments...")
    results = []
    
    # Preprocessing for VAE
    to_tensor = transforms.ToTensor()
    resize_vae = transforms.Resize((SIZE_VAE, SIZE_VAE), antialias=True)
    
    for idx, row in tqdm(df_complexity.iterrows(), total=len(df_complexity), desc="Processing images"):
        img_path = row['img_path']
        
        try:
            # 1) Load and preprocess image
            pil_img = Image.open(img_path).convert("RGB")
            
            # 2) Prepare VAE input
            x = to_tensor(pil_img).unsqueeze(0).to(device)
            x = resize_vae(x)
            x = (x - 0.5) * 2.0  # normalize to [-1, 1]
            
            # 3) Encode to latent space
            with torch.no_grad():
                z_local = vae_encode(vae, x, sf)
            
            # 4) Local decode (no channel)
            with torch.no_grad():
                x_recon_local = vae_decode(vae, z_local, sf)
            
            # 5) Compute local semantic similarity
            # Ensure x_recon_local is 3D tensor (C, H, W)
            x_recon_local_3d = x_recon_local.squeeze(0) if x_recon_local.dim() == 4 else x_recon_local
            sim_local = cosine_sim(
                clip_img_embed(clip_model, clip_preprocess, pil_img, device),
                clip_tensor_embed(clip_model, clip_preprocess, x_recon_local_3d, device)
            )
            
            # 6) Uncertainty
            uncertainty = 1.0 - sim_local
            
            # 7) Transmission decision
            transmit = (uncertainty > TAU)
            
            # 8) If transmit, send through channel
            if transmit:
                with torch.no_grad():
                    z_rx = channel(z_local, sigma=SIGMA, n_bits=N_BITS, p_drop=DROPOUT_P)
                    x_recon_rx = vae_decode(vae, z_rx, sf)
                
                # Ensure x_recon_rx is 3D tensor
                x_recon_rx_3d = x_recon_rx.squeeze(0) if x_recon_rx.dim() == 4 else x_recon_rx
                sim_rx = cosine_sim(
                    clip_img_embed(clip_model, clip_preprocess, pil_img, device),
                    clip_tensor_embed(clip_model, clip_preprocess, x_recon_rx_3d, device)
                )
            else:
                sim_rx = sim_local  # if not transmitted, receiver uses local quality
            
            # 9) Effective similarity
            effective_sim = sim_rx if transmit else sim_local
            
            results.append({
                'img_name': row['img_name'],
                'complexity_group': row['complexity_group'],
                'edge_density': row['edge_density'],
                'color_variance': row['color_variance'],
                'high_freq_ratio': row['high_freq_ratio'],
                'complexity_score': row['complexity_score'],
                'sim_local': sim_local,
                'sim_rx': sim_rx,
                'transmit': transmit,
                'uncertainty': uncertainty,
                'effective_sim': effective_sim,
                'semantic_degradation': sim_local - sim_rx,
            })
            
        except Exception as e:
            print(f"\n⚠️  Error processing {img_path.name}: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    
    # Aggregate by complexity group
    print("\n" + "=" * 70)
    print("Aggregated Results by Complexity Group")
    print("=" * 70)
    
    aggregated = df_results.groupby('complexity_group').agg({
        'img_name': 'count',
        'complexity_score': 'mean',
        'sim_local': 'mean',
        'sim_rx': 'mean',
        'effective_sim': 'mean',
        'transmit': 'sum',
        'semantic_degradation': 'mean',
    }).rename(columns={'img_name': 'n_samples', 'transmit': 'n_transmit'})
    
    aggregated['transmit_rate'] = aggregated['n_transmit'] / aggregated['n_samples']
    aggregated = aggregated[['n_samples', 'complexity_score', 'sim_local', 'sim_rx', 
                             'effective_sim', 'transmit_rate', 'semantic_degradation']]
    
    # Reorder (low → medium → high)
    complexity_order = ['low', 'medium', 'high']
    aggregated = aggregated.reindex([g for g in complexity_order if g in aggregated.index])
    
    print(aggregated.to_string())
    
    # Save detailed results
    detail_csv_path = RESULTS_FOLDER / "complexity_robustness_detail.csv"
    df_results.to_csv(detail_csv_path, index=False)
    print(f"\n✅ Detailed results saved: {detail_csv_path}")
    
    # Save aggregated results
    agg_csv_path = RESULTS_FOLDER / "complexity_robustness_results.csv"
    aggregated.to_csv(agg_csv_path)
    print(f"✅ Aggregated results saved: {agg_csv_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_complexity_analysis(df_results, aggregated, complexity_order)
    
    print("\n" + "=" * 70)
    print("🎉 Experiment completed!")
    print("=" * 70)
    print(f"Results: {agg_csv_path}")
    print(f"Plots: {PLOTS_FOLDER}/")
    
    # Print key insights
    print("\n" + "=" * 70)
    print("💡 Key Insights:")
    print("=" * 70)
    
    if 'high' in aggregated.index and 'low' in aggregated.index:
        sim_rx_high = aggregated.loc['high', 'sim_rx']
        sim_rx_low = aggregated.loc['low', 'sim_rx']
        degradation_high = aggregated.loc['high', 'semantic_degradation']
        degradation_low = aggregated.loc['low', 'semantic_degradation']
        
        print(f"1. Transmission quality (sim_rx):")
        print(f"   Low complexity:  {sim_rx_low:.4f}")
        print(f"   High complexity: {sim_rx_high:.4f}")
        print(f"   Difference: {(sim_rx_high - sim_rx_low):.4f} ({(sim_rx_high/sim_rx_low - 1)*100:+.1f}%)")
        
        print(f"\n2. Semantic degradation:")
        print(f"   Low complexity:  {degradation_low:.4f}")
        print(f"   High complexity: {degradation_high:.4f}")
        print(f"   Difference: {(degradation_high - degradation_low):.4f}")
        
        if sim_rx_high < sim_rx_low:
            print("\n   → High-complexity images are MORE SENSITIVE to compression+noise")
        else:
            print("\n   → No significant sensitivity difference detected")


# ============================================================================
# Visualization
# ============================================================================

def plot_complexity_analysis(df_results, aggregated, complexity_order):
    """Generate plots for complexity analysis."""
    
    # Style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    
    # ===== Fig 1: Distributions =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1.1 Complexity score
    ax = axes[0, 0]
    for group in complexity_order:
        if group in df_results['complexity_group'].values:
            data = df_results[df_results['complexity_group'] == group]['complexity_score']
            ax.hist(data, alpha=0.6, label=group.capitalize(), bins=15)
    ax.set_xlabel('Complexity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('(a) Complexity Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.2 Edge density
    ax = axes[0, 1]
    for group in complexity_order:
        if group in df_results['complexity_group'].values:
            data = df_results[df_results['complexity_group'] == group]['edge_density']
            ax.hist(data, alpha=0.6, label=group.capitalize(), bins=15)
    ax.set_xlabel('Edge Density')
    ax.set_ylabel('Frequency')
    ax.set_title('(b) Edge Density Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.3 Color variance
    ax = axes[1, 0]
    for group in complexity_order:
        if group in df_results['complexity_group'].values:
            data = df_results[df_results['complexity_group'] == group]['color_variance']
            ax.hist(data, alpha=0.6, label=group.capitalize(), bins=15)
    ax.set_xlabel('Color Variance')
    ax.set_ylabel('Frequency')
    ax.set_title('(c) Color Variance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.4 High-frequency ratio
    ax = axes[1, 1]
    for group in complexity_order:
        if group in df_results['complexity_group'].values:
            data = df_results[df_results['complexity_group'] == group]['high_freq_ratio']
            ax.hist(data, alpha=0.6, label=group.capitalize(), bins=15)
    ax.set_xlabel('High-Frequency Energy Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('(d) High-Frequency Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_FOLDER / "complexity_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved: {plot_path}")
    
    # ===== Fig 2: Robustness bar charts =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Prepare data
    groups = [g for g in complexity_order if g in aggregated.index]
    x_pos = np.arange(len(groups))
    
        # 2.1 Average similarities
    ax = axes[0, 0]
    width = 0.25
    ax.bar(x_pos - width, [aggregated.loc[g, 'sim_local'] for g in groups], 
           width, label='Local (no channel)', color='#2ecc71')
    ax.bar(x_pos, [aggregated.loc[g, 'sim_rx'] for g in groups], 
           width, label='After channel', color='#e74c3c')
    ax.bar(x_pos + width, [aggregated.loc[g, 'effective_sim'] for g in groups], 
           width, label='Effective (adaptive)', color='#3498db')
    ax.set_xlabel('Complexity Group')
    ax.set_ylabel('Semantic Similarity')
    ax.set_title('(a) Semantic Quality by Complexity')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.6, 1.0])
    
    # 2.2 Semantic degradation
    ax = axes[0, 1]
    degradation = [aggregated.loc[g, 'semantic_degradation'] for g in groups]
    colors = ['#27ae60', '#f39c12', '#c0392b'][:len(groups)]
    bars = ax.bar(x_pos, degradation, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Complexity Group')
    ax.set_ylabel('Semantic Degradation (sim_local - sim_rx)')
    ax.set_title('(b) Quality Loss by Complexity')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2.3 Transmission rate
    ax = axes[1, 0]
    transmit_rates = [aggregated.loc[g, 'transmit_rate'] * 100 for g in groups]
    bars = ax.bar(x_pos, transmit_rates, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Complexity Group')
    ax.set_ylabel('Transmission Rate (%)')
    ax.set_title(f'(c) Transmission Rate (tau={TAU})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2.4 Complexity scores
    ax = axes[1, 1]
    complexity_scores = [aggregated.loc[g, 'complexity_score'] for g in groups]
    bars = ax.bar(x_pos, complexity_scores, color='#34495e', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Complexity Group')
    ax.set_ylabel('Average Complexity Score')
    ax.set_title('(d) Complexity Score by Group')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = PLOTS_FOLDER / "complexity_robustness_bar.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved: {plot_path}")
    
    # ===== Fig 3: Scatter plots (complexity vs metrics) =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 3.1 Complexity vs sim_rx
    ax = axes[0]
    for group in complexity_order:
        if group in df_results['complexity_group'].values:
            data = df_results[df_results['complexity_group'] == group]
            ax.scatter(data['complexity_score'], data['sim_rx'], 
                      alpha=0.6, label=group.capitalize(), s=50)
    ax.set_xlabel('Complexity Score')
    ax.set_ylabel('Semantic Similarity (sim_rx)')
    ax.set_title('Complexity vs Transmission Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Trend line
    z = np.polyfit(df_results['complexity_score'], df_results['sim_rx'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_results['complexity_score'].min(), 
                         df_results['complexity_score'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label='Trend')
    
    # 3.2 Complexity vs semantic degradation
    ax = axes[1]
    for group in complexity_order:
        if group in df_results['complexity_group'].values:
            data = df_results[df_results['complexity_group'] == group]
            ax.scatter(data['complexity_score'], data['semantic_degradation'], 
                      alpha=0.6, label=group.capitalize(), s=50)
    ax.set_xlabel('Complexity Score')
    ax.set_ylabel('Semantic Degradation')
    ax.set_title('Complexity vs Quality Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Trend line
    z = np.polyfit(df_results['complexity_score'], df_results['semantic_degradation'], 1)
    p = np.poly1d(z)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label='Trend')
    
    plt.tight_layout()
    plot_path = PLOTS_FOLDER / "complexity_vs_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved: {plot_path}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    run_complexity_experiment()
