# Experiments Guide
# Experiments Guide

## 📁 Project Structure

```
semantic-channel-modeling-main/
├── semcom_main.py              # Main demo script
├── semcom_utils.py             # Shared utility functions
├── experiments/                # Experiment scripts
│   ├── exp_quantization_noise.py   # Exp1: Quantization + noise analysis
│   ├── exp_tau_scan.py             # Exp2: Uncertainty threshold scan
│   ├── exp_sigma_tau_joint.py      # Exp3: Sigma × Tau joint optimization (recommended)
│   └── exp_quick_test.py           # Quick sanity check
├── data/                       # Test images
├── results/                    # Experiment outputs
│   ├── *.csv                   # Data tables
│   └── plots/                  # Visualizations
└── out_demo/                   # Demo reconstructions
```

## 🚀 Quick Start

### 1) Quick test (use an included image)

Run a minimal sweep on the bundled `people2.png`:

```powershell
python experiments/exp_quick_test.py
```

Outputs:
- `results/quick_test_results.csv` — summary table
- Aggregated metrics in the terminal

What it runs:
- 4 quantization levels: 2, 4, 6, 8 bits
- 3 noise levels: σ = 0.0, 0.1, 0.2
- 12 configurations total
- Runtime: ~1–2 minutes

### 2) Exp1: Quantization and noise (multiple images)

Systematic evaluation over multiple images:

Step 1 — prepare images
```powershell
# Copy your test images into the data/ folder
Copy-Item people2.png data/
# …or add more images
```

Step 2 — run the full experiment
```powershell
python experiments/exp_quantization_noise.py
```

Outputs:
- `results/quantization_noise_results.csv` — full dataset
- `results/plots/rate_distortion_quantization.png` — rate–distortion curves
- `results/plots/noise_robustness.png` — noise robustness curves
- `results/plots/degradation_heatmap.png` — semantic degradation heatmap
- `results/plots/3d_rate_noise_distortion.png` — 3D view

Scale:
- 6 quantization levels (2, 4, 6, 8, 12, 16 bits)
- 6 noise levels (σ = 0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
- 36 configs per image → 360 for 10 images
- Runtime: ~5–15 minutes (image count and GPU dependent)

### 3) Exp2: Uncertainty threshold τ scan

Study how the transmit decision threshold impacts performance:

```powershell
python experiments/exp_tau_scan.py
```

Outputs:
- `results/tau_scan_results.csv` — aggregated stats
- `results/plots/tau_tradeoff.png` — transmit rate vs semantic quality
- `results/plots/tau_transmission_rate.png` — transmit rate vs τ

Setup:
- Fixed quantization: `n_bits = 6` (best from Exp1)
- Fixed noise: `σ = 0.1` (medium channel)
- Scan: `τ ∈ [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]`
- Defaults to 50 images

Key finding: at `σ = 0.1`, `τ = 0.20` is optimal (≈4% transmit, ≈17% quality lift).

---

### 4) Exp3: Sigma × Tau joint optimization ⭐

Map optimal τ as a function of channel quality:

```powershell
python experiments/exp_sigma_tau_joint.py
```

Outputs:
- `results/sigma_tau_joint_results.csv` — full data
- `results/plots/sigma_tau_heatmap_quality.png` — quality heatmap
- `results/plots/sigma_tau_heatmap_transmit.png` — transmit-rate heatmap
- `results/plots/optimal_tau_vs_sigma.png` — optimal τ vs σ curve
- `results/plots/quality_vs_tau_by_sigma.png` — layered quality curves

Design:
- Scan noise: `σ ∈ [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]`
- Scan thresholds: `τ ∈ [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]`
- 6 × 7 = 42 combinations
- 30 images by default
- ≈1,260 evaluations, ≈40 minutes

Why it matters:
- Reveals how optimal τ shifts with channel quality
- Basis for adaptive transmit policies
- Complete performance atlas over the parameter space

Expected trend:
```
σ = 0.0 (clean channel)  → optimal τ ≈ 0.0 (transmit more)
σ = 0.3 (noisy channel)  → optimal τ ≈ 0.2 (transmit less)
```

---

### 5) Exp4: Image complexity robustness 🆕

Study how content complexity impacts semantic transmission quality:

```powershell
python experiments/exp_complexity_robustness.py
```

Outputs:
- `results/complexity_robustness_detail.csv` — per-image details
- `results/complexity_robustness_results.csv` — group-level aggregates
- `results/plots/complexity_distribution.png` — complexity distributions (4 subplots)
- `results/plots/complexity_robustness_bar.png` — robustness bar charts (4 subplots)
- `results/plots/complexity_vs_performance.png` — complexity vs performance (2 subplots)

Design:
- Fixed: `n_bits=6`, `σ=0.1`, `τ=0.05`
- 150 images under `data/all_images/`
- Complexity metrics:
  - Edge density (Canny)
  - Color variance (RGB std)
  - High-frequency energy ratio (FFT)
  - Combined score (weighted average)
- Auto grouping:
  - Low: < 33rd percentile
  - Medium: 33rd–67th percentile
  - High: > 67th percentile
- Runtime: ~40 minutes

Core findings:
```
Medium-complexity images are most robust! (the "blonde girl" region)

sim_rx:
  Medium: 0.846  (best ⭐)
  Low:    0.817  (-3.5%)
  High:   0.830  (-1.9%)

Semantic degradation:
  Medium: 0.072  (lowest ⭐)
  Low:    0.088  (+22%, worst)
  High:   0.081  (+13%)
```

Why it matters:
- First quantification of complexity vs semantic robustness
- Challenges the assumption “simpler content = more robust”
- Supports content-aware adaptive transmission strategies
- Guides capture policy and transmit prioritization

---

## 📊 Experiment Parameters

### Quantization bits (n_bits)

| n_bits | Levels | Compression | Typical use |
|--------|--------|-------------|-------------|
| 2      | 4      | Extreme     | Ultra-low bandwidth |
| 4      | 16     | High        | Edge devices |
| 6      | 64     | Medium      | Balanced default |
| 8      | 256    | Low         | Standard compression |
| 12     | 4096   | Very low    | High quality |
| 16     | 65536  | Near-lossless | Reference |

### Noise level (sigma)

| sigma  | Description  | Channel |
|--------|--------------|---------|
| 0.0    | No noise     | Ideal |
| 0.05   | Very light   | Excellent |
| 0.1    | Low          | Good |
| 0.15–0.2 | Medium     | Typical |
| 0.3+   | High         | Poor |

---

## 📄 Result Reports

See detailed analyses here:
- `EXP1`: Quantization & Noise — rate–distortion, 6-bit best: `../docs/EXP1_RESULTS.md`
- `EXP2`: Tau Scan — transmit vs quality, ≈17% lift: `../docs/EXP2_RESULTS.md`
- `EXP3`: Sigma×Tau Joint — universal τ* ≈ 0.2: `../docs/EXP3_RESULTS.md`
- `EXP4`: Complexity Robustness — content-aware strategy: `../docs/EXP4_RESULTS.md`

---

## 📈 Reading Results

### CSV fields

Generated CSVs include:

- `img_name`: image filename
- `n_bits`: quantization bits
- `sigma`: noise stddev
- `sim_local`: CLIP similarity of local reconstruction (no channel)
- `sim_rx`: CLIP similarity after channel
- `uncertainty`: 1 − sim_local
- `transmit`: whether transmit triggered (uncertainty-gated)
- `semantic_degradation`: sim_local − sim_rx

### Key metrics

1) `sim_rx` — semantic fidelity (higher is better)
   - > 0.95: excellent
   - 0.90–0.95: good
   - 0.80–0.90: acceptable
   - < 0.80: degraded

2) `semantic_degradation` — loss (lower is better)
   - < 0.01: near lossless
   - 0.01–0.05: mild
   - 0.05–0.10: moderate
   - > 0.10: severe

## 🔧 Customize Experiments

### Adjust parameter ranges

Edit `experiments/exp_quantization_noise.py`:

```python
# Modify these lists to change the sweep
N_BITS_LIST = [2, 4, 6, 8, 12, 16]
SIGMA_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
```

### Use more images

```powershell
# Pull diverse images from ImageNet/COCO/etc.
# Recommended quality: resolution > 256×256, PNG/JPG format
```

## 💡 Tips

### First run

1) Run `exp_quick_test.py` to validate environment
2) Check CSV schema in `results/`
3) Then run the full sweeps

### Image selection

For meaningful statistics, use:
- Diverse content: people, scenes, objects, text
- Varying complexity: simple backgrounds vs busy scenes
- At least 5–10 images

### Interpreting results

Focus on:
1) Quantization floor: which bits cause sharp drops?
2) Noise sensitivity: robustness across n_bits
3) Best operating point under bandwidth constraints

## 🐛 Troubleshooting

### Issue: Module not found

```
ModuleNotFoundError: No module named 'semcom_utils'
```

Fix:
- Run from the project root, or verify `sys.path`.

### Issue: CUDA out of memory

Fix:
```python
# At the top of the script
DEVICE = "cpu"  # force CPU
# Or reduce image count
```

### Issue: No plots generated

Fix: ensure plotting deps are installed
```powershell
pip install matplotlib seaborn
```

## 📚 What Next

After running, you can:
1) Analyze rate–distortion and pick operating points
2) Write concise experiment reports
3) Explore next steps (adaptive transmit, task metrics)
4) Try advanced quantization (vector/learned quantization)

## 📧 Need help?

Checklist:
1) Dependencies installed (`requirements.txt`)
2) GPU optional; CPU works but is slower
3) Images exist and have valid formats
