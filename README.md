# 🧠 Uncertainty-Aware and Noise-Resilient Task-Oriented Semantic Communication (TOSC)

This repository contains a **minimal simulation framework** for studying how *uncertainty-aware transmission* and *semantic robustness* can be modeled in **AI-native communication systems**.  
The implementation combines pretrained **Variational Autoencoders (VAEs)** for image encoding and **Vision-Language Models (CLIP)** for semantic evaluation, allowing the simulation of noisy semantic channels and adaptive transmission decisions.

---

## 📖 Overview

The pipeline demonstrates a simplified version of the semantic communication loop proposed in recent 6G AI-RAN research:

1. **Semantic Encoding (VAE):**  
   Images are encoded into latent vectors `z`, which serve as compressed semantic representations.

2. **Channel Simulation:**  
   Additive White Gaussian Noise (AWGN), quantization, and dropout are applied to `z` to emulate physical-layer degradation.

3. **Semantic Evaluation (CLIP):**  
   CLIP embeddings quantify the similarity between the original image and reconstructed outputs, measuring *semantic fidelity* rather than pixel accuracy.

4. **Uncertainty-Aware Transmission Gate:**  
   A confidence-based rule decides whether to transmit through the channel.  
   If the uncertainty `u = 1 − similarity` exceeds a threshold `τ`, transmission is triggered; otherwise, it is skipped to save bandwidth.

The project follows the structure below:

```
semantic-channel-modeling/
│
├── semcom_main.py          # Main demo script
├── semcom_utils.py         # Utility functions (VAE, CLIP, channel, etc.)
├── experiments/            # Experimental scripts
│   ├── exp_quantization_noise.py      # Exp1: Quantization & noise
│   ├── exp_tau_scan.py                # Exp2: Tau threshold scan
│   ├── exp_sigma_tau_joint.py         # Exp3: Sigma×Tau joint (recommended)
│   ├── exp_complexity_robustness.py   # Exp4: Image complexity robustness
│   ├── exp_quick_test.py              # Quick test script
│   └── README.md                      # Experiments guide
├── docs/                   # Documentation
│   ├── USAGE_GUIDE.md             # Installation & usage guide
│   ├── EXP1_RESULTS.md            # Experiment 1 results
│   ├── EXP2_RESULTS.md            # Experiment 2 results
│   ├── EXP3_RESULTS.md            # Experiment 3 results
│   └── EXP4_RESULTS.md            # Experiment 4 results
├── data/                   # Input images directory
│   └── all_images/        # Test image dataset (150 images)
├── results/                # Experimental results
│   ├── *.csv              # Data tables
│   └── plots/             # Visualization plots
├── out_demo/               # Demo output images
└── requirements.txt        # Python dependencies
```

---

## 📚 Documentation

- **[Usage Guide](docs/USAGE_GUIDE.md)** - Installation, quick start, troubleshooting
- **[Experiments Guide](experiments/README.md)** - Experimental scripts details

### 📊 Experimental Results
- **[Experiment 1: Quantization & Noise](docs/EXP1_RESULTS.md)** - Rate-distortion analysis, 6-bit optimal finding
- **[Experiment 2: Tau Threshold Scan](docs/EXP2_RESULTS.md)** - Uncertainty threshold optimization, 17% quality gain
- **[Experiment 3: Sigma×Tau Joint](docs/EXP3_RESULTS.md)** - Channel-adaptive transmission strategy (recommended)
- **[Experiment 4: Image Complexity Robustness](docs/EXP4_RESULTS.md)** - Content-aware robustness analysis, "Goldilocks Zone" discovery

---

## 🧩 Features

- ✅ Pretrained **Stable Diffusion VAE** for latent-space encoding  
- ✅ Pretrained **OpenCLIP (ViT-B/32)** for semantic similarity evaluation  
- ✅ Configurable **noise, quantization, and dropout** parameters  
- ✅ Adaptive **uncertainty-based transmission gate**  
- ✅ Modular, research-ready structure (easy to extend for BLIP, multi-image, or text-conditioned setups)

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<yourusername>/semantic-channel-modeling.git
cd semantic-channel-modeling
```

### 2. Create and activate a virtual environment
**Windows (PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ Note: for faster model loading, it is recommended to install:
> ```bash
> pip install accelerate hf_xet
> ```
> (optional, improves performance when downloading Hugging Face weights)

---

## 🧪 Running the Simulation

To execute the experiment with default settings:

```bash
python semcom_main.py
```

Expected console output:
```
{'transmit': False, 'sim_local': 0.9783, 'sim_rx': 0.9783, 'uncertainty': 0.0216}
Images saved in: out_demo/
```

### Output Files (in `out_demo/`)
| File | Description |
|------|--------------|
| `input_resized.png` | Original resized input image |
| `recon_baseline.png` | Local VAE reconstruction (no channel) |
| `recon_rx.png` | Reconstruction after channel corruption *(only if transmitted)* |

---

## 🧮 Key Parameters

| Parameter | Description | Default |
|------------|--------------|----------|
| `SIGMA` | Standard deviation of AWGN noise | `0.10` |
| `N_BITS` | Quantization bit-depth | `6` |
| `DROPOUT_P` | Dropout probability on latent vector | `0.00` |
| `TAU` | Transmission threshold for uncertainty gating | `0.02` |
| `SIZE_VAE` | Input image resolution for VAE | `512 × 512` |

---

## 🧠 Conceptual Diagram

```
           ┌───────────────────────────┐
           │        Input Image        │
           └────────────┬──────────────┘
                        │
                        ▼
              [ Semantic Encoder (VAE) ]
                        │
                        ▼
               Latent Representation z
                        │
             ┌──────────┴──────────┐
             │                     │
     Low Uncertainty         High Uncertainty
    (skip transmission)        (transmit z→channel)
             │                     │
             ▼                     ▼
   Local reconstruction     Channel corruption + decode
             │                     │
             └──────────┬──────────┘
                        ▼
              [ Semantic Evaluator (CLIP) ]
                        │
                        ▼
           Semantic Similarity & Uncertainty Metric
```

---

## 📊 Example Interpretation

| Metric | Meaning |
|---------|----------|
| `sim_local` | CLIP cosine similarity before transmission |
| `uncertainty` | `1 - sim_local`; low means high confidence |
| `transmit` | `True` if uncertainty > τ (channel used) |
| `sim_rx` | Semantic similarity after noisy transmission |

---

## 📚 References

- Gündüz, D., Quek, T. Q. S., Strinati, E. C., & Kim, S.-L. (2022). *Beyond transmitting bits: Context, semantics, and task-oriented communications.*  
  [arXiv:2207.09353](https://discovery.ucl.ac.uk/10162483/1/Semantic_JSAC_SI_submitted.pdf)

- Oh, S., Kim, J., Park, J., Ko, S.-W., Quek, T. Q. S., & Kim, S.-L. (2025). *Uncertainty-aware hybrid inference with on-device small and remote large language models.*  
  IEEE ICMLCN. [link](https://ieeexplore.ieee.org/abstract/document/10508293)

- Liu, X., Zhang, Y., Zhou, F., & Zhang, H. (2022). *Task-oriented image semantic communication based on rate-distortion theory.*  
  [arXiv:2201.10929](https://arxiv.org/abs/2201.10929)

- Strinati, E. C., et al. (2024). *Goal-oriented and semantic communication in 6G AI-native networks: The 6G-GOALS approach.*  
  [arXiv:2402.07573](https://arxiv.org/abs/2402.07573)

---

## 🧩 License

This work is provided for **academic and research purposes only**.  
