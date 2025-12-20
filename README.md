# 🧠 Uncertainty-Aware and Noise-Resilient Task-Oriented Semantic Communication (TOSC)

This repository contains a **minimal simulation framework** for studying how *uncertainty-aware transmission* and *semantic robustness* can be modeled in **AI-native communication systems**.  
The implementation combines pretrained **Variational Autoencoders (VAEs)** for image encoding and **Vision-Language Models (CLIP)** for semantic evaluation, allowing the simulation of noisy semantic channels and adaptive transmission decisions.

**🆕 Now includes an interactive web dashboard** for real-time visualization and experimentation!

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

---

## 📁 Project Structure

```
Uncertainty-Aware-TOSC/
│
├── Backend_semcom_system_and_API/     # 🐍 Python Backend (FastAPI + ML Models)
│   ├── app/
│   │   └── main.py                    # FastAPI server with endpoints
│   ├── semcom_main.py                 # Main demo script (CLI)
│   ├── semcom_utils.py                # Utility functions (VAE, CLIP, channel)
│   ├── experiments/                   # Experimental scripts
│   │   ├── exp_quantization_noise.py  # Exp1: Quantization & noise
│   │   ├── exp_tau_scan.py            # Exp2: Tau threshold scan
│   │   ├── exp_sigma_tau_joint.py     # Exp3: Sigma×Tau joint
│   │   ├── exp_complexity_robustness.py # Exp4: Image complexity
│   │   └── README.md                  # Experiments guide
│   ├── docs/                          # Documentation
│   ├── data/                          # Test images (150 images)
│   ├── results/                       # CSV results & plots
│   └── requirements.txt               # Python dependencies
│
├── semcom-dashboard-app/              # ⚛️ React Frontend (Vite + TypeScript)
│   ├── src/
│   │   ├── components/                # UI components
│   │   │   ├── semantic-comm-dashboard.tsx  # Main dashboard
│   │   │   └── dashboard/             # Dashboard sub-components
│   │   ├── lib/
│   │   │   └── api.ts                 # API client
│   │   └── hooks/
│   │       └── use-api.ts             # React hooks for API
│   ├── package.json                   # Node dependencies
│   └── vite.config.ts                 # Vite configuration
│
└── README.md                          # This file
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (for backend)
- **Node.js 18+** (for frontend)
- **GPU recommended** (CUDA) for faster inference, but CPU works too

---

### 1️⃣ Backend Setup (FastAPI + ML Models)

```bash
# Navigate to backend folder
cd Backend_semcom_system_and_API

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) For faster model loading:
pip install accelerate
```

#### Run the API Server

```bash
cd Backend_semcom_system_and_API
uvicorn app.main:app --reload --port 8000
```

The API will be available at: **http://127.0.0.1:8000**

📚 API Documentation: **http://127.0.0.1:8000/docs**

---

### 2️⃣ Frontend Setup (React Dashboard)

```bash
# Navigate to frontend folder
cd semcom-dashboard-app

# Install dependencies
npm install

# Start development server
npm run dev
```

The dashboard will be available at: **http://localhost:5173**

---

### 3️⃣ Using Both Together

1. **Start the backend** (Terminal 1):
   ```bash
   cd Backend_semcom_system_and_API
   uvicorn app.main:app --reload --port 8000
   ```

2. **Start the frontend** (Terminal 2):
   ```bash
   cd semcom-dashboard-app
   npm run dev
   ```

3. **Open the dashboard** in your browser: http://localhost:5173

---

## 🖥️ What Each Component Does

### Backend (`Backend_semcom_system_and_API/`)

The Python backend provides:

| Endpoint | Description |
|----------|-------------|
| `POST /api/upload-run` | **Upload your own image** and run it through the semantic pipeline. Returns original, local reconstruction, and channel reconstruction images with metrics. |
| `GET /api/samples` | List available test images from the dataset |
| `POST /api/run` | Run pipeline on a dataset image |
| `GET /api/experiments/tau-scan` | Pre-computed tau threshold experiment results |
| `GET /api/experiments/quantization-noise` | Pre-computed quantization/noise results |
| `GET /api/experiments/complexity` | Pre-computed image complexity results |

**Key Features:**
- 🧠 **Stable Diffusion VAE** for semantic encoding/decoding
- 🔍 **CLIP ViT-B/32** for semantic similarity evaluation
- 📡 **Channel simulation** with noise, quantization, and dropout
- 🎯 **Uncertainty-based transmission gating**

### Frontend (`semcom-dashboard-app/`)

An interactive React dashboard that provides:

- 📤 **Image Upload**: Upload your own images to test the pipeline
- 🎛️ **Parameter Controls**: Adjust `n_bits`, `sigma`, `tau` in real-time
- 📊 **Visualization**: See original vs reconstructed images side-by-side
- 📈 **Experiment Results**: Interactive charts from pre-computed experiments
- 📉 **Metrics Display**: Semantic similarity, uncertainty, transmission decisions

---

## 🧪 Running Experiments (CLI)

For batch experiments without the web interface:

```bash
cd Backend_semcom_system_and_API

# Quick test (few configurations)
python experiments/exp_quick_test.py

# Full experiments
python experiments/exp_quantization_noise.py    # Exp1
python experiments/exp_tau_scan.py              # Exp2
python experiments/exp_sigma_tau_joint.py       # Exp3 (recommended)
python experiments/exp_complexity_robustness.py # Exp4
```

Results are saved to `results/` as CSV files and plots.

---

## 📚 Documentation

- **[Usage Guide](Backend_semcom_system_and_API/docs/USAGE_GUIDE.md)** - Detailed installation & usage
- **[Experiments Guide](Backend_semcom_system_and_API/experiments/README.md)** - Experimental scripts details

### 📊 Experimental Results
- **[Experiment 1: Quantization & Noise](Backend_semcom_system_and_API/docs/EXP1_RESULTS.md)** - Rate-distortion analysis
- **[Experiment 2: Tau Threshold Scan](Backend_semcom_system_and_API/docs/EXP2_RESULTS.md)** - Uncertainty threshold optimization
- **[Experiment 3: Sigma×Tau Joint](Backend_semcom_system_and_API/docs/EXP3_RESULTS.md)** - Channel-adaptive strategy
- **[Experiment 4: Image Complexity](Backend_semcom_system_and_API/docs/EXP4_RESULTS.md)** - Content-aware robustness

---

## 🧩 Features

- ✅ Pretrained **Stable Diffusion VAE** for latent-space encoding  
- ✅ Pretrained **OpenCLIP (ViT-B/32)** for semantic similarity evaluation  
- ✅ Configurable **noise, quantization, and dropout** parameters  
- ✅ Adaptive **uncertainty-based transmission gate**
- ✅ **REST API** for integration with other applications
- ✅ **Interactive web dashboard** for visualization and experimentation
- ✅ **Image upload** to test your own images
- ✅ Modular, research-ready structure

---

## 🧮 Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_bits` | Quantization bit-depth (2-16) | `6` |
| `sigma` | Standard deviation of AWGN noise | `0.10` |
| `tau` | Transmission threshold for uncertainty gating | `0.05` |
| `dropout_p` | Dropout probability on latent vector | `0.00` |

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
|--------|---------|
| `sim_local` | CLIP cosine similarity before transmission |
| `sim_rx` | Semantic similarity after noisy transmission |
| `uncertainty` | `1 - sim_local`; low means high confidence |
| `transmit` | `True` if uncertainty > τ (channel used) |
| `effective_sim` | Final similarity (sim_rx if transmitted, else sim_local) |
| `semantic_degradation` | `sim_local - sim_rx`; quality loss from channel |

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
