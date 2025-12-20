# app/main.py
from pathlib import Path
from typing import List, Optional
import uuid

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

import torch
from torchvision import transforms
from diffusers import AutoencoderKL
import open_clip
from PIL import Image
import pandas as pd

# Importa tus utilidades
from semcom_utils import (
    vae_encode, vae_decode, channel,
    clip_img_embed, clip_tensor_embed, cosine_sim
)

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "all_images"
RESULTS_DIR = PROJECT_ROOT / "results"
WEB_RUNS_DIR = RESULTS_DIR / "web_runs"

WEB_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------
app = FastAPI(title="Semantic Communication Dashboard API")

# CORS - permitir requests desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Servir estáticos (plots, imágenes de salida, etc.)
app.mount("/static", StaticFiles(directory=RESULTS_DIR), name="static")

# --------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------
class Sample(BaseModel):
    name: str
    path: str  # URL relativa para mostrar en frontend

class RunRequest(BaseModel):
    image_name: str
    n_bits: int
    sigma: float
    tau: float
    dropout_p: float = 0.0

class RunResponse(BaseModel):
    transmitted: bool
    uncertainty: float
    sim_local: float
    sim_rx: float
    effective_sim: float
    original_url: str
    local_recon_url: str
    channel_recon_url: Optional[str]


class UploadRunResponse(BaseModel):
    """Respuesta para el endpoint de upload de imagen."""
    run_id: str
    transmitted: bool
    uncertainty: float
    sim_local: float
    sim_rx: float
    effective_sim: float
    semantic_degradation: float
    # URLs de las imágenes
    original_url: str
    local_recon_url: str
    channel_recon_url: str
    # Parámetros usados
    n_bits: int
    sigma: float
    tau: float


# --------------------------------------------------------------------
# Utilidades internas
# --------------------------------------------------------------------
def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """
    Convierte un tensor VAE en [0, 1] con shape (C, H, W) a PIL.Image RGB.
    Nota: vae_decode ya devuelve valores en [0, 1], no en [-1, 1].
    """
    x = x.detach().clone()
    x = x.clamp(0, 1)  # Asegurar rango [0, 1]
    x = (x * 255).byte()
    if x.dim() == 3:
        x = x.permute(1, 2, 0)  # (H, W, C)
    arr = x.cpu().numpy()
    return Image.fromarray(arr)


def list_image_samples() -> List[Path]:
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        image_paths.extend(DATA_DIR.glob(ext))
    # Evitar duplicados (case-insensitive)
    seen = set()
    unique_paths = []
    for p in image_paths:
        key = p.name.lower()
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)
    return sorted(unique_paths)


# --------------------------------------------------------------------
# Carga de modelos (una sola vez)
# --------------------------------------------------------------------
@app.on_event("startup")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] Using device: {device}")

    # VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    sf = getattr(vae.config, "scaling_factor", 1.0)

    # CLIP
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="openai",
        device=device,
    )
    clip_model.eval()

    # Preprocesado VAE
    size_vae = 512
    pre_vae = transforms.Compose([
        transforms.Resize((size_vae, size_vae),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Guardar en app.state
    app.state.device = device
    app.state.vae = vae
    app.state.sf = sf
    app.state.clip_model = clip_model
    app.state.clip_preprocess = clip_preprocess
    app.state.pre_vae = pre_vae

    # Cachear lista de imágenes
    app.state.samples = list_image_samples()
    print(f"[startup] Found {len(app.state.samples)} images in {DATA_DIR}")


# --------------------------------------------------------------------
# Endpoints "live"
# --------------------------------------------------------------------
@app.get("/api/samples", response_model=List[Sample])
def get_samples():
    samples: List[Path] = app.state.samples
    result = []
    for p in samples:
        # Si quieres poder previsualizarlas en la web:
        # podrías copiar estas imágenes a RESULTS_DIR/static_images
        result.append(Sample(
            name=p.name,
            path=f"/static/../data/all_images/{p.name}"  # o solo el nombre si las sirves por otro lado
        ))
    return result


@app.post("/api/run", response_model=RunResponse)
def run_single(req: RunRequest):
    img_path = DATA_DIR / req.image_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image {req.image_name} not found")

    device = app.state.device
    vae = app.state.vae
    sf = app.state.sf
    clip_model = app.state.clip_model
    clip_preprocess = app.state.clip_preprocess
    pre_vae = app.state.pre_vae

    # 1) Cargar imagen
    try:
        pil = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading image: {e}")

    # 2) Preparar para VAE
    x_in = pre_vae(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # 3) Encode + decode local
        z0 = vae_encode(vae, x_in, sf)
        x_hat_local = vae_decode(vae, z0, sf)

        # 4) CLIP embeddings
        emb_ref = clip_img_embed(clip_model, clip_preprocess, pil, device)
        emb_loc = clip_tensor_embed(clip_model, clip_preprocess, x_hat_local[0], device)
        sim_local = float(cosine_sim(emb_ref, emb_loc))

        # 5) Uncertainty + decisión de transmisión
        uncertainty = 1.0 - sim_local
        transmit = bool(uncertainty > req.tau)

        # 6) Canal (siempre lo aplicamos para sim_rx; el gate decide qué usamos)
        z_tx = channel(z0, sigma=req.sigma, n_bits=req.n_bits, p_drop=req.dropout_p)
        x_hat_rx = vae_decode(vae, z_tx, sf)
        emb_rx = clip_tensor_embed(clip_model, clip_preprocess, x_hat_rx[0], device)
        sim_rx = float(cosine_sim(emb_ref, emb_rx))

    # 7) Effective similarity
    effective_sim = sim_rx if transmit else sim_local

    # 8) Guardar imágenes para que el frontend las vea
    run_id = uuid.uuid4().hex[:8]
    run_dir = WEB_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Original
    pil.save(run_dir / "original.png")

    # Reconstrucciones
    pil_local = tensor_to_pil(x_hat_local[0])
    pil_local.save(run_dir / "local.png")

    pil_channel = tensor_to_pil(x_hat_rx[0])
    pil_channel.save(run_dir / "channel.png")

    return RunResponse(
        transmitted=transmit,
        uncertainty=uncertainty,
        sim_local=sim_local,
        sim_rx=sim_rx,
        effective_sim=effective_sim,
        original_url=f"/static/web_runs/{run_id}/original.png",
        local_recon_url=f"/static/web_runs/{run_id}/local.png",
        channel_recon_url=f"/static/web_runs/{run_id}/channel.png",
    )


# --------------------------------------------------------------------
# Endpoints "experiments" (leyendo CSV)
# --------------------------------------------------------------------
@app.get("/api/experiments/quantization-noise")
def get_quantization_noise():
    """
    Devuelve métricas promediadas de quantization_noise_results.csv
    para dibujar rate–semantic distortion y noise robustness.
    """
    csv_path = RESULTS_DIR / "quantization_noise_results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="quantization_noise_results.csv not found")

    df = pd.read_csv(csv_path)

    grouped = (
        df.groupby(["n_bits", "sigma"])
          .agg(
              sim_rx_mean=("sim_rx", "mean"),
              degradation_mean=("semantic_degradation", "mean"),
          )
          .reset_index()
    )
    return grouped.to_dict(orient="records")


@app.get("/api/experiments/tau-scan")
def get_tau_scan():
    """
    Lee tau_scan_results.csv y devuelve los arrays para la gráfica
    transmisión vs calidad.
    """
    csv_path = RESULTS_DIR / "tau_scan_results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="tau_scan_results.csv not found")

    df = pd.read_csv(csv_path)
    return {
        "tau": df["tau"].tolist(),
        "transmit_rate": df["transmit_rate"].tolist(),
        "mean_effective_sim": df["mean_effective_sim"].tolist(),
        "mean_sim_rx": df["mean_sim_rx"].tolist(),
        "mean_sim_local": df["mean_sim_local"].tolist(),
    }


@app.get("/api/experiments/complexity")
def get_complexity():
    """
    Lee complexity_robustness_results.csv y expone las métricas por grupo
    (low / medium / high).
    """
    csv_path = RESULTS_DIR / "complexity_robustness_results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="complexity_robustness_results.csv not found")

    df = pd.read_csv(csv_path)
    # Asegurar orden (low, medium, high) si existe la columna
    order = ["low", "medium", "high"]
    if "complexity_group" in df.columns:
        df["complexity_group"] = pd.Categorical(df["complexity_group"], order)
        df = df.sort_values("complexity_group")
    return df.to_dict(orient="records")


# --------------------------------------------------------------------
# Endpoint: Upload imagen y procesar
# --------------------------------------------------------------------
@app.post("/api/upload-run", response_model=UploadRunResponse)
async def upload_and_run(
    image: UploadFile = File(..., description="Imagen a procesar (PNG, JPG, etc.)"),
    n_bits: int = Form(default=6, description="Bits de cuantización (2-16)"),
    sigma: float = Form(default=0.1, description="Desviación estándar del ruido de canal"),
    tau: float = Form(default=0.05, description="Umbral de incertidumbre para transmitir"),
    dropout_p: float = Form(default=0.0, description="Probabilidad de dropout")
):
    """
    🚀 Sube tu propia imagen y obtén el análisis completo del pipeline semántico.
    
    Devuelve:
    - Imagen original
    - Reconstrucción local (VAE encode → decode, sin canal)
    - Reconstrucción después del canal (con cuantización + ruido)
    - Métricas de similaridad semántica (CLIP)
    - Decisión de transmisión basada en incertidumbre
    """
    # Validar tipo de archivo
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail=f"Archivo no válido. Se esperaba imagen, se recibió: {image.content_type}"
        )
    
    # Validar parámetros
    if not (1 <= n_bits <= 16):
        raise HTTPException(status_code=400, detail="n_bits debe estar entre 1 y 16")
    if sigma < 0:
        raise HTTPException(status_code=400, detail="sigma debe ser >= 0")
    if not (0 <= tau <= 1):
        raise HTTPException(status_code=400, detail="tau debe estar entre 0 y 1")
    
    # Leer imagen subida
    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer imagen: {e}")
    
    # Obtener modelos del estado de la app
    device = app.state.device
    vae = app.state.vae
    sf = app.state.sf
    clip_model = app.state.clip_model
    clip_preprocess = app.state.clip_preprocess
    pre_vae = app.state.pre_vae
    
    # Preparar para VAE
    x_in = pre_vae(pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 1) Encode + decode local (sin canal)
        z0 = vae_encode(vae, x_in, sf)
        x_hat_local = vae_decode(vae, z0, sf)
        
        # 2) CLIP embeddings
        emb_ref = clip_img_embed(clip_model, clip_preprocess, pil, device)
        emb_loc = clip_tensor_embed(clip_model, clip_preprocess, x_hat_local[0], device)
        sim_local = float(cosine_sim(emb_ref, emb_loc))
        
        # 3) Uncertainty + decisión de transmisión
        uncertainty = 1.0 - sim_local
        transmit = bool(uncertainty > tau)
        
        # 4) Aplicar canal (cuantización + ruido + dropout)
        z_tx = channel(z0, sigma=sigma, n_bits=n_bits, p_drop=dropout_p)
        x_hat_rx = vae_decode(vae, z_tx, sf)
        emb_rx = clip_tensor_embed(clip_model, clip_preprocess, x_hat_rx[0], device)
        sim_rx = float(cosine_sim(emb_ref, emb_rx))
    
    # Métricas derivadas
    effective_sim = sim_rx if transmit else sim_local
    semantic_degradation = sim_local - sim_rx
    
    # Guardar imágenes para visualización
    run_id = uuid.uuid4().hex[:8]
    run_dir = WEB_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar original (redimensionada a 512x512 como se procesa)
    pil_resized = pil.resize((512, 512), Image.LANCZOS)
    pil_resized.save(run_dir / "original.png")
    
    # Guardar reconstrucción local
    pil_local = tensor_to_pil(x_hat_local[0])
    pil_local.save(run_dir / "local_recon.png")
    
    # Guardar reconstrucción después del canal
    pil_channel = tensor_to_pil(x_hat_rx[0])
    pil_channel.save(run_dir / "channel_recon.png")
    
    return UploadRunResponse(
        run_id=run_id,
        transmitted=transmit,
        uncertainty=round(uncertainty, 4),
        sim_local=round(sim_local, 4),
        sim_rx=round(sim_rx, 4),
        effective_sim=round(effective_sim, 4),
        semantic_degradation=round(semantic_degradation, 4),
        original_url=f"/static/web_runs/{run_id}/original.png",
        local_recon_url=f"/static/web_runs/{run_id}/local_recon.png",
        channel_recon_url=f"/static/web_runs/{run_id}/channel_recon.png",
        n_bits=n_bits,
        sigma=sigma,
        tau=tau
    )
