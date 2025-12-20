"""
Uncertainty-Aware and Noise-Resilient Task-Oriented Semantic Communication (TOSC) Demo
--------------------------------------------------------------------------------------

This script implements a simplified semantic communication loop combining:

1. A visual autoencoder (VAE) acting as the semantic codec that encodes and
   reconstructs images through a latent bottleneck. Channel noise and quantization are
   applied in this latent space to simulate transmission over a noisy link. 
   (its implemented on a basic simulation level, further improvements are being worked on)

2. A semantic evaluator (CLIP) a pretrained vision-language model that measures
   semantic similarity between the original image and its reconstructions, providing a
   quantitative notion of meaning preservation.

3. An **uncertainty-based transmission gate that computes an uncertainty score
   (u = 1 - similarity) and decides whether to transmit through the channel depending
   on a user-defined threshold (τ). If uncertainty is high, the system transmits and
   decodes after the simulated channel; if low, it skips transmission to save bandwidth.

"""

# ----------------------------------------------------------------------
# Import of dependencies
# ----------------------------------------------------------------------
import torch, numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
import open_clip
import os

# ----------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------
IMG_PATH = "people2.png" # Input image for evaluation
OUT_DIR = "out_demo" # Directory to store visual outputs
SIZE_VAE = 512 # Input resolution expected by the Stable Diffusion VAE
SIGMA = 0.10 # Standard deviation of Gaussian noise (AWGN) in latent channel
N_BITS = 6 # Uniform quantization bit depth (controls compression)
DROPOUT_P = 0.00 # Element-wise dropout probability on latent features
TAU = 0.02 # Uncertainty threshold for transmission decision
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if available
torch.manual_seed(0)

# Create output directory for saved images
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------

# 1 - Semantic encoder/decoder (visual autoencoder)
# Load the pretrained Stable Diffusion VAE (fine-tuned with MSE loss).
# This component encodes an image into a latent representation z and
# reconstructs it through a decoder, simulating a semantic codec.
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
sf  = getattr(vae.config, "scaling_factor", 1.0)  # Internal scaling factor used by SD VAEs

# 2 - Semantic evaluator (CLIP)
# Load OpenCLIP with ViT-B/32 backbone pretrained on OpenAI’s CLIP dataset.
# CLIP maps images into a shared semantic embedding space for similarity comparison.
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai",
    device=DEVICE
)
clip_model.eval()

# ----------------------------------------------------------------------
# Preprocessing pipelines
# ----------------------------------------------------------------------

# VAE preprocessing: resize and normalize input images to [-1, 1].
pre_vae = transforms.Compose([
    transforms.Resize((SIZE_VAE, SIZE_VAE), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

@torch.no_grad()
def vae_encode(x):
    """
    Encode an image tensor into latent space using the pretrained VAE.
    Returns the latent mean (mode) multiplied by the scaling factor.
    """
    posterior = vae.encode(x)
    z = posterior.latent_dist.mode() * sf
    return z


@torch.no_grad()
def vae_decode(z):
    """
    Decode a latent representation z back to image space.
    Output is clamped to [0,1] for visualization.
    """
    x_hat = vae.decode(z / sf).sample
    x_hat = (x_hat.clamp(-1,1) + 1) * 0.5
    return x_hat


def save_tensor_image(x, path):
    """
    Save a single image tensor (CHW, [0,1]) as a PNG file.
    Used to visualize baseline and channel reconstructions.
    """
    x = (x * 255).byte().cpu().squeeze(0).permute(1,2,0).numpy()
    Image.fromarray(x).save(path)


def channel(z, sigma=0.1, n_bits=6, p_drop=0.0):
    """
    Simulate a noisy communication channel in the latent space.

    - Adds additive white Gaussian noise (AWGN) with std=sigma.
    - Applies uniform quantization with n_bits precision.
    - Optionally applies random dropout to emulate packet loss.

    Returns the perturbed latent z_out.
    """
    z_out = z.clone()

    # Add Gaussian noise
    if sigma and sigma > 0:
        z_out = z_out + sigma * torch.randn_like(z_out)

    # Apply quantization
    if n_bits is not None:
        L = (2 ** int(n_bits)) - 1
        zc = torch.clamp(z_out, -1, 1)
        zq = torch.round((zc + 1) * 0.5 * L) / L
        z_out = zq * 2 - 1

    # Optional dropout
    if p_drop and p_drop > 0:
        mask = (torch.rand_like(z_out) > p_drop).float()
        z_out = z_out * mask

    return z_out


@torch.no_grad()
def clip_img_embed(pil_img):
    """
    Compute a CLIP embedding for a PIL image using OpenCLIP.
    Returns an L2-normalized embedding vector (unit norm).
    """
    x = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
    emb = clip_model.encode_image(x)
    return emb / emb.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_tensor_embed(x_tensor):
    """
    Convert a decoded image tensor back to a PIL image and obtain its CLIP embedding.
    Used to evaluate the semantic similarity of reconstructed outputs.
    """
    x = (x_tensor * 255).byte().cpu().permute(1,2,0).numpy()
    pil = Image.fromarray(x)
    return clip_img_embed(pil)


def cosine_sim(a, b):
    """
    Compute cosine similarity between two CLIP embeddings.
    This serves as the semantic similarity metric (0–1 range).
    """
    return (a @ b.t()).item()


# ----------------------------------------------------------------------
# Main semantic communication pipeline
# ----------------------------------------------------------------------

@torch.no_grad()
def run(img_path, tau=0.02, sigma=None, n_bits=None, p_drop=None, save_images=True):
    """
    Executes the complete uncertainty-aware semantic communication loop:
    1. Encode and locally reconstruct the image via VAE.
    2. Measure semantic similarity (CLIP) between original and local reconstruction.
    3. Compute uncertainty (u = 1 - similarity) and decide whether to transmit.
    4. If transmit=True, apply channel noise/quantization and re-decode.
    5. Measure semantic robustness after channel.
    6. Optionally save all visual results.

    Args:
        img_path: Path to input image
        tau: Uncertainty threshold for transmission decision
        sigma: Channel noise std (uses global SIGMA if None)
        n_bits: Quantization bits (uses global N_BITS if None)
        p_drop: Dropout probability (uses global DROPOUT_P if None)
        save_images: Whether to save output images

    Returns:
        dict with keys:
            'transmit'     (bool): whether channel transmission occurred
            'sim_local'    (float): similarity before channel
            'sim_rx'       (float): similarity after channel
            'uncertainty'  (float): uncertainty score
            'n_bits'       (int/None): quantization bits used
            'sigma'        (float): noise std used
    """
    # Use provided parameters or fall back to globals
    sigma = SIGMA if sigma is None else sigma
    n_bits = N_BITS if n_bits is None else n_bits
    p_drop = DROPOUT_P if p_drop is None else p_drop
    
    # --- Load and preprocess input image ---
    pil = Image.open(img_path).convert("RGB")
    x_in = pre_vae(pil).unsqueeze(0).to(DEVICE)

    # --- Encode and decode locally (no channel) ---
    z0 = vae_encode(x_in)
    x_hat0 = vae_decode(z0)
    if save_images:
        save_tensor_image(x_hat0, os.path.join(OUT_DIR, "recon_baseline.png"))

    # --- Compute CLIP semantic similarity (original vs local reconstruction) ---
    emb_ref = clip_img_embed(pil)
    emb_loc = clip_tensor_embed(x_hat0[0])
    sim_loc = cosine_sim(emb_ref, emb_loc)
    u = 1.0 - sim_loc  # Uncertainty estimate

    # --- Transmission gate decision ---
    transmit = u > tau

    # Prepare output record
    result = {
        "transmit": bool(transmit),
        "sim_local": float(sim_loc),
        "uncertainty": float(u),
        "n_bits": n_bits,
        "sigma": float(sigma) if sigma is not None else None,
    }

    # --- If uncertain, simulate channel transmission ---
    if transmit:
        # Apply latent-space corruption
        z_tx = channel(z0, sigma=sigma, n_bits=n_bits, p_drop=p_drop)
        # Decode transmitted latent
        x_hatR = vae_decode(z_tx)
        if save_images:
            save_tensor_image(x_hatR, os.path.join(OUT_DIR, "recon_rx.png"))

        # Measure post-channel semantic similarity
        emb_rx = clip_tensor_embed(x_hatR[0])
        sim_rx = cosine_sim(emb_ref, emb_rx)
        result["sim_rx"] = float(sim_rx)
    else:
        # If not transmitted, post-channel similarity equals local one
        result["sim_rx"] = float(sim_loc)

    # --- Save resized original image for visualization consistency ---
    if save_images:
        pil.resize((SIZE_VAE, SIZE_VAE), Image.LANCZOS).save(os.path.join(OUT_DIR, "input_resized.png"))

    return result


# ----------------------------------------------------------------------
# Run experiment
# ----------------------------------------------------------------------

out = run(IMG_PATH, tau=TAU)
print(out)
print(f"Images saved in: {OUT_DIR}")

"""
Example Output:
---------------
{'transmit': False, 'sim_local': 0.9783, 'sim_rx': 0.9783, 'uncertainty': 0.0216}
Images saved in: out_demo/

Interpretation:
---------------
- sim_local:   CLIP semantic similarity before channel (high → low uncertainty)
- uncertainty: 1 - sim_local
- transmit:    True if uncertainty > τ
- sim_rx:      CLIP similarity after channel (if transmitted)
"""
