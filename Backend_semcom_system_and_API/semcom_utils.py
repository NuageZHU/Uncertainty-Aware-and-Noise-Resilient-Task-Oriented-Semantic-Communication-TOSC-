"""
Utility functions for semantic communication experiments.

This module provides reusable functions for:
- VAE encoding/decoding
- Channel simulation (noise, quantization, dropout)
- CLIP semantic similarity evaluation
- Image I/O operations
"""

import torch
import os
from PIL import Image


# ----------------------------------------------------------------------
# VAE Encoding/Decoding
# ----------------------------------------------------------------------

@torch.no_grad()
def vae_encode(vae, x, scaling_factor):
    """
    Encode an image tensor into latent space using the pretrained VAE.
    
    Args:
        vae: The VAE model
        x: Input image tensor (NCHW, [-1, 1])
        scaling_factor: VAE scaling factor
    
    Returns:
        z: Latent representation
    """
    posterior = vae.encode(x)
    z = posterior.latent_dist.mode() * scaling_factor
    return z


@torch.no_grad()
def vae_decode(vae, z, scaling_factor):
    """
    Decode a latent representation z back to image space.
    
    Args:
        vae: The VAE model
        z: Latent representation
        scaling_factor: VAE scaling factor
    
    Returns:
        x_hat: Reconstructed image tensor (NCHW, [0,1])
    """
    x_hat = vae.decode(z / scaling_factor).sample
    x_hat = (x_hat.clamp(-1, 1) + 1) * 0.5
    return x_hat


# ----------------------------------------------------------------------
# Channel Simulation
# ----------------------------------------------------------------------

def channel(z, sigma=0.1, n_bits=6, p_drop=0.0):
    """
    Simulate a noisy communication channel in the latent space.

    Applies three types of degradation:
    1. Additive White Gaussian Noise (AWGN)
    2. Uniform quantization
    3. Random dropout (packet loss simulation)

    Args:
        z: Input latent tensor
        sigma: Standard deviation of AWGN noise (set to 0 or None to skip)
        n_bits: Quantization bit depth (set to None to skip quantization)
        p_drop: Dropout probability (set to 0 or None to skip)

    Returns:
        z_out: Perturbed latent tensor
    """
    z_out = z.clone()

    # Add Gaussian noise
    if sigma and sigma > 0:
        z_out = z_out + sigma * torch.randn_like(z_out)

    # Apply uniform quantization
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


# ----------------------------------------------------------------------
# CLIP Semantic Similarity
# ----------------------------------------------------------------------

@torch.no_grad()
def clip_img_embed(clip_model, clip_preprocess, pil_img, device):
    """
    Compute a CLIP embedding for a PIL image.
    
    Args:
        clip_model: CLIP model
        clip_preprocess: CLIP preprocessing transform
        pil_img: PIL Image
        device: torch device
    
    Returns:
        emb: L2-normalized CLIP embedding
    """
    x = clip_preprocess(pil_img).unsqueeze(0).to(device)
    emb = clip_model.encode_image(x)
    return emb / emb.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_tensor_embed(clip_model, clip_preprocess, x_tensor, device):
    """
    Convert a decoded image tensor to CLIP embedding.
    
    Args:
        clip_model: CLIP model
        clip_preprocess: CLIP preprocessing transform
        x_tensor: Image tensor (CHW, [0,1])
        device: torch device
    
    Returns:
        emb: L2-normalized CLIP embedding
    """
    x = (x_tensor * 255).byte().cpu().permute(1, 2, 0).numpy()
    pil = Image.fromarray(x)
    return clip_img_embed(clip_model, clip_preprocess, pil, device)


def cosine_sim(a, b):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        a, b: Embedding tensors
    
    Returns:
        similarity: Cosine similarity (float, range [0, 1] for normalized embeddings)
    """
    return (a @ b.t()).item()


# ----------------------------------------------------------------------
# Image I/O
# ----------------------------------------------------------------------

def save_tensor_image(x, path):
    """
    Save a single image tensor as a PNG file.
    
    Args:
        x: Image tensor (NCHW or CHW, [0,1])
        path: Output file path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    x = (x * 255).byte().cpu().squeeze(0).permute(1, 2, 0).numpy()
    Image.fromarray(x).save(path)
