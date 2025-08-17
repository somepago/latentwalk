"""
VAE utilities for encoding and decoding images using DC-AE VAE.
"""

import torch
from typing import Optional

# VAE imports
try:
    from diffusers import AutoencoderDC
    VAE_AVAILABLE = True
except ImportError:
    print("Warning: diffusers not installed. VAE encoding/decoding will not be available.")
    print("Install with: pip install diffusers")
    VAE_AVAILABLE = False

# Text encoder imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5EncoderModel
    TEXT_ENCODER_AVAILABLE = True
except ImportError:
    print("Warning: transformers not installed. Text encoding will not be available.")
    print("Install with: pip install transformers")
    TEXT_ENCODER_AVAILABLE = False


def load_vae(device="cuda", dtype=torch.float16):
    """
    Load the DC-AE VAE encoder/decoder.
    """
    if not VAE_AVAILABLE:
        return None
    
    print("Loading DC-AE VAE...")
    vae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        torch_dtype=dtype
    ).to(device)
    vae.eval()
    print("VAE loaded successfully")
    return vae


def vae_encode(name, vae, images, sample_posterior, device):
    """
    Encode images to latents using VAE.
    
    Args:
        name: VAE type ("sdxl", "sd3", "dc-ae", "AutoencoderDC")
        vae: VAE model
        images: Input images [B, C, H, W]
        sample_posterior: Whether to sample from posterior or use mode
        device: Device to use
    
    Returns:
        Encoded latents [B, C, H, W]
    """
    # Get VAE dtype and convert images to match
    vae_dtype = next(vae.parameters()).dtype
    images = images.to(device, dtype=vae_dtype)
    
    if name == "sdxl" or name == "sd3":
        posterior = vae.encode(images).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
    elif "dc-ae" in name:
        ae = vae
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
        z = ae.encode(images)
        z = z * scaling_factor
    elif "AutoencoderDC" in name:
        ae = vae
        scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
        z = ae.encode(images)[0]
        z = z * scaling_factor
    else:
        print("error load vae")
        exit()
    return z


def vae_decode(name, vae, latent):
    """
    Decode latents to images using VAE.
    
    Args:
        name: VAE type ("sdxl", "sd3", "dc-ae", "AutoencoderDC")
        vae: VAE model
        latent: Input latents [B, C, H, W]
    
    Returns:
        Decoded images [B, C, H, W]
    """
    # Get VAE dtype and convert latents to match
    vae_dtype = next(vae.parameters()).dtype
    latent = latent.to(vae.device, dtype=vae_dtype)
    
    if name == "sdxl" or name == "sd3":
        latent = (latent.detach() / vae.config.scaling_factor) + vae.config.shift_factor
        samples = vae.decode(latent).sample
    elif "dc-ae" in name:
        ae = vae
        vae_scale_factor = (
            2 ** (len(ae.config.encoder_block_out_channels) - 1)
            if hasattr(ae, "config") and ae.config is not None
            else 32
        )
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
        if latent.shape[-1] * vae_scale_factor > 4000 or latent.shape[-2] * vae_scale_factor > 4000:
            from patch_conv import convert_model
            ae = convert_model(ae, splits=4)
        samples = ae.decode(latent.detach() / scaling_factor)
    elif "AutoencoderDC" in name:
        ae = vae
        scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
        try:
            samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
        except torch.cuda.OutOfMemoryError as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            ae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024)
            samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
    else:
        print("error load vae")
        exit()
    return samples
