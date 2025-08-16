"""
MVP inference script for Sana 1600M model.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
import os

from models.diffusion_model import Sana_600M
from models.diffusion_utils import FlowMatchingScheduler, DPMSolver

# Text encoder imports
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5EncoderModel

# VAE imports
try:
    from diffusers import AutoencoderDC
    VAE_AVAILABLE = True
except ImportError:
    print("Warning: diffusers not installed. VAE decoding will not be available.")
    print("Install with: pip install diffusers")
    VAE_AVAILABLE = False



def get_tokenizer_and_text_encoder(name="gemma-2-2b-it", device="cuda"):
    """
    Load tokenizer and text encoder.
    Supports T5, Gemma, and Qwen models.
    """
    text_encoder_dict = {
        "T5": "DeepFloyd/t5-v1_1-xxl",
        "T5-small": "google/t5-v1_1-small",
        "T5-base": "google/t5-v1_1-base",
        "T5-large": "google/t5-v1_1-large",
        "T5-xl": "google/t5-v1_1-xl",
        "T5-xxl": "google/t5-v1_1-xxl",
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
        "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    }

    assert name in list(text_encoder_dict.keys()), f"Not supported text encoder: {name}"

    if "T5" in name:
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_dict[name])
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_dict[name], torch_dtype=torch.float16).to(device)
    elif "gemma" in name or "Qwen" in name:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dict[name])
        tokenizer.padding_side = "right"
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(text_encoder_dict[name], torch_dtype=torch.bfloat16)
            .get_decoder()
            .to(device)
        )
    else:
        raise ValueError(f"Error loading text encoder: {name}")

    return tokenizer, text_encoder


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    max_sequence_length: int = 300,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text prompts using the text encoder.
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    # Tokenize
    text_inputs = tokenizer(
        prompt,
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Encode
    with torch.no_grad():
        text_embeddings = text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )[0]

        # Use the same indexing pattern as original Sana
        select_index = [0] + list(range(-max_sequence_length + 1, 0))
        text_embeddings = text_embeddings[:, None][:, :, select_index]
        attention_mask = text_inputs.attention_mask[:, select_index]

    return text_embeddings, attention_mask


def load_vae(device="cuda", dtype=torch.float16):
    """
    Load the DC-AE VAE decoder.
    """
    if not VAE_AVAILABLE:
        return None

    print("Loading DC-AE VAE decoder...")
    vae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        torch_dtype=dtype
    ).to(device)
    vae.eval()
    print("VAE loaded successfully")
    return vae


def decode_latents(latents: torch.Tensor, vae=None, vae_scale_factor=0.41407) -> List[Image.Image]:
    """
    Decode latents to images using VAE or create visualization.

    Args:
        latents: Generated latents [B, C, H, W]
        vae: VAE decoder model
        vae_scale_factor: Scaling factor for DC-AE (0.41407)

    Returns:
        List of PIL images
    """
    if vae is not None:
        # Decode with actual VAE
        with torch.no_grad():
            # Scale latents
            latents = latents / vae_scale_factor
            # Ensure latents have the same dtype as VAE
            latents = latents.to(dtype=vae.dtype)
            # Decode to images
            images = vae.decode(latents).sample
            # Convert to PIL images
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype(np.uint8)
            return [Image.fromarray(img) for img in images]
    else:
        # Fallback: visualize latents directly
        return latents_to_image(latents)


def latents_to_image(latents: torch.Tensor) -> List[Image.Image]:
    """
    Convert latents to images without VAE (visualization only).
    """
    # Normalize latents to [0, 1]
    latents = (latents - latents.min()) / (latents.max() - latents.min())

    # Convert to images
    images = []
    for i in range(latents.shape[0]):
        # Take mean across channels and convert to RGB
        latent = latents[i].mean(dim=0)
        latent = latent.cpu().numpy()

        # Resize to larger size for visualization
        from scipy.ndimage import zoom
        latent = zoom(latent, (8, 8), order=1)

        # Convert to uint8
        latent = (latent * 255).astype(np.uint8)

        # Create RGB image
        image = Image.fromarray(np.stack([latent] * 3, axis=-1))
        images.append(image)

    return images


def load_model(
    checkpoint_path: str,
    device: torch.device = torch.device("cuda"),
    use_ema: bool = True,
) -> Sana_600M:
    """
    Load model from checkpoint.
    """
    # Create model
    model = Sana_600M()

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Handle EMA weights
        if use_ema and "ema_shadow" in checkpoint:
            print("Using EMA weights")
            state_dict = checkpoint["ema_shadow"]

        # Load state dict with strict=False to handle missing/unexpected keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
            print(f"First few missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            print(f"First few unexpected keys: {unexpected_keys[:5]}...")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")

    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def generate_images(
    model: Sana_600M,
    prompts: Union[str, List[str]],
    tokenizer,
    text_encoder,
    vae=None,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    height: int = 32,
    width: int = 32,
    num_images_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cuda"),
    flow_shift: float = 3.0,
    max_sequence_length: int = 300,
) -> List[Image.Image]:
    """
    Generate images from text prompts.

    Args:
        model: Sana model
        prompts: Text prompt(s)
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        height: Latent height
        width: Latent width
        num_images_per_prompt: Number of images per prompt
        generator: Random generator
        device: Device to run on
        flow_shift: Flow shift parameter

    Returns:
        List of generated images
    """
    # Handle single prompt
    if isinstance(prompts, str):
        prompts = [prompts]

    batch_size = len(prompts) * num_images_per_prompt

    # Encode prompts with the real text encoder
    text_embeddings, attention_mask = encode_prompt(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompts * num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
    ) # shapes for len(prompts) = 1, num_images_per_prompt = 1 - text_embeddings:[ 1, 1, 300, 2304 ], attention_mask: [1, 300]

    # Create null embeddings (empty string)
    null_text_embeddings, null_attention_mask = encode_prompt(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=[""] * batch_size,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    # Create scheduler and sampler
    scheduler = FlowMatchingScheduler(
        num_train_timesteps=1000,
        shift=flow_shift,
    )

    sampler = DPMSolver(
        model=model,
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        flow_shift=flow_shift,
    )

    # Generate latents
    print(f"Generating {batch_size} images...")
    latents = sampler.sample(
        text_embeddings=text_embeddings,
        attention_mask=attention_mask,
        null_embeddings=null_text_embeddings,
        height=height,
        width=width,
        generator=generator,
        device=device,
    )

    # Decode latents to images
    images = decode_latents(latents, vae=vae)

    return images


def main():
    parser = argparse.ArgumentParser(description="Generate images with Sana 600M")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/sana_600m.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights if available")
    parser.add_argument("--text_encoder", type=str, default="gemma-2-2b-it",
                        help="Text encoder to use (e.g., gemma-2-2b-it, T5-xl)")

    # Generation arguments
    parser.add_argument("--prompt", type=str, default="A beautiful landscape with mountains and trees",
                        help="Text prompt for generation")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--height", type=int, default=32, help="Latent height")
    parser.add_argument("--width", type=int, default=32, help="Latent width")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="Guidance scale")
    parser.add_argument("--flow_shift", type=float, default=3.0, help="Flow shift parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_vae", action="store_true", help="Skip VAE decoding (visualize latents only)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        use_ema=args.use_ema,
    )
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Load text encoder
    print(f"Loading text encoder: {args.text_encoder}")
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=args.text_encoder, device=str(device))
    print(f"Text encoder loaded successfully")

    # Load VAE if requested
    vae = None
    if not args.no_vae:
        vae = load_vae(device=device, dtype=torch.float16)
        if vae is None:
            print("Warning: VAE not available, will save latent visualizations instead")

    # Set random seed
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Generate images
    images = generate_images(
        model=model,
        prompts=args.prompt,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_images_per_prompt=args.num_images,
        generator=generator,
        device=device,
        flow_shift=args.flow_shift,
    )

    # Save images
    for i, image in enumerate(images):
        output_path = os.path.join(args.output_dir, f"generated_{i:03d}.png")
        image.save(output_path)
        print(f"Saved image to {output_path}")

    print(f"\nGeneration complete! Generated {len(images)} images.")
    print(f"Prompt: {args.prompt}")
    print(f"Settings: steps={args.steps}, guidance={args.guidance_scale}, size={args.height}x{args.width}")


if __name__ == "__main__":
    main()
