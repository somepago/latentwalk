import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Union
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from PIL import Image
import numpy as np

from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback


class ImageProjectionLayer(nn.Module):
    """
    Projects SigLIP2 image embeddings to T5 text embedding dimension space.
    """
    def __init__(self, image_embed_dim: int = 1152, text_embed_dim: int = 4096, hidden_dim: int = 2048):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(image_embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, text_embed_dim),
            nn.LayerNorm(text_embed_dim)
        )
        
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: [batch_size, image_embed_dim]
        Returns:
            projected_embeds: [batch_size, text_embed_dim]
        """
        return self.projection(image_embeds)


class WanImageToImagePipeline(WanPipeline):
    """
    Pipeline for image-to-video generation using Wan with SigLIP2 image conditioning.
    
    This pipeline extends WanPipeline to support image conditioning by encoding reference images
    with SigLIP2 and projecting them to the text embedding space.
    """
    
    def __init__(
        self,
        tokenizer,
        text_encoder,
        vae,
        scheduler,
        transformer=None,
        transformer_2=None,
        boundary_ratio=None,
        expand_timesteps=False,
        siglip_model_id: str = "google/siglip2-base-patch16-512",
        image_projection_layer: Optional[ImageProjectionLayer] = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
        )
        
        # Initialize SigLIP2 components
        self.siglip_model = AutoModel.from_pretrained(siglip_model_id).eval()
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_id)
        
        # Initialize or create image projection layer
        if image_projection_layer is None:
            # Get dimensions from models
            siglip_dim = self.siglip_model.config.vision_config.hidden_size  # 1152 for base model
            text_dim = self.text_encoder.config.d_model  # 4096 for UMT5-XXL
            self.image_projection = ImageProjectionLayer(siglip_dim, text_dim)
        else:
            self.image_projection = image_projection_layer
            
        # Register the new components
        self.register_modules(
            siglip_model=self.siglip_model,
            image_projection=self.image_projection,
        )

    def _encode_image(
        self,
        image: Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Encode reference image(s) using SigLIP2.
        
        Args:
            image: Input image(s) to encode
            device: Target device
            dtype: Target dtype
            
        Returns:
            image_embeds: [batch_size, image_embed_dim]
        """
        device = device or self._execution_device
        dtype = dtype or self.siglip_model.dtype
        
        # Handle different input types
        if isinstance(image, (Image.Image, np.ndarray)):
            image = [image]
        elif isinstance(image, torch.Tensor):
            # Assume it's already processed
            return image.to(device=device, dtype=dtype)
            
        # Process images
        inputs = self.siglip_processor(images=image, return_tensors="pt").to(device)
        
        # Extract image features
        with torch.no_grad():
            image_embeds = self.siglip_model.get_image_features(**inputs)
            
        return image_embeds.to(dtype=dtype)

    def _project_image_embeds(
        self,
        image_embeds: torch.Tensor,
        max_sequence_length: int = 226,
        num_videos_per_prompt: int = 1,
    ) -> torch.Tensor:
        """
        Project image embeddings to text embedding space and format for sequence processing.
        
        Args:
            image_embeds: [batch_size, image_embed_dim]
            max_sequence_length: Maximum sequence length for padding
            num_videos_per_prompt: Number of videos per prompt
            
        Returns:
            projected_embeds: [batch_size * num_videos_per_prompt, max_sequence_length, text_embed_dim]
        """
        batch_size = image_embeds.shape[0]
        
        # Project to text embedding dimension
        projected_embeds = self.image_projection(image_embeds)  # [batch_size, text_embed_dim]
        
        # Expand to sequence length (repeat the image embedding across sequence)
        # This creates a "sequence" where each position has the same image information
        projected_embeds = projected_embeds.unsqueeze(1).expand(-1, max_sequence_length, -1)
        
        # Handle multiple videos per prompt
        projected_embeds = projected_embeds.repeat(1, num_videos_per_prompt, 1)
        projected_embeds = projected_embeds.view(batch_size * num_videos_per_prompt, max_sequence_length, -1)
        
        return projected_embeds

    def encode_image_prompt(
        self,
        image: Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        image_conditioning_scale: float = 1.0,
        text_conditioning_scale: float = 0.5,
    ):
        """
        Encode both image and text prompts, combining them for conditioning.
        
        Args:
            image: Reference image(s) for conditioning
            prompt: Optional text prompt to combine with image conditioning
            negative_prompt: Negative text prompt
            do_classifier_free_guidance: Whether to use CFG
            num_videos_per_prompt: Number of videos per prompt
            prompt_embeds: Pre-computed text embeddings
            negative_prompt_embeds: Pre-computed negative text embeddings
            image_embeds: Pre-computed image embeddings
            max_sequence_length: Maximum sequence length
            device: Target device
            dtype: Target dtype
            image_conditioning_scale: Scale factor for image conditioning
            text_conditioning_scale: Scale factor for text conditioning (when combining with text)
            
        Returns:
            Tuple of (combined_prompt_embeds, combined_negative_prompt_embeds)
        """
        device = device or self._execution_device
        
        # Encode image if not provided
        if image_embeds is None:
            image_embeds = self._encode_image(image, device=device, dtype=dtype)
        
        # Project image embeddings to text space
        image_prompt_embeds = self._project_image_embeds(
            image_embeds, max_sequence_length, num_videos_per_prompt
        )
        
        # Scale image conditioning
        image_prompt_embeds = image_prompt_embeds * image_conditioning_scale
        
        # Handle text prompt combination
        if prompt is not None and prompt_embeds is None:
            text_prompt_embeds, text_negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            
            # Combine image and text embeddings
            combined_prompt_embeds = (
                image_prompt_embeds + text_prompt_embeds * text_conditioning_scale
            )
            combined_negative_prompt_embeds = text_negative_prompt_embeds
            
        elif prompt_embeds is not None:
            # Use provided text embeddings
            combined_prompt_embeds = image_prompt_embeds + prompt_embeds * text_conditioning_scale
            combined_negative_prompt_embeds = negative_prompt_embeds
            
        else:
            # Image-only conditioning
            combined_prompt_embeds = image_prompt_embeds
            
            # Create negative embeddings for CFG
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                # Use zero embeddings as negative conditioning
                combined_negative_prompt_embeds = torch.zeros_like(combined_prompt_embeds)
            else:
                combined_negative_prompt_embeds = negative_prompt_embeds
        
        return combined_prompt_embeds, combined_negative_prompt_embeds

    def check_image_inputs(
        self,
        image,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        """Extended input validation for image-to-video pipeline."""
        # Call parent validation
        super().check_inputs(
            prompt, negative_prompt, height, width, prompt_embeds, 
            negative_prompt_embeds, callback_on_step_end_tensor_inputs, guidance_scale_2
        )
        
        # Image-specific validation
        if image is None and image_embeds is None:
            raise ValueError("Must provide either `image` or `image_embeds`")
            
        if image is not None and image_embeds is not None:
            raise ValueError("Cannot provide both `image` and `image_embeds`")

    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        image_conditioning_scale: float = 1.0,
        text_conditioning_scale: float = 0.5,
    ):
        r"""
        Generate video from reference image with optional text conditioning.

        Args:
            image: Reference image for video generation
            prompt: Optional text prompt to combine with image conditioning
            negative_prompt: Negative text prompt
            height: Height of generated video
            width: Width of generated video
            num_frames: Number of frames in generated video
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            guidance_scale_2: Guidance scale for second transformer (if applicable)
            num_videos_per_prompt: Number of videos per prompt
            generator: Random generator for reproducibility
            latents: Pre-generated latents
            prompt_embeds: Pre-computed text embeddings
            negative_prompt_embeds: Pre-computed negative text embeddings
            image_embeds: Pre-computed image embeddings
            output_type: Output format ("np" or "latent")
            return_dict: Whether to return dict or tuple
            attention_kwargs: Additional attention parameters
            callback_on_step_end: Step callback function
            callback_on_step_end_tensor_inputs: Tensors to pass to callback
            max_sequence_length: Maximum sequence length for embeddings
            image_conditioning_scale: Scale factor for image conditioning strength
            text_conditioning_scale: Scale factor for text conditioning strength
            
        Returns:
            WanPipelineOutput or tuple with generated video frames
        """
        
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_image_inputs(
            image, prompt, negative_prompt, height, width,
            prompt_embeds, negative_prompt_embeds, image_embeds,
            callback_on_step_end_tensor_inputs, guidance_scale_2
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = 1

        # 3. Encode image and text prompts (MODIFIED SECTION)
        prompt_embeds, negative_prompt_embeds = self.encode_image_prompt(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            image_conditioning_scale=image_conditioning_scale,
            text_conditioning_scale=text_conditioning_scale,
        )

        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop (same as parent)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(transformer_dtype)
                if self.config.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # 7. Decode latents (same as parent)
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


# Example usage function
def create_wan_i2v_pipeline_from_pretrained(
    model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    siglip_model_id: str = "google/siglip2-base-patch16-512",
    projection_checkpoint: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    Create a WanImageToVideoPipeline from pretrained models.
    
    Args:
        model_id: HuggingFace model ID for the base Wan model
        siglip_model_id: HuggingFace model ID for SigLIP2
        projection_checkpoint: Path to trained projection layer weights
        device: Device to load models on
        torch_dtype: Data type for models
        
    Returns:
        Configured WanImageToVideoPipeline
    """
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    
    # Load base components
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToImagePipeline.from_pretrained(
        model_id, 
        vae=vae, 
        torch_dtype=torch_dtype,
        siglip_model_id=siglip_model_id
    )
    
    # Configure scheduler
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    
    # Load trained projection layer if available
    if projection_checkpoint:
        checkpoint = torch.load(projection_checkpoint, map_location="cpu")
        pipe.image_projection.load_state_dict(checkpoint)
    
    pipe.to(device)
    return pipe