"""
MVP utilities for Sana training including loss functions and scheduler.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
import math


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute the density for sampling timesteps using different weighting schemes.
    """
    if weighting_scheme == "uniform":
        u = torch.rand(batch_size, device=device)
    elif weighting_scheme == "logit_normal":
        # Sample from logit-normal distribution
        normal_samples = torch.randn(batch_size, device=device) * logit_std + logit_mean
        u = torch.sigmoid(normal_samples)
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")

    return u


class FlowMatchingScheduler:
    """
    Flow matching scheduler for training and sampling.
    Implements linear flow interpolation between noise and data.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        noise_schedule: str = "linear_flow",
        pred_sigma: bool = False,
        learn_sigma: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.noise_schedule = noise_schedule
        self.pred_sigma = pred_sigma
        self.learn_sigma = learn_sigma

        # Create timestep schedule
        self.timesteps = torch.linspace(0, 1, num_train_timesteps + 1)[:-1]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples using flow matching.

        Flow matching formula: x_t = (1 - t) * x_0 + t * noise
        where t is in [0, 1]
        """
        # Ensure timesteps are normalized to [0, 1]
        t = timesteps.float() / self.num_train_timesteps

        # Reshape for broadcasting
        t = t.view(-1, 1, 1, 1)

        # Linear interpolation between data and noise
        noisy_samples = (1 - t) * original_samples + t * noise

        return noisy_samples

    def get_velocity(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the velocity for flow matching.

        Velocity v = noise - original_samples
        """
        return noise - original_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Single denoising step.
        """
        # For flow matching, we predict velocity
        velocity = model_output

        # Update sample: x_{t-dt} = x_t - dt * v_t
        dt = 1.0 / self.num_train_timesteps
        prev_sample = sample - dt * velocity

        if return_dict:
            return {"prev_sample": prev_sample, "velocity": velocity}
        return prev_sample


class FlowMatchingLoss(nn.Module):
    """
    Flow matching loss for training diffusion models.
    """

    def __init__(
        self,
        prediction_type: str = "flow_velocity",
        weighting_scheme: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        flow_shift: float = 3.0,
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.flow_shift = flow_shift

    def compute_loss_weighting(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute loss weighting based on timestep and weighting scheme.
        """
        # Normalize timesteps to [0, 1]
        t = timesteps.float() / 1000.0

        if self.weighting_scheme == "uniform":
            # Uniform weighting
            weights = torch.ones_like(t)
        elif self.weighting_scheme == "logit_normal":
            # Logit-normal weighting to emphasize certain timesteps
            # This helps focus training on more difficult timesteps
            # Apply sigmoid weighting based on the logit-normal scheme
            weights = 1.0 / (1.0 + torch.exp(-self.flow_shift * (t - 0.5)))
        else:
            weights = torch.ones_like(t)

        return weights.view(-1, 1, 1, 1)

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            model_pred: Model prediction (velocity)
            target: Target velocity (noise - original_samples)
            timesteps: Timesteps for each sample
            reduction: How to reduce the loss ("mean", "none")
        """
        # Compute MSE loss
        loss = (model_pred - target) ** 2

        # Apply timestep-dependent weighting
        weights = self.compute_loss_weighting(timesteps)
        weighted_loss = weights * loss

        # Reduce loss
        if reduction == "mean":
            return weighted_loss.mean()
        elif reduction == "none":
            return weighted_loss.view(weighted_loss.shape[0], -1).mean(dim=1)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_schedule: str = "constant",
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler.
    """
    if lr_schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda current_step: 1.0 if current_step >= num_warmup_steps else current_step / num_warmup_steps,
        )
    elif lr_schedule == "cosine":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return current_step / num_warmup_steps
            progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


def sample_timesteps(
    batch_size: int,
    num_train_timesteps: int,
    weighting_scheme: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sample timesteps for training based on weighting scheme.
    """
    # Sample density
    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=batch_size,
        logit_mean=logit_mean,
        logit_std=logit_std,
        device=device,
    )

    # Convert to timesteps
    timesteps = (u * num_train_timesteps).long()
    timesteps = torch.clamp(timesteps, 0, num_train_timesteps - 1)

    return timesteps


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer for model training.
    """
    # Get parameters that require grad
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    return optimizer


class EMA:
    """
    Exponential Moving Average for model parameters.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def compute_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm for monitoring."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

class DPMSolver:
    """
    Simple DPM-Solver for flow matching inference.
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: FlowMatchingScheduler,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        flow_shift: float = 3.0,
    ):
        self.model = model
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.flow_shift = flow_shift

        # Create timestep schedule for inference (reverse order for sampling)
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

    @torch.no_grad()
    def sample(
        self,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        null_embeddings: torch.Tensor,
        height: int = 32,
        width: int = 32,
        generator: Optional[torch.Generator] = None,
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        """
        Generate samples using DPM-Solver.

        Args:
            text_embeddings: Text embeddings [B, 1, L, D]
            attention_mask: Attention mask [B, L]
            null_embeddings: Null embeddings [B, 1, L, D]
            height: Latent height
            width: Latent width
            generator: Random generator for reproducibility
            device: Device to run on

        Returns:
            Generated latents [B, C, H, W]
        """
        batch_size = text_embeddings.shape[0]
        latent_channels = 32  # DC-AE VAE uses 32 channels

        # Initialize from noise
        latents = torch.randn(
            (batch_size, latent_channels, height, width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        # Prepare for classifier-free guidance
        if self.guidance_scale > 1.0:
            # Concatenate conditional and unconditional embeddings
            text_embeddings = torch.cat([text_embeddings, null_embeddings], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

        # DPM-Solver sampling loop
        # Store previous model output for 2nd order step
        prev_output = None
        for i, t in enumerate(self.timesteps):
            # Prepare timestep (scale to scheduler range)
            timestep = torch.full((batch_size,), t * self.scheduler.num_train_timesteps, device=device)

            if self.guidance_scale > 1.0:
                timestep = torch.cat([timestep, timestep], dim=0)
                latent_input = torch.cat([latents, latents], dim=0)
            else:
                latent_input = latents

            # Model prediction
            model_output = self.model(
                x=latent_input,
                timestep=timestep,
                y=text_embeddings,
                mask=attention_mask,
            )

            # Apply classifier-free guidance
            if self.guidance_scale > 1.0:
                cond_pred, uncond_pred = model_output.chunk(2, dim=0)
                model_output = uncond_pred + self.guidance_scale * (cond_pred - uncond_pred)

            # DPM-Solver update
            if prev_output is None:
                # First step: Euler method
                dt = self.timesteps[0] - self.timesteps[1] if len(self.timesteps) > 1 else self.timesteps[0]
                latents = latents - dt * model_output
            else:
                # Subsequent steps: 2nd order DPM-Solver
                dt = self.timesteps[i-1] - self.timesteps[i]
                latents = latents - dt * (1.5 * model_output - 0.5 * prev_output)

            prev_output = model_output

        return latents


if __name__ == "__main__":
    # Test scheduler
    scheduler = FlowMatchingScheduler()

    # Test noise addition
    batch_size = 2
    original = torch.randn(batch_size, 4, 32, 32)
    noise = torch.randn_like(original)
    timesteps = torch.tensor([250, 750])

    noisy = scheduler.add_noise(original, noise, timesteps)
    velocity = scheduler.get_velocity(original, noise, timesteps)

    print(f"Noisy shape: {noisy.shape}")
    print(f"Velocity shape: {velocity.shape}")

    # Test loss
    loss_fn = FlowMatchingLoss()
    model_pred = torch.randn_like(velocity)
    loss = loss_fn(model_pred, velocity, timesteps)
    print(f"Loss: {loss.item():.4f}")
