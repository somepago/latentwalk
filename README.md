# latentwalk
High level goal - do interpolation in image conditioner space of a diffusion model - using GPRO

Image conditoner - Dino V3
Diffusion model - SANA (or anything smaller - if any)
Dataset - simple shapes 
Projection - TBD

Todos
- Generate dataset + dataloading - Kamal ✅
- Inference pipeline for diffusion - Gowthami ✅
- End to end inference pipeline with dino + projector + diffusion - Gowthami 
- Training pipeline with diffusion loss (without interpolation)
- Finetuning pipeling with GRPO (with image_interpolation)
- Reward model + metrics  

  
## Test

```
uv pip install -e .

python tests/test_shapes.py
python tests/test_shapes.py
```

## Image to image inference

```
python inference.py --steps 2 --image_prompts <image_path>
```