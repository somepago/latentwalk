# latentwalk
High level goal - do interpolation in image conditioner space of a diffusion model - using GPRO

Image conditoner - Dino V3
Diffusion model - SANA (or anything smaller - if any)
Dataset - simple shapes 
Projection - TBD

Todos
- Generate dataset + dataloading - Kamal
- Inference pipeline for diffusion - Gowthami
- End to end inference pipeline with dino + projector + diffusion - 
- Training pipeline with loss 
- Reward model + metrics  
  
## Test

```
uv pip install -e .

python tests/test_shapes.py
python tests/test_shapes.py
```
