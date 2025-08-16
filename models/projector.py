import torch
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self,):
        super().__init__()
        # input features [num_images, 256, 384]
        # output features [ num_images, 1, 300, 2304 ] , mask of 1s and 0s with shape [num_images, 300]
        self.linear = nn.Linear(384, 2304)
        self.gelu = nn.GELU(approximate="tanh")

    def from_pretrained(self, path):
        self.linear.load_state_dict(torch.load(path))

    def forward(self, x):
        # x shape: [num_images, 256, 384]
        num_images, seq_len, hidden_dim = x.shape
        
        # Reshape to process all tokens: [num_images * 256, 384]
        x = x.reshape(-1, hidden_dim)
        
        # Apply linear transformation: [num_images * 256, 2304]
        x = self.linear(x)
        x = self.gelu(x)
        
        # Reshape back: [num_images, 256, 2304]
        x = x.reshape(num_images, seq_len, -1)
        
        # Add the required dimension and pad to 300 tokens if needed
        # [num_images, 1, 256, 2304]
        x = x.unsqueeze(1)
        
        # Pad the sequence dimension to 300 if it's shorter
        if x.size(2) < 300:
            # Pad with zeros if sequence is shorter than 300
            padding = torch.zeros(num_images, 1, 300 - x.size(2), 2304, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=2)
        else:
            raise ValueError(f"Sequence length from DINO is greater than 300: {x.size(2)}")

        # Create mask of 1s and 0s with shape [num_images, 300]
        mask = torch.ones(num_images, 300, device=x.device, dtype=x.dtype)
        mask[:, x.size(2):] = 0
        
        # Final shape: [num_images, 1, 300, 2304] or [num_images, 1, seq_len, 2304] if seq_len > 300
        return x, mask
    

if __name__ == "__main__":
    projector = Projector()
    x = torch.randn(10, 256, 384)
    x, mask = projector(x)
    print(x.shape) # torch.Size([10, 1, 300, 2304])
    print(mask.shape) # torch.Size([10, 300])







