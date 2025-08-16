import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMultiHeadCrossAttention(nn.Module):
    """
    Custom multi-head cross attention that can handle different embedding dimensions
    for query and key/value projections.
    """
    def __init__(self, query_dim, kv_dim, num_heads, head_dim=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (query_dim // num_heads)
        self.scale = self.head_dim ** -0.5
        
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        assert kv_dim % num_heads == 0, "kv_dim must be divisible by num_heads"
        
        # Query projection (from query_dim to query_dim)
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        
        # Key and Value projections (from kv_dim to query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, query_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim, bias=False)
        
        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: (B, N, query_dim) - query tensor
            key_value: (B, M, kv_dim) - key/value tensor
            mask: (B, N, M) or (B, M) - attention mask
        Returns:
            output: (B, N, query_dim) - attended output
        """
        B, N, _ = query.shape
        _, M, _ = key_value.shape
        
        # Project query, key, value
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = self.k_proj(key_value).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, M, head_dim)
        v = self.v_proj(key_value).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, M, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, M)
        
        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, M)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(1)  # (B, 1, N, M)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        
        # Reshape and project output
        output = output.transpose(1, 2).reshape(B, N, self.query_dim)  # (B, N, query_dim)
        output = self.out_proj(output)
        output = self.proj_drop(output)
        
        return output

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
    
class CrossAttentionProjector(nn.Module):
    """
    input features [batch_size, num_conditione, 256, 384]
    output features [ batch_size, 1, 300, 2304 ] , mask of all 1s with shape [batch_size, 300]
    """
    def __init__(self, input_dim=384, num_op_tokens=300, op_dim=2304, num_heads=8):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_op_tokens = num_op_tokens
        self.op_dim = op_dim
        
        # Learnable query tokens for cross attention
        self.query_tokens = nn.Parameter(torch.randn(1, num_op_tokens, op_dim) / op_dim**0.5)
        
        # Cross attention block with different embedding dimensions
        # query_dim=2304 (output dimension), kv_dim=384 (input dimension)
        self.cross_attention_block = FlexibleMultiHeadCrossAttention(
            query_dim=op_dim,  # 2304
            kv_dim=input_dim,   # 384
            num_heads=num_heads,
            attn_drop=0.0,
            proj_drop=0.0
        )
        
        # Output projection and activation
        self.output_projection = nn.Linear(op_dim, op_dim)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # x shape: [batch_size, num_conditione, 256, 384]
        batch_size, num_conditione, seq_len, hidden_dim = x.shape
        
        # Reshape to combine all conditions: [batch_size, num_conditione * 256, 384]
        x_combined = x.reshape(batch_size, num_conditione * seq_len, hidden_dim)
        
        # Expand query tokens to match batch size
        # [batch_size, 300, 2304]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Apply cross attention: query_tokens attend to all conditions combined
        # query_tokens: [batch_size, 300, 2304] (query_dim=2304)
        # x_combined: [batch_size, num_conditione * 256, 384] (kv_dim=384)
        attended_features = self.cross_attention_block(query_tokens, x_combined)
        
        # Apply output projection and activation
        attended_features = self.output_projection(attended_features)
        attended_features = self.gelu(attended_features)
        
        # Add the required dimension to get [batch_size, 1, 300, 2304]
        output = attended_features.unsqueeze(1)
        
        # Create mask of all 1s with shape [batch_size, 300]
        mask = torch.ones(batch_size, self.num_op_tokens, device=x.device, dtype=x.dtype)
        
        return output, mask


if __name__ == "__main__":
    # projector = Projector()
    # x = torch.randn(10, 256, 384)
    # x, mask = projector(x)
    # print(x.shape) # torch.Size([10, 1, 300, 2304])
    # print(mask.shape) # torch.Size([10, 300])

    projector = CrossAttentionProjector()
    im_conditioning  = torch.randn(10 ,2, 256, 384)
    x, mask = projector(im_conditioning)
    print(x.shape) # torch.Size([10, 1, 300, 2304])
    print(mask.shape) # torch.Size([10, 300])








