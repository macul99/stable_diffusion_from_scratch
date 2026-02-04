import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (Batch_Size, Seq_Len, D_Embed)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len, D_Embed/3)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(intermim_shape).transpose(1, 2)  # (Batch_Size, N_Heads, Seq_Len, D_Head)
        k = k.view(intermim_shape).transpose(1, 2)  # (Batch_Size, N_Heads, Seq_Len, D_Head)
        v = v.view(intermim_shape).transpose(1, 2)  # (Batch_Size, N_Heads, Seq_Len, D_Head)

        weight = torch.matmul(q, k.transpose(-2, -1))  # (Batch_Size, N_Heads, Seq_Len, Seq_Len)
        
        if causal_mask:
            # mask where the upper triangle (above the main diagonal) is made up of 1
            mask = torch.ones_like(weight).triu(1)
            weight = weight.masked_fill(mask, -torch.inf)

        weight = weight / math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = torch.matmul(weight, v)  # (Batch_Size, N_Heads, Seq_Len, D_Head)

        # (Batch_Size, Seq_Len, N_Heads, D_Head)
        output = output.transpose(1, 2)

        output = output.contiguous().view(input_shape)  # (Batch_Size, Seq_Len, D_Embed)
        output = self.out_proj(output)  # (Batch_Size, Seq_Len, D_Embed)

        return output
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (Batch_Size, Seq_Len_Q, D_Embed_Q)
        # y: (Batch_Size, Seq_Len_KV, D_Embed_KV)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply query by Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1,2).contiguous().view(input_shape)

        output = self.out_proj(output)

        return output