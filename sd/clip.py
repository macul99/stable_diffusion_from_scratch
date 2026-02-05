import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    # token embedding + position embedding

    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Embedding_Dim)
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x
    
class CLIPLayer(nn.Module):
    # self-attention + MLP with layer norm and skip connection
    def __init__(self, n_heads: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_heads, n_embd)

        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(), # x * torch.sigmoid(1.702 * x)
            nn.Linear(n_embd * 4, n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residue = x

        x = self.layernorm_1(x)
        x = self.attention(x, casual_mask=True)
        x = x + residue
        residue = x
        x = self.layernorm_2(x)
        x = self.mlp(x)
        x = x + residue
        return x

class CLIP(nn.Module):
    # Embedding + 12 x CLIPLayer + LayerNorm
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList(
            [CLIPLayer(12, 768) for i in range(12)]
        )

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Embedding_Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_Size, Seq_Len, Embedding_Dim)
        output = self.layernorm(state)
        return output