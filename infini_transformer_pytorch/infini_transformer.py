import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v

def default(v, d):
    return v if exists(v) else d

# classes

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class FeedForward(Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        dim_inner = int(mult * dim * 2 / 3)

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, dim_inner * 2)
        self.proj_out = nn.Linear(dim_inner, dim)

    def forward(self, x):
        x = self.norm(x)
        x, gates = self.proj_in(x).chunk(2, dim = -1)
        x = F.gelu(gates) * x
        return self.proj_out(x)

# attention

class CausalAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 128,
        heads = 8
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.scale = dim_head ** -0.5
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(self, x):
        x = self.norm(x)

        x = self.to_qkv(x)
        q, k, v = self.split_heads(x)

        q = q * self.scale

        # similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # causal mask

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = sim.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attend

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = self.merge_heads(out)
        return self.to_out(out)

# main class

class InfiniTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 128,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        for _ in range(depth):

            attn = CausalAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads
            )

            ff = FeedForward(
                dim = dim,
                mult = ff_mult
            )

            self.layers.append(ModuleList([attn, ff]))

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(
        self,
        x
    ):
        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        embed = self.norm(x)

        return self.to_logits(embed)
