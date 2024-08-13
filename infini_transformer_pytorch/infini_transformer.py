from __future__ import annotations
from typing import Tuple, List, NamedTuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import einsum, rearrange, reduce
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

# constants

class Memories(NamedTuple):
    kv_mem: Tensor
    k_norm: Tensor

class TransformerReturn(NamedTuple):
    logits: Tensor
    cached_kvs: List[Tensor] | None
    past_memories: List[Memories] | None

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def detach_memories_(memories: List[Memories]):
    for (mem_kv, mem_norm) in memories:
        mem_kv.detach_()
        mem_norm.detach_()

def detach_cached_kv_(cached_kvs: List[Tensor]):
    for cached_kv in cached_kvs:
        cached_kv.detach_()

# classes

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(mult * dim * 2 / 3)

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, dim_inner * 2)
        self.proj_out = nn.Linear(dim_inner, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x, gates = self.proj_in(x).chunk(2, dim = -1)
        x = F.gelu(gates) * x
        x = self.dropout(x)
        return self.proj_out(x)

# fastweight memory

def retrieve_from_kv_memories(t, past_memories: Memories, eps = 1e-10):
    past_memories_kv, past_memories_norm = past_memories

    numer = einsum(t, past_memories_kv, 'b h n dk, b h dk dv -> b h n dv')
    denom = einsum(t, past_memories_norm, 'b h n d, b h d -> b h n')

    denom = rearrange(denom, '... -> ... 1')
    return numer / denom.clamp(min = eps) # eq (3)

class FastweightMemory(Module):
    def __init__(
        self,
        heads: int,
        head_gate_init_value = 10.,
        use_mem_delta_rule = False,
    ):
        super().__init__()
        self.use_mem_delta_rule = use_mem_delta_rule
        self.head_gates = nn.Parameter(torch.ones(heads) * head_gate_init_value)

    def create_new_memories(
        self,
        keys: Tensor,
        values: Tensor,
        past_memories: Memories | None = None
    ) -> Memories:
        # Katharopoulos linear attention activation

        keys = F.elu(keys) + 1

        # create the next memories

        if exists(past_memories) and self.use_mem_delta_rule:
            delta_v = retrieve_from_kv_memories(keys, past_memories)

            # eq (5) - the delta rule
            values = values - delta_v

        new_memories_kv = einsum(keys, values, '... n dk, ... n dv -> ... dk dv')
        new_memories_norm = reduce(keys, 'b h n d -> b h d', 'sum')

        if exists(past_memories):
            past_memories_kv, past_memories_norm = past_memories

            new_memories_kv = new_memories_kv + past_memories_kv          # eq (4)
            new_memories_norm = new_memories_norm + past_memories_norm    # eq (4)

        return Memories(new_memories_kv, new_memories_norm)

    def retrieve_and_add_to_output(
        self,
        out: Tensor,
        queries: Tensor,
        past_memories: Memories | None = None
    ) -> Tensor:

        if not exists(past_memories):
            return out

        # the main contribution of the paper
        # Katharopoulos linear attention to kv memory of shape (batch, heads, dim keys, dim values)
        # it makes sense the author would try this, as he is ex-shmidhuber lab (linear transformers are fast weights paper)

        queries = F.elu(queries) + 1

        # retrieve from past memories

        mem_out = retrieve_from_kv_memories(queries, past_memories)

        # combine the current timestep output of queries with the outputs querying the past 'compressed' key/value memories
        # in paper, they use a sigmoid gating scheme with learned gate per head

        gates = rearrange(self.head_gates, 'h -> h 1 1')
        gates = gates.sigmoid()

        out = out * gates + mem_out * (1. - gates)  # eq (6) - figure 3 shows how heads emergently specialize to look either at the present, past, or a bit of both

        return out

# attention

class CausalAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 128,
        heads = 8,
        dropout = 0.,
        head_gate_init_value = 10.,
        use_mem_delta_rule = False
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.scale = dim_head ** -0.5
        self.norm = RMSNorm(dim)

        self.rotary_emb = RotaryEmbedding(dim_head)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.fastweight_mem = FastweightMemory(
            heads = heads,
            head_gate_init_value = head_gate_init_value,
            use_mem_delta_rule = use_mem_delta_rule
        )

    def forward(
        self,
        x,
        cached_kv: Tensor | None = None,
        past_memories: Memories | None = None,
        return_new_memories = False,
        eps = 1e-10
    ) -> Tuple[Tensor, Tensor, Memories]:
        """
        ein notation:

        b - batch
        h - heads
        n - sequence
        i - source sequence (q)
        j - target sequence (kv)
        d - feature dimension
        dk - feature dimension keys (and queries)
        dv - feature dimension of values
        """

        x = self.norm(x)

        x = self.to_qkv(x)
        q, k, v = self.split_heads(x)

        # handle cached key / values

        if exists(cached_kv):
            cached_k, cached_v = cached_kv
            k = torch.cat((cached_k, k), dim = -2)
            v = torch.cat((cached_v, v), dim = -2)

        # similarity

        q_scaled = q * self.scale
        q_rotated, k_rotated = self.rotary_emb.rotate_queries_with_cached_keys(q_scaled, k)

        sim = einsum(q_rotated, k_rotated, '... i d, ... j d -> ... i j')

        # causal mask

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = sim.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attend

        attn = sim.softmax(dim = -1)

        # dropout

        attn = self.dropout(attn)

        # aggregate values

        out = einsum(attn, v, '... i j, ... j d -> ... i d')

        out = self.fastweight_mem.retrieve_and_add_to_output(out, q, past_memories)

        # merge heads and combine

        out = self.merge_heads(out)
        out = self.to_out(out)

        # if new memories are not needed, early return
        # at inference time, kv cache up to segment length and then compress memories into kv

        if not return_new_memories:
            cached_kv = torch.stack((k, v))

            return out, cached_kv, past_memories

        new_memories = self.fastweight_mem.create_new_memories(k, v, past_memories)

        return out, None, new_memories

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
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        use_mem_delta_rule = False,     # in the paper, the delta rule didn't seem to do that much, but will include for completeness
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        for _ in range(depth):

            attn = CausalAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                use_mem_delta_rule = use_mem_delta_rule,
                dropout = attn_dropout
            )

            ff = FeedForward(
                dim = dim,
                mult = ff_mult,
                dropout = ff_dropout
            )

            self.layers.append(ModuleList([attn, ff]))

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(
        self,
        x,
        past_memories: List[Memories] | None = None,
        cached_kv: List[Tensor] | None = None,
        return_new_memories = False,
        detach_memories = False
    ) -> TransformerReturn:

        x = self.token_emb(x)

        # handle cached key values

        if exists(cached_kv):
            x = x[:, -1:]

        new_cached_kv = []
        cached_kv_iter = iter(default(cached_kv, []))

        # iterator for past compressed memories

        new_memories = []
        past_memories_iter = iter(default(past_memories, []))

        # going through layers of infini-transformer

        for attn, ff in self.layers:

            attn_out, layer_cached_kv, layer_new_memories = attn(
                x,
                cached_kv = next(cached_kv_iter, None),
                past_memories = next(past_memories_iter, None),
                return_new_memories = return_new_memories
            )

            x = attn_out + x
            x = ff(x) + x

            new_cached_kv.append(layer_cached_kv)
            new_memories.append(layer_new_memories)

        # final norm

        embed = self.norm(x)

        # logits

        logits = self.to_logits(embed)

        if detach_memories:
            detach_cached_kv_(new_cached_kv)

        if not return_new_memories:
            return TransformerReturn(logits, new_cached_kv, past_memories)

        if detach_memories:
            detach_memories_(new_memories)

        return TransformerReturn(logits, None, new_memories)
