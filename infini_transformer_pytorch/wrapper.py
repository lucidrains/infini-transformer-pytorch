from __future__ import annotations
from typing import Callable

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einops import pack, rearrange
from tqdm import tqdm

from infini_transformer_pytorch.infini_transformer import (
    InfiniTransformer,
    detach_memories_,
    detach_cached_kv_
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True, eps = 1e-10):
    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, frac_num_tokens = 0.1, k: int | None = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# class

class InfiniTransformerWrapper(Module):
    def __init__(
        self,
        model: InfiniTransformer,
        segment_length = 512,
        ignore_index = -1
    ):
        super().__init__()
        self.model = model
        self.segment_length = segment_length

        # loss related
        self.ignore_index = ignore_index

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        *,
        seq_len,
        prompt = None,
        batch_size = 1,
        temperature = 1.,
        filter_fn: Callable = top_p,
        filter_kwargs: dict = dict(thres = 0.9),
        exclude_prompt = True,
        segment_length = None
    ):
        segment_length = default(segment_length, self.segment_length)
        device, train_state = self.device, self.training
        self.eval()

        out = default(prompt, torch.empty((batch_size, 0), device = device, dtype = torch.long))
        init_len = out.shape[-1]

        # sample from the model token by token
        # keeping track of kv cache and when to compress into new memories        

        cached_kv = None
        past_memories = None

        for curr_len in tqdm(range(init_len, seq_len)):

            # forward the model with cached key / values and past memories

            logits, cached_kv, past_memories = self.model(
                prompt,
                cached_kv = cached_kv,
                past_memories = past_memories,
                return_new_memories = divisible_by(curr_len, segment_length)
            )

            # grab the last logit

            logits = logits[:, -1]

            # filter by either topk or nucleus
            # and sample

            filtered_logits = filter_fn(logits, **filter_kwargs)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)

            # concat sampled token

            out, _ = pack((out, sampled), 'b *')

        # return output

        if exclude_prompt:
            out = out[:, init_len:]

        self.train(train_state)
        return out

    def forward(
        self,
        seq,
        segment_length = None
    ):
        segment_length = default(segment_length, self.segment_length)
        assert divisible_by(seq.shape[-1] - 1, segment_length)

        seq, label = seq[:, :-1], seq[:, 1:]

        # split the sequence by segment length

        split_seq = seq.split(segment_length, dim = -1)
        split_label = label.split(segment_length, dim = -1)

        # go over each segment length and calculate cross entropy loss

        losses = []
        past_memories = None

        for segment_seq, segment_label in zip(split_seq, split_label):

            logits, _, past_memories = self.model(
                segment_seq,
                past_memories = past_memories,
                return_new_memories = True
            )

            segment_losses = F.cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                segment_label,
                reduction = 'none',
                ignore_index = self.ignore_index,
            )

            losses.append(segment_losses)

        # combine all losses and average

        losses, _ = pack(losses, 'b *')
        loss = losses.mean()

        return loss
