from __future__ import annotations
from math import ceil
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import pack, rearrange
from tqdm import tqdm

from infini_transformer_pytorch.infini_transformer import (
    InfiniTransformer,
    detach_memories_
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_empty(t: Tensor):
    return t.numel() == 0

def round_down_multiple(n, mult):
    return n // mult * mult

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
        detach_mems_every_num_segments = 2,
        ignore_index = -1
    ):
        super().__init__()
        self.model = model

        self.segment_length = segment_length
        self.detach_mems_every_num_segments = detach_mems_every_num_segments

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

            # what is fed into the model is always at the start of the very last segment

            start_ind = round_down_multiple(curr_len - 1, segment_length)
            model_input = out[:, start_ind:]

            # forward the model with cached key / values and past memories

            logits, cached_kv, past_memories = self.model(
                model_input,
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
        segment_length = None,
        backward = False,
        grad_accum_scale = 1.
    ):
        segment_length = default(segment_length, self.segment_length)

        seq, label = seq[:, :-1], seq[:, 1:]

        # put into train mode if doing backwards within forward call

        if backward:
            self.model.train()

        total_tokens = (label != self.ignore_index).sum().item()

        # split the sequence by segment length

        split_seq = seq.split(segment_length, dim = -1)
        split_label = label.split(segment_length, dim = -1)

        num_segments = len(split_seq)

        # go over each segment length and calculate cross entropy loss

        total_loss = 0.
        past_memories = None

        running_loss = 0.

        for ind, (segment_seq, segment_label) in enumerate(zip(split_seq, split_label)):
            segment_num = ind + 1
            is_last = segment_num == num_segments

            should_detach_memories = divisible_by(segment_num, self.detach_mems_every_num_segments)
            should_backward = backward and (is_last or should_detach_memories)

            # model forwards for logits and past memories

            logits, _, past_memories = self.model(
                segment_seq,
                past_memories = past_memories,
                return_new_memories = True
            )

            # calculate cross entropy loss for segment

            segment_loss = F.cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                segment_label,
                reduction = 'none'
            )

            # make sure segment losses do not include ignored index
            # then also make sure the segment loss is scaled

            segment_mask = segment_label != self.ignore_index
            num_segment_tokens = segment_mask.sum()
            frac_tokens = num_segment_tokens / total_tokens

            segment_loss = segment_loss[segment_mask]
            segment_scaled_loss = segment_loss.mean() * frac_tokens

            total_loss = total_loss + segment_scaled_loss
            running_loss = running_loss + segment_scaled_loss

            # perform backwards every `(num_segment * detach_mems_every_num_segments)`

            if should_backward:
                (running_loss / grad_accum_scale).backward()
                running_loss = 0.

            # detach memories if need be

            if should_detach_memories and not is_last:
                detach_memories_(past_memories)

        return total_loss
