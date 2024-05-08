from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einops import pack, rearrange

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

    @torch.no_grad()
    def generate(
        self,
        *,
        seq_len,
        prompt = None,
        segment_length = None
    ):
        segment_length = default(segment_length, self.segment_length)
        train_state = self.training
        self.eval()

        raise NotImplementedError

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
