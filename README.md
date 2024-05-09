<img src="./infini-attention.png" width="300px"></img>

## Infini-Transformer - Pytorch

Implementation of <a href="https://arxiv.org/abs/2404.07143">Infini-Transformer</a> in Pytorch. They use a linear attention scheme to compress past memories and demonstrate multiple SOTAs for long context benchmarks.

Although unlikely to beat <a href="https://github.com/lucidrains/ring-attention-pytorch">Ring Attention</a>, I think it is worth exploring, as the techniques are orthogonal.

<a href="https://www.youtube.com/watch?v=r_UBBfTPcF0">Yannic Kilcher's explanation</a>

## Install

```bash
$ pip install infini-transformer-pytorch
```

## Usage

```python
import torch
from infini_transformer_pytorch import InfiniTransformer

transformer = InfiniTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    dim_head = 128,  # high head dimension may be part of the reason they got good results (kv has high capacity)
    heads = 8,
    use_mem_delta_rule = True
)

x = torch.randint(0, 256, (1, 1024))

logits1, _, mem1 = transformer(x, return_new_memories = False)
logits2, _, mem2 = transformer(x, past_memories = mem1, return_new_memories = False)
logits3, _, mem3 = transformer(x, past_memories = mem2, return_new_memories = True)

```

Training a transformer with recurrence usually trips up a lot of researchers, so to make it easy, just wrap it with `InfiniTransformerWrapper`

```python
import torch

from infini_transformer_pytorch import (
    InfiniTransformer,
    InfiniTransformerWrapper
)

# model and wrapper

model = InfiniTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    dim_head = 128,
    heads = 8,
    use_mem_delta_rule = True
)

wrapper = InfiniTransformerWrapper(
    model,
    segment_length = 512,
    detach_mems_every_num_segments = 2 # greater than 1 so the network can learn how to 'write' to the fast weight memories
).cuda()

# mock input

seq = torch.randint(0, 256, (2, 10000)).cuda() # can be arbitrarily long sequence

# training

loss = wrapper(
    seq,
    backward = True # will automatically segment and accumulate gradients when it detaches the memories
)

# after much data...

# calculating eval loss

with torch.no_grad():
    wrapper.eval()
    eval_loss = wrapper(seq)

# generating is as easy as

output = wrapper.generate(seq_len = 8192, prompt = seq[:, :1])

output.shape # (2, 8192 - 1)
```

## Testing

Train an autoregressive enwik8

```bash
$ python train.py
```

## Todo

- [ ] `detach_mems_every_num_segments` hyperparameter is too confusing, get rid of it
- [ ] experiment with enhanced recurrence, perhaps with a linear projection (talking heads on kv or linear projection on k, v separately) before sending the memories to the layer before
- [x] working example with enwik8

## Citations

```bibtex
@inproceedings{Munkhdalai2024LeaveNC,
    title   = {Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention},
    author  = {Tsendsuren Munkhdalai and Manaal Faruqui and Siddharth Gopal},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:269033427}
}
```
