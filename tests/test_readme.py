
import torch

from infini_transformer_pytorch import (
    InfiniTransformer,
    InfiniTransformerWrapper
)

def test_readme():
    transformer = InfiniTransformer(
        num_tokens = 256,
        dim = 64,
        depth = 1,
        dim_head = 128,
        heads = 8,
        use_mem_delta_rule = True
    )

    x = torch.randint(0, 256, (1, 1024))

    logits1, _, mem1 = transformer(x, return_new_memories = False)
    logits2, _, mem2 = transformer(x, past_memories = mem1, return_new_memories = False)
    logits3, _, mem3 = transformer(x, past_memories = mem2, return_new_memories = True)

def test_generate():
    # model and wrapper

    model = InfiniTransformer(
        num_tokens = 256,
        dim = 64,
        depth = 1,
        dim_head = 128,
        heads = 8,
        use_mem_delta_rule = True
    )

    wrapper = InfiniTransformerWrapper(
        model,
        segment_length = 32,
        detach_mems_every_num_segments = 2 # greater than 1 so the network can learn how to 'write' to the fast weight memories
    )

    # mock input

    seq = torch.randint(0, 256, (2, 128)) # can be arbitrarily long sequence

    # training

    wrapper(
        seq,
        backward = True # will automatically segment and accumulate gradients when it detaches the memories
    )

    # after much data...

    # calculating eval loss

    with torch.no_grad():
        wrapper.eval()
        wrapper(seq)

    # generating is as easy as

    wrapper.generate(seq_len = 128, prompt = seq[:, :1])
