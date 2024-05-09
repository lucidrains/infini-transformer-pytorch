from infini_transformer_pytorch import (
    InfiniTransformer,
    InfiniTransformerWrapper
)

import tqdm
import gzip
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 250
PRIME_LEN = 100
SEQ_LEN = 1024
SEGMENT_LENGTH = 128

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

model = InfiniTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    dim_head = 64,
    heads = 8,
    use_mem_delta_rule = True
)

wrapper = InfiniTransformerWrapper(
    model,
    segment_length = SEGMENT_LENGTH,
    detach_mems_every_num_segments = 2
).cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    x = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    train_x, valid_x = np.split(x, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (train_x, valid_x))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = 1))

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.):

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = wrapper(
            next(train_loader),
            backward = True,
            grad_accum_scale = GRADIENT_ACCUMULATE_EVERY ** -1.
        )        

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        with torch.no_grad():
            wrapper.eval()
            loss = wrapper(next(val_loader))
            print(f'validation loss: {loss.item()}')

    if i % GENERATE_EVERY == 0:
        ids = next(val_loader)[:, :PRIME_LEN]
        prime = decode_tokens(ids.flatten())
        print('%s \n\n %s', (prime, '*' * 100))

        sample = wrapper.generate(
            prompt = ids,
            seq_len = SEQ_LEN
        )

        decoded_string = decode_tokens(sample.flatten())
        print(decoded_string)
        print("\n")
