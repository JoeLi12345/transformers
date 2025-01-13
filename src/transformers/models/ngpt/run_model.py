from transformers import NgptModel, NgptLMHeadModel, NgptConfig 
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader
import math
import inspect
import os
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

# get the shard filenames
        data_root = "openwebtextB"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        #if master_process:
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
# advance the position in the tensor
        self.current_position += B * T * self.num_processes
# if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, x

class TrainDataset(Dataset):
    def __init__(self, T):
        self.dataset = load_dataset("stas/openwebtext-10k")['train']
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.tokens = []
        print(len(self.dataset))
        for i in range(len(self.dataset)):
            self.message = self.dataset[i]["text"]
            token_append = self.tokenizer(self.message)["input_ids"]
            remainder = len(token_append) % T
            if remainder != 0:
                token_append = token_append[:-remainder]
            self.tokens.extend(token_append)

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        self.T = T
        print(f"loaded {self.tokens.size(0)} tokens")

    def __len__(self):
        return self.tokens.size(0) // self.T


    def __getitem__(self, idx):
        return self.tokens[idx*self.T:(idx+1)*self.T], self.tokens[idx*self.T:(idx+1)*self.T]

max_steps = 2000

def get_lr(it, max_lr=6e-4, min_lr=6e-5, warmup_steps=10):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def configure_optimizers(model, weight_decay, learning_rate, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"

    print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

#message = "Hello, I'm a language model,"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"using device: {device}")

torch.manual_seed(1337)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(1337)

total_batch_size = 65536
B, T = 64, 1024
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B*T)
print(f"total desired batch size: {total_batch_size}")
print(f"gradient accumulation steps: {grad_accum_steps}")


torch.set_float32_matmul_precision('high')
#train_data = TrainDataset(T=T)
#train_dataloader = DataLoader(train_data, batch_size=B)

train_dataloader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="train")

gpt2_model = GPT2LMHeadModel(GPT2Config(vocab_size=50304))
gpt2_model.to(device)

gpt2_model = torch.compile(gpt2_model)

optimizer = configure_optimizers(gpt2_model, weight_decay=0.1, learning_rate=6e-4, device_type=device)

#train_iterator = iter(train_dataloader)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        '''sample_batch = next(train_iterator)
        sample_batch[0] = sample_batch[0].to(device)
        sample_batch[1] = sample_batch[1].to(device)'''
        x, y = train_dataloader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            #ret = gpt2_model(input_ids=sample_batch[0],labels=sample_batch[1])
            ret = gpt2_model(input_ids=x, labels=y)
            logits,loss = ret["logits"], ret["loss"]

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(gpt2_model.parameters(), 1.0)
    
    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000
    tokens_per_sec = B*T*grad_accum_steps/(t1-t0)
    print(f"step {step}, loss: {loss_accum.item()}, time: {dt:.2f}ms, norm: {norm:.4f}, lr: {lr}, tok/sec: {tokens_per_sec:.4f}")

import sys
sys.exit(0)

for i in range(50):
    optimizer.zero_grad()
    ret = gpt2_model(input_ids=buf,labels=buf)
    logits,loss = ret["logits"], ret["loss"]
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")

import sys
sys.exit(0)

max_length = 32

while tokens.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = gpt2_model(tokens)["logits"]
        # take the logits at the last position
        logits = logits[:,-1,:] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        tokens = torch.cat((tokens, xcol), dim=1)


for i in range(5):
    string = tokens[i,:max_length].tolist()
    string = tokenizer.decode(string)
    print(string)


'''
ngpt_model = NgptLMHeadModel(NgptConfig(vocab_size=50, n_positions = 128, n_embd=24, n_layer=4, n_head=4))
input_ids = [[4]*2]*10  # vector of input ids
input = torch.tensor(input_ids, dtype=torch.long)
original_output = ngpt_model(input)'''
