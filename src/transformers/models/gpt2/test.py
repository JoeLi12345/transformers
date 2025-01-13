import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from torchtune.modules import RotaryPositionalEmbeddings

def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
# Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
# Apply the rotary embeddings to the query and key
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)
    return q_rot, k_rot

def get_sinusoidal_embeddings( n_positions, dim):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim))
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb


B, T, x, y = 1, 2, 2, 2

key = torch.rand(B, T, x, y)
query = torch.rand(B, T, x, y)
key = torch.ones(B, T, x, y)
query = torch.ones(B, T, x, y)

'''
sinusoidal_pos = get_sinusoidal_embeddings(T, y)
q_rot, k_rot = apply_rotary_position_embeddings(sinusoidal_pos, query.transpose(1, 2), key.transpose(1, 2))
q_rot = q_rot.transpose(2, 1)
k_rot = k_rot.transpose(2, 1)'''
rope = RotaryPositionalEmbeddings(dim=y, max_seq_len=T)

q_rot2, k_rot2 = rope(query), rope(key)

print("QUERY")
print(q_rot2)
print("KEY")
print(k_rot2)

print(key, query)

