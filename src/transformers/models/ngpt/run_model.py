from transformers import NgptModel, NgptLMHeadModel, NgptConfig 
import torch

model = NgptLMHeadModel(NgptConfig(vocab_size=50, n_positions = 128, n_embd=24, n_layer=4, n_head=4))
input_ids = [[4]*2]*10  # vector of input ids
input = torch.tensor(input_ids, dtype=torch.long)
original_output = model.forward(input)