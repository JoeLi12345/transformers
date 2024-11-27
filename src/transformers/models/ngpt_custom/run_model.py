from transformers import Ngpt_customModel, Ngpt_customConfig
import torch

model = Ngpt_customModel(Ngpt_customConfig(vocab_size=50, hidden_size=8, intermediate_size=8))
input_ids = [[4]]*8  # vector of input ids
input = torch.tensor(input_ids, dtype=torch.long)
original_output = model.forward(input)
print(original_output)