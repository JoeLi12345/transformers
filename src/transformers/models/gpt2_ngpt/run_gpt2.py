from transformers import GPT2Model, GPT2Config
import torch
import torch.nn.functional as F

model = GPT2Model(GPT2Config(vocab_size=50, n_positions=50, n_embd=8, n_head=4))
input_ids = [[4]]*8  # vector of input ids
input = torch.tensor(input_ids, dtype=torch.long)
original_output = model.forward(input)
#print("Hello")
#print(original_output)
'''
vector = torch.tensor([3.0, 4.0])

# Normalize the vector
normalized_vector = F.normalize(vector, p=2, dim=0)

print("Normalized vector:", normalized_vector)'''