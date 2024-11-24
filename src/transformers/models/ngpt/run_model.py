'''from transformers import LlamaModel, LlamaConfig
import torch

model = LlamaModel(LlamaConfig(vocab_size=50, hidden_size=8, intermediate_size=8))
input_ids = [[4]]*8  # vector of input ids
input = torch.tensor(input_ids, dtype=torch.long)
original_output = model.forward(input)
print(original_output)'''

from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]