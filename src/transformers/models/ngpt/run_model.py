from transformers import NgptModel, NgptLMHeadModel, NgptConfig 
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, T=32):
        self.dataset = load_dataset("karpathy/tiny_shakespeare")['train']
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.message = self.dataset[0]["text"]
        #print(self.message)
        self.tokens = self.tokenizer(self.message)["input_ids"]
        self.tokens = torch.tensor(self.tokens,dtype=torch.long)
        self.T = T
        print(f"loaded {self.tokens.size(0)} tokens")

    def __len__(self):
        return self.tokens.size(0) // self.T


    def __getitem__(self, idx):
        return self.tokens[idx*self.T:(idx+1)*self.T], self.tokens[idx*self.T:(idx+1)*self.T]


#message = "Hello, I'm a language model,"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"using device: {device}")

torch.manual_seed(1337)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(1337)

train_data = TrainDataset()
train_dataloader = DataLoader(train_data, batch_size=4)

gpt2_model = GPT2LMHeadModel(GPT2Config())
gpt2_model.to(device)
#gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")

#print(gpt2_model(input_ids=buf,labels=buf)["loss"])
optimizer = torch.optim.AdamW(gpt2_model.parameters(), lr=3e-4)

cnt = 0
for i_batch, sample_batch in enumerate(train_dataloader):
    optimizer.zero_grad()
    sample_batch[0], sample_batch[1] = sample_batch[0].to(device), sample_batch[1].to(device)
    ret = gpt2_model(input_ids=sample_batch[0],labels=sample_batch[1])
    logits,loss = ret["logits"], ret["loss"]
    loss.backward()
    optimizer.step()
    print(f"step {i_batch}, loss: {loss.item()}")

    if (cnt > 50):
        break
    cnt += 1

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