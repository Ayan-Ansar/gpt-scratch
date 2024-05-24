import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters 
batch_size = 32 # how many independent sequences we will process in parallel
block_size = 8 # what is the maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eval_iters =200 
#----------------------------------------------------------------

torch.manual_seed(1337)

with open('wizard_oz.txt','r', encoding='utf-8') as f:
    text = f.read()
    
# unique characters occuring in the text 
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping for strings to integers 
stoi = {c:idx for idx,c in enumerate(chars)}
itos = {idx:c for idx,c in enumerate(chars)}

encode = lambda text: [stoi[c] for c in text]
decode = lambda text: ''.join([itos[idx] for idx in text])

# train and test splits 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading 
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))    
    xb = torch.stack([data[i:i+block_size] for i in ix]) # B, T --- I am the
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix]) # B, T --> best pl
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb 

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()   
    return out 

# implementing Bigram Language model (pytorch)
class BigramLngModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # B, T, C 
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context 
        for _ in range(max_new_tokens):
            # get the predictions 
            logits, loss = self(idx)
            # focus only on the last time stampe
            logits = logits [:, -1, :]
            # apply softmax to get probabilities 
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLngModel(vocab_size)
m = model.to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets 
    if iter %  eval_interval == 0:
        loss = estimate_loss()
        print(f"step {iter}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")
    
    # sample a batch of training examples
    xb, yb = get_batch('train')
    
    # evaluate the loss 
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() 
    
# generate from the model 
context = torch.zeros((1,1), dtype=torch.long).to(device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

