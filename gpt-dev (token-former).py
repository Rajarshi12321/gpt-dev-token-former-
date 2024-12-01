import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
# max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('pokemon.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f'vocab size: {vocab_size}')
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # print("X shape:", x.shape)
    # print("Y shape:", y.shape)
    x, y = x.to(device), y.to(device)
    return x, y



class pattention(nn.Module):
    def __init__(self, input_dim, output_dim, intermediate_dim=128, scale_factor=None, device=None):
        """
        Initializes the attention mechanism with key and value projections using nn.Linear layers.
        """
        super().__init__()  # Initialize nn.Module
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        # Set the device
        # self.device = device or torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize key and value projection layers as nn.Linear
        self.KP = nn.Linear(input_dim, intermediate_dim, bias=False).to(self.device)  # Key projection layer
        self.VP = nn.Linear(intermediate_dim, output_dim, bias=False).to(self.device)  # Value projection layer

        # Optional scale factor (defaults to output_dim if not provided)
        self.scale_factor = scale_factor or output_dim

    def forward(self, X):
        # Ensure input is on the same device as KP and VP
        X = X.to(self.device)

        # Compute the attention scores (dot product between input and key projections)
        A = self.KP(X)  # A will have shape (B, T, intermediate_dim)
        # print(f"KP Shape: {A.shape}")


        # Normalize and scale the attention scores
        norm_A_sq = torch.norm(A, p=2, dim=-1, keepdim=True) ** 2
        S = (A * self.scale_factor) / (norm_A_sq + 1e-6)
        S = F.gelu(S)

        # print(f"S Shape: {S.shape}")


        # Compute the output by applying the value projection to the scaled attention
        O = self.VP(S)  # O will have shape (B, T, output_dim)

        # print(f"VP Shape: {O.shape}\n--------------------\n")

        return O

    def __call__(self, X):
        return self.forward(X)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = pattention(head_size, head_size)
        self.query = pattention(head_size, head_size)
        self.value = pattention(head_size, head_size)
        # self.register_buffer('tril', torch.tril(torch.ones(64, 64)))  # Assuming block_size=64
        # self.register_buffer('tril', torch.tril(torch.ones(T, T)))  # Assuming block_size=head_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        # print(f"Head Input Shape: {x.shape}")


        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        # print(f"Key Shape: {k.shape}, Query Shape: {q.shape}")


        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # Dynamically create the mask
        tril = torch.tril(torch.ones(T, T, device=x.device))  # Ensure tril is on the same device as x
        wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf')).to(self.device)  # Ensure wei is on the correct device
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # print(f"Attention Weights Shape: {wei.shape}")

        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        # print(f"Head Output Shape: {out.shape}")

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)
        self.proj = pattention(num_heads * head_size, num_heads * head_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape

        # print(f"MultiHeadAttention Input Shape: {x.shape}")


        # Split the input tensor into num_heads chunks of size head_size
        # (B, T, C) -> (B, T, num_heads, head_size)
        x_split = x.view(B, T, self.num_heads, self.head_size)
        x_split = x_split.permute(2, 0, 1, 3)  # (num_heads, B, T, head_size)

        # Pass each chunk into its corresponding head
        out = torch.cat([head(x_split[i]) for i, head in enumerate(self.heads)], dim=-1)  # Concatenate along the last dimension

        # Projection and dropout after concatenation
        out = self.proj(out)  # (B, T, num_heads * head_size)
        out = self.dropout(out)

        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Print the input shape before passing through the feedforward network
        # print(f"Input shape to FeedForward: {x.shape}")
        # print(f"FeedForward Input Shape: {x.shape}")
        x = self.net(x)
        # print(f"FeedForward Output Shape: {x.shape}")
        return x

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        # print((n_head, head_size),"let's see")
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
         # Print the input shape before passing through the block
        # print(f"Input shape to Block: {x.shape}")

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        # Print the output shape after block processing
        # print(f"Output shape from Block: {x.shape}")

        return x

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # print("X shape:", X.shape)  # Check the shape of input batch
            # print("Y shape:", Y.shape)  # Check the shape of target batch
            logits, loss = model(X, Y)

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self,n_embd):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        # print(f"Shape after embedding: {x.shape}")  # (B,T,C)

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# ----------------------Extending pattention layer Key and Value Matrices---------------------------
import torch
import torch.nn as nn

def extend_linear_layer_for_KP(layer, new_out_features):
    """
    Extends the given linear layer to have new_out_features while preserving existing weights.

    Args:
        layer (nn.Linear): The original linear layer.
        new_out_features (int): The desired number of output features.

    Returns:
        nn.Linear: The extended linear layer.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    old_out_features, in_features = layer.weight.shape
    print(f"old layer KP: {layer.weight.shape}")

    if new_out_features <= old_out_features:
        raise ValueError("new_out_features must be greater than the current out_features.")

    in_features = layer.in_features
    old_out_features = layer.out_features

    # Create a new linear layer with the new output features
    new_layer = nn.Linear(in_features, new_out_features, bias=False).to(device)

    print(f"new layer KP: {new_layer.weight.shape}")
    # print(f"new layer VP: {new_layer.weight.shape}")

    # Preserve the original weights by copying them over
    with torch.no_grad():
        new_layer.weight.data[:old_out_features, :] = layer.weight.data

        # Initialize the new rows (for the added output features) with random values or zeros
        new_layer.weight.data[old_out_features:, :] = torch.randn(
            new_out_features - old_out_features, in_features).to(device)


    return new_layer



def extend_linear_layer_for_VP(layer, new_in_features):
    """
    Extends the given linear layer to have new input features while preserving existing weights.

    Args:
        layer (nn.Linear): The original linear layer.
        new_in_features (int): The desired number of input features.

    Returns:
        nn.Linear: The extended linear layer.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    old_in_features, old_out_features = layer.weight.shape
    # print(layer.weight.shape,old_in_features, old_out_features,new_in_features )
    # print(old_out_features <= new_in_features,old_out_features,new_in_features,"HII")
    if new_in_features <= old_out_features:
        raise ValueError("new_in_features must be greater than the current in_features.")

    # Create a new linear layer with the new input features and the same output features
    new_layer = nn.Linear(new_in_features, old_in_features, bias=False).to(device)

    print(f"old layer VP: {layer.weight.shape}")
    print(f"New layer VP: {new_layer.weight.shape}")

    # print(f"New layer KP: {new_layer.weight.shape}")
    # print(f"OLD layer KP: {layer.weight.shape}")

    # Preserve the original weights by copying them over to the new layer
    with torch.no_grad():
        # Copy the old weight values into the new layer

        new_layer.weight.data[:, :old_out_features] = layer.weight.data

        # Optionally initialize the weights for the new input features (for the added columns)
        # Initialize the new columns (for the additional input features) with zeros or random values
        new_layer.weight.data[:, old_out_features:] = torch.randn(
            old_in_features, new_in_features - old_out_features).to(device)


    return new_layer

def extend_attention_dimensions(model, new_features_dim):
    """
    Extends key, query, value layers in the model to new_out_features while preserving weights.
    """
    new_in_features = new_out_features = new_features_dim
    for block in model.blocks:
        for head in block.sa.heads:

            # print(head.key.KP.weight.shape)
            # print(head.key.VP.weight.shape,"VPP")
            head.key.KP = extend_linear_layer_for_KP(head.key.KP, new_out_features)

            head.key.VP = extend_linear_layer_for_VP(head.key.VP, new_in_features)
            # return

            head.query.KP = extend_linear_layer_for_KP(head.query.KP, new_out_features)
            head.query.VP = extend_linear_layer_for_VP(head.query.VP, new_in_features)

            head.value.KP = extend_linear_layer_for_KP(head.value.KP, new_out_features)
            head.value.VP = extend_linear_layer_for_VP(head.value.VP, new_in_features)

        # Update proj layer to match the concatenated output of the new heads
        num_heads = len(block.sa.heads)
        block.sa.proj.KP = extend_linear_layer_for_KP(block.sa.proj.KP, num_heads * new_out_features)
        block.sa.proj.VP = extend_linear_layer_for_VP(block.sa.proj.VP, num_heads * new_out_features)

    return model


def extend_model(model, new_out_features):
    """
    Extend the entire model: token embeddings, position embeddings, attention layers,
    feedforward layers, and the final output layer (lm_head and ln_f) to the new output size,
    while preserving the pre-trained weights.
    """
  
    # Extend the multi-head attention layers
    model = extend_attention_dimensions(model, new_out_features)


    return model



# -------------------------------------------------

model = BigramLanguageModel(n_embd = 4)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Using max iterations as {max_iters}")

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')


    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))


#---------------------------Trained on expanded model with increased parameters----------------------------------

# Example usage:
new_out_features = 256  # Desired new output size for the model
# new_out_features = 10  # Desired new output size for the model
model = extend_model(model, new_out_features)

print(f"Using max iterations as {max_iters}")

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')


    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))



