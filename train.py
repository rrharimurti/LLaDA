import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
from generate import generate

# -------------------------------
# 1. Load BeaverTails dataset
# -------------------------------
ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")

benign_prompts = [ex["prompt"] for ex in ds if ex["is_safe"] == True]
jailbreak_prompts = [ex["prompt"] for ex in ds if ex["is_safe"] == False]

print(f"Training on {len(benign_prompts)} benign prompts")
print(f"Testing on {len(jailbreak_prompts)} jailbreak prompts")

# -------------------------------
# 2. Load LM for hidden states
# -------------------------------
device = "cuda"

model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

def encode_batch(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    input_ids = torch.tensor(inputs["input_ids"]).to(device)
    with torch.no_grad():
        outputs = generate(model, input_ids, gen_length=0, block_length=0)
    hidden_states = outputs.hidden_states[len(outputs.hidden_states) // 2]  # Middle Layer
    return hidden_states.mean(dim=1).cpu()

# Precompute embeddings with chat template applied to each prompt
benign_embeds = torch.cat([
    encode_batch([
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        for prompt in benign_prompts[i:i+32]
    ])
    for i in range(0, len(benign_prompts), 32)
])
jailbreak_embeds = torch.cat([
    encode_batch([
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        for prompt in jailbreak_prompts[i:i+32]
    ])
    for i in range(0, len(jailbreak_prompts), 32)
])

# -------------------------------
# 3. Define Sparse Autoencoder
# -------------------------------
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

# -------------------------------
# 4. Training
# -------------------------------
input_dim = benign_embeds.shape[1]
hidden_dim = 512

sae = SparseAutoencoder(input_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def sparsity_loss(z, lam=1e-3):
    return lam * torch.mean(torch.abs(z))

train_loader = DataLoader(TensorDataset(benign_embeds), batch_size=32, shuffle=True)

for epoch in range(5):
    sae.traitn()
    for (batch,) in train_loader:
        batch = batch.to(device)
        x_hat, z = sae(batch)
        loss = criterion(x_hat, batch) + sparsity_loss(z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# -------------------------------
# 5. Evaluation with ROC–AUC
# -------------------------------
sae.eval()

def reconstruction_error(x):
    with torch.no_grad():
        x_hat, _ = sae(x.to(device))
        return torch.mean((x_hat - x.to(device))**2, dim=1).cpu()

benign_errors = reconstruction_error(benign_embeds)
jailbreak_errors = reconstruction_error(jailbreak_embeds)

print(f"Avg benign error:   {benign_errors.mean():.4f}")
print(f"Avg jailbreak error:{jailbreak_errors.mean():.4f}")

# Labels: 0 = benign, 1 = jailbreak
labels = [0] * len(benign_errors) + [1] * len(jailbreak_errors)
scores = torch.cat([benign_errors, jailbreak_errors]).numpy()

roc_auc = roc_auc_score(labels, scores)
print(f"ROC–AUC: {roc_auc:.4f}")
