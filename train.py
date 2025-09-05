import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
from generate import generate
"""
# OLD CODE
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

model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16, output_hidden_states=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

def encode_batch(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    input_ids = torch.tensor(inputs["input_ids"]).to(device)
    with torch.no_grad():
        outputs = []
        hidden_states = []
        for ids in input_ids:
            out, hidden_state = generate(model, ids.unsqueeze(0), gen_length=52, block_length=26)
            outputs.append(out)
            hidden_states.append(hidden_state)
        outputs = torch.cat(outputs, dim=0)
        hidden_states = torch.cat(hidden_states, dim=0)
    return hidden_states.mean(dim=1).cpu()

# Precompute embeddings with chat template applied to each prompt
benign_embeds_list = []
for i in range(0, len(benign_prompts), 32):
    batch_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        for prompt in benign_prompts[i:i+32]
    ]
    embeds = encode_batch(batch_prompts)
    benign_embeds_list.append(embeds)
    print(f"{i}/{len(benign_prompts)} benign prompts done")
benign_embeds = torch.cat(benign_embeds_list)

jailbreak_embeds_list = []
for i in range(0, len(jailbreak_prompts), 32):
    batch_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        for prompt in jailbreak_prompts[i:i+32]
    ]
    embeds = encode_batch(batch_prompts)
    jailbreak_embeds_list.append(embeds)
    print(f"{i}/{len(jailbreak_prompts)} jailbreak prompts done")
jailbreak_embeds = torch.cat(jailbreak_embeds_list)
"""

# -------------------------------
# 1. Load BeaverTails dataset
# -------------------------------
ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")

benign_prompts = [ex["prompt"] for ex in ds if ex["is_safe"]]
jailbreak_prompts = [ex["prompt"] for ex in ds if not ex["is_safe"]]

print(f"Training on {len(benign_prompts)} benign prompts")
print(f"Testing on {len(jailbreak_prompts)} jailbreak prompts")

# -------------------------------
# 2. Load LM
# -------------------------------
device = "cuda"

model = AutoModel.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    output_hidden_states=True,
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct",
    trust_remote_code=True
)

# -------------------------------
# 3. Torch Dataset for batching
# -------------------------------
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer):
        # Pre-tokenize once, apply chat template
        self.examples = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False
            )
            for p in prompts
        ]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    return tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )


# -------------------------------
# 4. Encode batches (no generate!)
# -------------------------------
def encode_dataset(prompts, batch_size=32):
    dataset = PromptDataset(prompts, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    all_embeds = []

    for i, inputs in enumerate(loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states is a tuple: (layer, batch, seq, dim)
            hidden_states = torch.stack(outputs.hidden_states)  # [n_layers, batch, seq, dim]
            # Treat each token embedding as one training sample
            embeds = outputs.hidden_states[-2].reshape(-1, hidden_dim)  # [(batch*seq), dim]
            all_embeds.append(embeds.cpu())
        if i % 10 == 0:
            print(f"{i * batch_size}/{len(prompts)} prompts done")

    return torch.cat(all_embeds, dim=0)


# -------------------------------
# 5. Run for benign & jailbreak
# -------------------------------
print("Encoding benign prompts...")
benign_embeds = encode_dataset(benign_prompts)

print("Encoding jailbreak prompts...")
jailbreak_embeds = encode_dataset(jailbreak_prompts)

print("Shapes:", benign_embeds.shape, jailbreak_embeds.shape)

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
    sae.train()
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
