import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ── Load pretrained LLM and tokenizer ─────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ── Transition network for scale dynamics ──────────────────────────────────────
class TransitionNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, s_prev, h_prev):
        x = torch.cat([s_prev, h_prev], dim=-1)
        return self.fc(x)

# ── Low-rank adapter & verifier ────────────────────────────────────────────────
class LowRankAdapter(nn.Module):
    def __init__(self, dim, rank=64):
        super().__init__()
        # factorized low-rank update
        self.U = nn.Linear(dim*2, rank, bias=False)
        self.V = nn.Linear(dim*2, rank, bias=False)
        self.W = nn.Linear(rank, dim)
    def forward(self, h, s):
        x = torch.cat([h, s], dim=-1)
        return self.W(torch.relu(self.U(x) * self.V(x)))

class Verifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 1)
        )
    def score(self, states):
        t = torch.from_numpy(states).float()
        with torch.no_grad():
            s = torch.sigmoid(self.fc(t)).squeeze(-1)
        return s.cpu().numpy()

# ── Spin-up training: 4D-Var style optimization ───────────────────────────────
def train_spinup(dataset, dim, lr=1e-3, steps=1000, Q_init=1.0, R_init=1.0, device='cpu'):
    """
    Train initial scale s0, transition network f_phi, and covariances Q, R.
    dataset: iterable of (prompt_ids, true_ids) batches (tensors on device)
    Saves spinup_s0.npy, spinup_Q.npy, spinup_R.npy, spinup_trans_net.pth
    """
    # learnable parameters
    s0 = nn.Parameter(torch.ones(dim, device=device))
    Q = nn.Parameter(torch.full((dim,), Q_init, device=device))
    R = nn.Parameter(torch.full((dim,), R_init, device=device))
    f_phi = TransitionNet(dim).to(device)
    optimizer = optim.Adam([s0, Q, R] + list(f_phi.parameters()), lr=lr)

    for step in range(steps):
        total_loss = 0.0
        count = 0
        for prompt_ids, true_ids in dataset:
            prompt_ids, true_ids = prompt_ids.to(device), true_ids.to(device)
            # teacher-forced forward
            out = model(input_ids=prompt_ids, labels=true_ids, output_hidden_states=True)
            hs = out.hidden_states[-1]  # [B, seq_len, D]
            B, T, D = hs.shape

            # flatten sequence
            h_seq = hs[:, :-1, :].reshape(-1, D)
            h_gt  = hs[:, 1:, :].reshape(-1, D)

            # build scale sequence
            s_seq = []
            s_prev = s0
            for t in range(B*(T-1)):
                s_seq.append(s_prev)
                s_prev = f_phi(s_prev.unsqueeze(0), h_seq[t].unsqueeze(0)).squeeze(0)
            s_seq = torch.stack(s_seq, dim=0)

            # compute losses
            x_scaled = h_seq * s_seq
            loss_state = ((x_scaled - h_gt)**2 / R.unsqueeze(0)).mean()
            # dynamics consistency
            s_pred = torch.stack([f_phi(s_seq[i:i+1], h_seq[i:i+1]).squeeze(0) for i in range(s_seq.size(0))], dim=0)
            loss_dyn = ((s_seq - s_pred)**2 / Q.unsqueeze(0)).mean()

            loss = loss_state + loss_dyn
            total_loss += loss.item()
            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 100 == 0:
            print(f"Spin-up step {step}/{steps}, avg loss {total_loss/count:.4f}")

    # save results
    np.save("spinup_s0.npy", s0.detach().cpu().numpy())
    np.save("spinup_Q.npy",  Q.detach().cpu().numpy())
    np.save("spinup_R.npy",  R.detach().cpu().numpy())
    torch.save(f_phi.state_dict(), "spinup_trans_net.pth")
    print("Spin-up complete. Files saved to current directory.")

# ── Load spin-up components ───────────────────────────────────────────────────
def load_spinup_components(prefix="spinup"):
    s0 = np.load(f"{prefix}_s0.npy")
    Q  = np.load(f"{prefix}_Q.npy")
    R  = np.load(f"{prefix}_R.npy")
    dim = s0.shape[0]
    f_phi = TransitionNet(dim)
    f_phi.load_state_dict(torch.load(f"{prefix}_trans_net.pth"))
    f_phi.eval()
    return s0, f_phi, Q, R

# ── Self-verification pseudo-observation ───────────────────────────────────────
def generate_candidates_from_scaled(h, s, K=5, max_len=20, input_ids=None, past=None):
    """Generate K token continuations and return their hidden-states."""
    states = []
    for _ in range(K):
        # greedy sampling from scaled hidden state
        logits = model.lm_head(torch.from_numpy(h * s).unsqueeze(0).float())
        next_id = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
        out = model(input_ids=input_ids, past_key_values=past, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].cpu().numpy()
        past = out.past_key_values
        states.append(h)
    return np.stack(states)

# ── Scale update combining self-verification & low-rank adapter ───────────────
def update_scale(h, s_prior, verifier, adapter, input_ids, past):
    # 1) pseudo-observation from candidates
    states = generate_candidates_from_scaled(h, s_prior, input_ids=input_ids, past=past)
    scores = verifier.score(states)            # [K]
    y_obs = (scores[:, None] * states).sum(axis=0)
    # 2) amortized low-rank update
    h_t = torch.from_numpy(h).unsqueeze(0).float()
    s_t = torch.from_numpy(s_prior).unsqueeze(0).float()
    delta = adapter(h_t, s_t).squeeze(0).detach().cpu().numpy()
    return s_prior + delta

# ── Online generation with adaptive embedding scaling ──────────────────────────
def generate_with_adaptive_scaling(prompt,
                                   spinup_prefix="spinup",
                                   reference_texts=None,
                                   max_steps=50):
    # load spin-up results
    s0, f_phi, Q, R = load_spinup_components(spinup_prefix)
    # instantiate verifier & adapter
    verifier = Verifier(s0.shape[0])
    adapter  = LowRankAdapter(s0.shape[0])

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    past = None
    s = s0.copy()
    generated = prompt

    for _ in range(max_steps):
        # forward pass
        out = model(input_ids=input_ids, past_key_values=past, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].cpu().numpy()
        past = out.past_key_values
        # predict next-scale
        s_prior = f_phi(torch.from_numpy(s).unsqueeze(0).float(),
                        torch.from_numpy(h).unsqueeze(0).float()).squeeze(0).detach().cpu().numpy()
        # update scale online
        s = update_scale(h, s_prior, verifier, adapter, input_ids.clone(), past)
        # decode next token
        logits = model.lm_head(torch.from_numpy(h * s).unsqueeze(0).float())
        next_id = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
        tok = tokenizer.decode(next_id)
        generated += tok
        if next_id == tokenizer.eos_token_id:
            break
    return generated


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Define prompt-response text pairs
examples = [
    ("The capital of France is", " Paris."),
    ("The tallest mountain is", " Mount Everest.")
]

def build_dataset(examples, max_length=20):
    dataset = []
    for prompt, response in examples:
        full_text = prompt + response
        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        targets = tokenizer(full_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        dataset.append((inputs["input_ids"].squeeze(0), targets["input_ids"].squeeze(0)))
    return dataset

# Build dataset
dataset = build_dataset(examples)

# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example dataset loader stub
    dataset = build_dataset(examples)  # list of (prompt_ids, true_ids) tensors
    train_spinup(dataset=[(p.unsqueeze(0), t.unsqueeze(0)) for p, t in dataset],
             dim=768,
             steps=200,
             lr=1e-4,
             device='cpu') #'cuda' if torch.cuda.is_available() else 'cpu')


    out = generate_with_adaptive_scaling(
        prompt="In a healthy team, communication is",
        spinup_prefix="spinup"
    )
    print("\nGenerated with adaptive scaling:\n", out)

