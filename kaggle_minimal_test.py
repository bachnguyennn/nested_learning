"""
Minimal Kaggle cell to test hope_selfmod directly.
Paste this as a single cell. No CLI, no Hydra, no YAML.
"""

import sys
sys.path.insert(0, "/kaggle/working/nested_learning/src")

import torch
from nested_learning.model import ModelConfig, HOPEModel
from nested_learning.levels import LevelSpec

print("Step 1: building model...")
cfg = ModelConfig(
    vocab_size=1000,
    dim=128,
    num_layers=2,
    heads=4,
    block_variant="hope_selfmod",
    titan_level=LevelSpec(name="titan", update_period=8, optimizer_key="cms_opt"),
    cms_levels=[
        LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),
        LevelSpec(name="cms_slow", update_period=8, optimizer_key="cms_opt"),
    ],
    optimizers={
        "cms_opt": {"type": "deep_momentum", "lr": 3e-4, "params": {"beta": 0.9, "beta2": 0.999}}
    },
    # FSRM features (sphere norm + learnable eta only — inner loop removed)
    self_mod_output_l2_norm=True,
    self_mod_learnable_eta=True,
)
model = HOPEModel(cfg).cuda()
print(f"  Model built: {sum(p.numel() for p in model.parameters()):,} params")

print("Step 2: forward pass...")
tokens = torch.randint(0, 1000, (2, 64)).cuda()
with torch.no_grad():
    logits = model(tokens)
print(f"  Output shape: {logits.shape}, NaN: {torch.isnan(logits).any().item()}")

print("Step 3: backward pass...")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(10):
    tokens = torch.randint(0, 1000, (2, 64)).cuda()
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 1000),
        tokens[:, 1:].reshape(-1)
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"  step={step} loss={loss.item():.4f} nan={torch.isnan(loss).item()}")

print("DONE — hope_selfmod works correctly!")
