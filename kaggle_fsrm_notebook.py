"""
Kaggle Notebook: FSRM-Inspired HOPE Ablation
=============================================

Paste this entire script into a Kaggle notebook cell (GPU runtime).
It will:
  1. Clone the repo & install dependencies
  2. Run the 15-test smoke suite
  3. Run baseline (all FSRM off) for 200 steps
  4. Run FSRM ablation (sphere norm + learnable eta + T=2) for 200 steps
  5. Compare loss curves

Setup: Kaggle → New Notebook → GPU T4 or P100 → paste & run.
"""

# ── Cell 1: Install ──────────────────────────────────────────────────────────

# !pip install -q uv
# !cd /kaggle/working && git clone https://github.com/<YOUR_USER>/nested_learning.git
# !cd /kaggle/working/nested_learning && uv sync --frozen

# If the repo is uploaded as a Kaggle dataset instead of cloned:
# !pip install -q uv
# !cd /kaggle/input/nested-learning && uv sync --frozen

import subprocess, sys, os

# ── Adjust this path to wherever your repo ends up ──
REPO_DIR = "/kaggle/working/nested_learning"

# If you uploaded the repo as a dataset, copy it to a writable location first:
# subprocess.run(["cp", "-r", "/kaggle/input/nested-learning", REPO_DIR], check=True)

os.chdir(REPO_DIR)

# Install the package
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# ── Cell 2: Smoke Tests ──────────────────────────────────────────────────────

print("=" * 60)
print("Running FSRM smoke tests...")
print("=" * 60)

result = subprocess.run(
    [sys.executable, "tests/test_fsrm_changes.py"],
    capture_output=True, text=True, cwd=REPO_DIR
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    raise RuntimeError("Smoke tests failed!")
print("✓ All smoke tests passed\n")

# ── Cell 3: Run Baseline (FSRM OFF) ──────────────────────────────────────────

print("=" * 60)
print("Run 1: Baseline (all FSRM features OFF)")
print("=" * 60)

# Use the existing baseline config — no FSRM features
subprocess.run([
    sys.executable, "-m", "nested_learning.cli", "train",
    "--config-name", "kaggle_baseline_test",
], check=True, cwd=REPO_DIR)

# ── Cell 4: Run FSRM Ablation (sphere norm + learnable eta + T=2) ────────────

print("=" * 60)
print("Run 2: FSRM Ablation (sphere norm + learnable eta + T=2)")
print("=" * 60)

subprocess.run([
    sys.executable, "-m", "nested_learning.cli", "train",
    "--config-name", "kaggle_fsrm_ablation",
], check=True, cwd=REPO_DIR)

# ── Cell 5: Compare Loss Curves ──────────────────────────────────────────────

import json
import matplotlib.pyplot as plt

def load_log(path):
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "loss" in entry and "step" in entry:
                    steps.append(entry["step"])
                    losses.append(entry["loss"])
            except json.JSONDecodeError:
                continue
    return steps, losses

baseline_log = "/kaggle/working/logs/baseline_hope.json"
fsrm_log = "/kaggle/working/logs/fsrm_ablation.json"

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

if os.path.exists(baseline_log):
    s, l = load_log(baseline_log)
    ax.plot(s, l, label="Baseline (FSRM OFF)", alpha=0.8)

if os.path.exists(fsrm_log):
    s, l = load_log(fsrm_log)
    ax.plot(s, l, label="FSRM (sphere + eta + T=2)", alpha=0.8)

ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("HOPE: Baseline vs FSRM-Inspired Changes")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/kaggle/working/fsrm_comparison.png", dpi=150)
plt.show()
print("✓ Comparison plot saved to /kaggle/working/fsrm_comparison.png")

# ── Cell 6: Quick NaN Check ──────────────────────────────────────────────────

import math

def check_nan(path, name):
    if not os.path.exists(path):
        print(f"  ⚠ {name}: log not found at {path}")
        return
    steps, losses = load_log(path)
    nan_count = sum(1 for l in losses if math.isnan(l) or math.isinf(l))
    final = losses[-1] if losses else float("nan")
    print(f"  {name}: {len(losses)} steps, {nan_count} NaN/Inf, final_loss={final:.4f}")

print("\nNaN/Inf Summary:")
check_nan(baseline_log, "Baseline")
check_nan(fsrm_log, "FSRM Ablation")
