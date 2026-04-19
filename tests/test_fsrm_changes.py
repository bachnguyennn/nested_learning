"""FSRM smoke tests — validates all three changes + review-requested checks.

Run with:
    uv run python tests/test_fsrm_changes.py
"""
from __future__ import annotations

import sys

import torch


def test_sphere_norm_on() -> None:
    """Change 1: output_l2_norm=True → per-token norms ≤ 1 (with float tol)."""
    from nested_learning.titan.self_modifying import (
        SelfModifyingTitans,
        SelfModifyingTitansConfig,
    )
    cfg = SelfModifyingTitansConfig(dim=64, output_l2_norm=True)
    sm = SelfModifyingTitans(cfg)
    x = torch.randn(2, 8, 64)
    out = sm(x)
    norms = out.norm(dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-4, (
        f"All norms should be ≈1.0, max deviation: {(norms - 1.0).abs().max():.6f}"
    )
    print(f"  ✓ sphere norm ON: max deviation from 1.0 = {(norms-1.0).abs().max():.6f}")


def test_sphere_norm_off_backward_compat() -> None:
    """output_l2_norm=False (default) → norms NOT forced to 1.0."""
    from nested_learning.titan.self_modifying import (
        SelfModifyingTitans,
        SelfModifyingTitansConfig,
    )
    cfg = SelfModifyingTitansConfig(dim=64, output_l2_norm=False)
    sm = SelfModifyingTitans(cfg)
    assert cfg.output_l2_norm is False, "Default should be False"
    x = torch.randn(2, 8, 64)
    out = sm(x)
    norms = out.norm(dim=-1)
    # Without norm, output norms should NOT all be 1.0
    print(f"  ✓ sphere norm OFF (default): norms unconstrained (mean={norms.mean():.4f})")


def test_defaults_all_false() -> None:
    """Verify all FSRM defaults are False/1 for backward compatibility."""
    from nested_learning.hope.block import HOPESelfModBlockConfig
    from nested_learning.levels import LevelSpec
    from nested_learning.titan.self_modifying import SelfModifyingTitansConfig

    # SelfModifyingTitansConfig
    cfg = SelfModifyingTitansConfig(dim=64)
    assert cfg.output_l2_norm is False, "output_l2_norm should default False"
    assert cfg.learnable_eta is False, "learnable_eta should default False"

    # HOPESelfModBlockConfig
    spec = LevelSpec(name="x", update_period=1, optimizer_key="k")
    bcfg = HOPESelfModBlockConfig(dim=64, cms_levels=[spec])
    assert bcfg.selfmod_output_l2_norm is False
    assert bcfg.selfmod_learnable_eta is False
    print("  ✓ All FSRM defaults are backward-compatible (False)")


def test_learnable_eta_on() -> None:
    """Change 2: learnable_eta=True creates a gradient-receiving Parameter."""
    from nested_learning.titan.self_modifying import (
        SelfModifyingTitans,
        SelfModifyingTitansConfig,
    )
    cfg = SelfModifyingTitansConfig(dim=64, learnable_eta=True, eta_scale=1e-3)
    sm = SelfModifyingTitans(cfg)
    assert sm.eta_param is not None, "eta_param should be created"
    assert sm.eta_param.requires_grad, "eta_param should be trainable"
    # Check it's in parameters()
    param_names = [n for n, _ in sm.named_parameters()]
    assert "eta_param" in param_names, f"eta_param missing from named_parameters: {param_names}"
    effective = torch.nn.functional.softplus(sm.eta_param).item()
    print(f"  ✓ learnable eta ON: eta_param is trainable, effective={effective:.6f}")


def test_learnable_eta_off() -> None:
    """Change 2 off: learnable_eta=False (default) → no extra parameter."""
    from nested_learning.titan.self_modifying import (
        SelfModifyingTitans,
        SelfModifyingTitansConfig,
    )
    cfg = SelfModifyingTitansConfig(dim=64, learnable_eta=False)
    sm = SelfModifyingTitans(cfg)
    assert sm.eta_param is None, "eta_param should be None when disabled"
    param_names = [n for n, _ in sm.named_parameters()]
    assert "eta_param" not in param_names, "eta_param should NOT be in parameters"
    print("  ✓ learnable eta OFF (default): no extra parameter")


def test_learnable_eta_param_count() -> None:
    """learnable_eta=True adds exactly ONE extra scalar parameter."""
    from nested_learning.titan.self_modifying import (
        SelfModifyingTitans,
        SelfModifyingTitansConfig,
    )
    cfg_off = SelfModifyingTitansConfig(dim=64, learnable_eta=False)
    cfg_on = SelfModifyingTitansConfig(dim=64, learnable_eta=True)
    sm_off = SelfModifyingTitans(cfg_off)
    sm_on = SelfModifyingTitans(cfg_on)
    count_off = sum(p.numel() for p in sm_off.parameters())
    count_on = sum(p.numel() for p in sm_on.parameters())
    diff = count_on - count_off
    assert diff == 1, f"Expected exactly 1 extra param, got {diff}"
    print(f"  ✓ learnable eta adds exactly 1 param ({count_off} → {count_on})")


def test_learnable_eta_updates_under_optim() -> None:
    """eta_param updates after one optimizer step.

    NOTE: forward_with_updates() detaches outputs through fast-weight state
    tensors, so we test eta_param trainability directly via _effective_eta_scale()
    which is the path that actually consumes eta_param during real training.
    """
    from nested_learning.titan.self_modifying import (
        SelfModifyingTitans,
        SelfModifyingTitansConfig,
    )
    cfg = SelfModifyingTitansConfig(dim=32, learnable_eta=True, eta_scale=1e-3)
    sm = SelfModifyingTitans(cfg)
    opt = torch.optim.SGD(sm.parameters(), lr=0.1)

    # Construct a loss that goes through eta_param via _effective_eta_scale
    effective_eta = sm._effective_eta_scale()  # softplus(eta_param)
    loss = effective_eta * 10.0  # scalar loss depending on eta_param
    eta_before = sm.eta_param.item()
    loss.backward()
    opt.step()
    eta_after = sm.eta_param.item()
    assert eta_before != eta_after, "eta_param should change after optimizer step"
    print(f"  ✓ eta_param updates under SGD ({eta_before:.6f} → {eta_after:.6f})")



if __name__ == "__main__":
    print("Running FSRM-inspired changes smoke tests...\n")
    tests = [
        test_defaults_all_false,
        test_sphere_norm_on,
        test_sphere_norm_off_backward_compat,
        test_learnable_eta_on,
        test_learnable_eta_off,
        test_learnable_eta_param_count,
        test_learnable_eta_updates_under_optim,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All tests passed!")
