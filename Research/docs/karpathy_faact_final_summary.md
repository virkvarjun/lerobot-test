# Karpathy FAACT Integration: Final Summary

## What was implemented

### 1. Optimizer param groups (modeling_act.py, configuration_act.py)

- **Before:** 2 groups (backbone, rest)
- **After:** 4 groups — rest (encoder/decoder), backbone, action_head, embeddings
- Added config: `optimizer_lr_action_head`, `optimizer_lr_embeddings` (optional overrides)
- Enables Karpathy-style per-group LR tuning

### 2. Karpathy-style scheduler (schedulers.py, configuration_act.py)

- New `KarpathyWarmdownSchedulerConfig`:
  - Absolute warmup steps
  - Warmdown phase (configurable ratio of total steps)
  - Nonzero final LR floor (`final_lr_frac * peak_lr`)
- ACT config: `use_karpathy_scheduler`, `scheduler_warmup_steps`, `scheduler_warmdown_ratio`, `scheduler_final_lr_frac`
- Default: off (`use_karpathy_scheduler=False`) for backward compatibility

### 3. Experiment infrastructure

- `scripts/train_faact_karpathy.sh` — smoke and RunPod run helpers
- `docs/experiments/faact_karpathy_opt_v1.md` — experiment template
- `docs/karpathy_faact_integration_plan.md` — full integration plan

## What was NOT implemented (deferred)

- **QK post-norm scaling** — Would require custom attention; lower priority
- **Cosine weight decay schedule** — Static WD kept; can add later
- **Init scale reduction** — Xavier init unchanged; conservative
- **RoPE / local attention** — ACT uses sinusoidal + full attention; not applicable

## What was tested

- Unit test `test_act_backbone_lr` updated for new param group structure
- Manual verification of param group counts and LR assignment
- Lint: no errors

## What helped (expected)

- Per-group LR: action head and embeddings can be tuned separately
- Warmdown + LR floor: avoids end-of-training collapse, common in Karpathy/nanochat
- Absolute warmup: more predictable than ratio-based

## What to try next (ablations)

1. **Scheduler only** — Enable `use_karpathy_scheduler=True`, keep optimizer groups as-is
2. **Action head LR** — Set `optimizer_lr_action_head=2e-5` or `5e-6` (higher/lower than base)
3. **Embeddings LR** — Set `optimizer_lr_embeddings` for VAE/encoder projections
4. **Warmdown ratio** — Try 0.4, 0.5, 0.65 (nanochat uses 0.65)

## Run commands

**Smoke (500 steps):**
```bash
cd Research && bash scripts/train_faact_karpathy.sh smoke
```

**Full RunPod:**
```bash
cd /workspace/Research && MUJOCO_GL=egl bash scripts/train_faact_karpathy.sh runpod
```

## Files modified

| File | Change |
|------|--------|
| `lerobot/.../modeling_act.py` | `get_optim_params()` refactor |
| `lerobot/.../configuration_act.py` | New optimizer/scheduler config fields |
| `lerobot/.../schedulers.py` | `KarpathyWarmdownSchedulerConfig` |
| `lerobot/.../test_policies.py` | `test_act_backbone_lr` update |

## Commits (in Research/lerobot)

1. `refactor: isolate optimizer param groups for ACT (Karpathy-style)`
2. `feat: add Karpathy-style warmdown scheduler (absolute warmup, LR floor)`
