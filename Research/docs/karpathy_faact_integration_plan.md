# Karpathy nanochat/autoresearch → FAACT Integration Plan

**Goal:** Integrate the most promising ideas from Karpathy's nanochat/autoresearch tuning into FAACT (Failure-Aware ACT) for improved training stability, sample efficiency, and downstream failure prediction accuracy.

**Philosophy:** Do not blindly port LLM-specific changes. Only keep changes that make sense for ACT's architecture and imitation learning dynamics.

---

## 1. Current FAACT Stack Summary

### 1.1 Model Architecture (lerobot ACT)

| Component | Location | Notes |
|-----------|----------|-------|
| ACT policy | `lerobot/src/lerobot/policies/act/modeling_act.py` | `ACTPolicy`, `ACT` |
| Encoder | `ACTEncoder` + `ACTEncoderLayer` | Self-attention, pre/post-norm |
| Decoder | `ACTDecoder` + `ACTDecoderLayer` | Self + cross-attention |
| Action head | `self.action_head = nn.Linear(dim_model, action_dim)` | Single linear layer |
| VAE encoder | Optional, `ACTEncoder(is_vae_encoder=True)` | For latent sampling |
| Positional embeddings | Sinusoidal (1D + 2D), **no RoPE** | `create_sinusoidal_pos_embedding`, `ACTSinusoidalPositionEmbedding2d` |
| Attention | `nn.MultiheadAttention` | **Full attention**, no sliding window |
| Init | Xavier uniform for encoder/decoder | `_reset_parameters()` |
| Backbone | ResNet via torchvision | Layer4 features |

**Key findings:**
- No RoPE (uses sinusoidal)
- No local/sliding attention (full context)
- No QK normalization
- No value embedding gates or logit softcap

### 1.2 Optimizer & Scheduler

| Component | Location | Current |
|-----------|----------|---------|
| Optimizer creation | `lerobot/optim/factory.py` | `make_optimizer_and_scheduler()` |
| Policy param groups | `modeling_act.py::get_optim_params()` | 2 groups: backbone vs rest |
| ACT optimizer preset | `configuration_act.py::get_optimizer_preset()` | AdamW, lr=1e-5, wd=1e-4 |
| ACT scheduler preset | `configuration_act.py::get_scheduler_preset()` | **Returns None** (no scheduler) |
| AdamW config | `lerobot/optim/optimizers.py` | Single lr, wd, betas for all groups |
| Cosine + warmup | `lerobot/optim/schedulers.py` | `CosineDecayWithWarmupSchedulerConfig` exists but ACT doesn't use it |

### 1.3 Failure Predictor

| Component | Location | Notes |
|-----------|----------|-------|
| MLP | `failure_prediction/models/failure_predictor.py` | input_dim → [256,128] → 1 |
| Training | `failure_prediction/scripts/train_failure_predictor.py` | Adam, single lr=1e-3, no scheduler |

### 1.4 Training Loop

- **Entry:** `lerobot-train` → `lerobot/src/lerobot/scripts/lerobot_train.py`
- **Flow:** `update_policy()` → forward, backward, gradient clip, optimizer.step(), scheduler.step()
- **Config:** draccus + `TrainPipelineConfig`, `use_policy_training_preset=True` loads optimizer from policy

---

## 2. Karpathy Ideas: Transferability Assessment

### 2.1 Directly Transferable (high confidence)

| Idea | Nanochat | FAACT adaptation | ROI |
|------|----------|------------------|-----|
| **Per-parameter-group LR** | embedding, unembedding, matrix, scalars | backbone, encoder/decoder, action_head, embeddings, norm/bias | High |
| **Stronger output-head LR** | lm_head at lower LR (scaled by √(dim/768)) | Action head: try slightly lower or higher LR; commonly output heads benefit from different scaling | High |
| **Absolute warmup steps** | `--warmup-steps 40` | Add `warmup_steps` (int) instead of ratio-only; ACT often trains 100k steps | Medium |
| **Warmdown with nonzero floor** | `--warmdown-ratio 0.65`, `--final-lr-frac 0.05` | Cosine/warmdown decay with final_lr_frac so LR doesn't go to zero | High |
| **Cosine weight decay schedule** | `0.5 * (1 + cos(π * step / steps))` | Add optional cosine WD schedule (currently static) | Medium |

### 2.2 Maybe Transferable (needs ablation)

| Idea | Nanochat | FAACT adaptation | Ablation priority |
|------|----------|------------------|-------------------|
| **QK post-norm scaling** | Q,K normalized, optional scaler | nn.MultiheadAttention doesn't expose Q/K; would need custom attention or post-scale of attn output | 2nd (after opt/scheduler) |
| **Smaller init scales** | embeddings, MLP, output head | Xavier → smaller gain; try 0.9x or init action head smaller | 3rd |
| **Betas tuning** | beta1=0.96 for some groups | ACT uses default (0.9, 0.999); try (0.9, 0.99) or group-specific | 4th |

### 2.3 Not Applicable to FAACT

| Idea | Reason |
|------|--------|
| RoPE theta increase | ACT uses sinusoidal embeddings, not RoPE |
| Local/sliding attention | ACT uses full attention; window pattern is LLM-specific |
| Logit softcap | ACT outputs actions (regression), not logits |
| Value embedding gates | No analogous structure in ACT |
| Muon optimizer | Keep AdamW; Muon is for LLM scaling |

---

## 3. Implementation Plan

### Phase 1: Repo Understanding ✅

- [x] Trace model, optimizer, scheduler, training loop
- [x] Identify param groups, attention, init
- [x] Document current state

### Phase 2: Optimizer / Scheduler Adaptations

**Files to edit:**
1. `lerobot/src/lerobot/policies/act/modeling_act.py` — extend `get_optim_params()` for more granular groups
2. `lerobot/src/lerobot/policies/act/configuration_act.py` — add optimizer/scheduler options
3. `lerobot/src/lerobot/optim/schedulers.py` — add `KarpathyStyleSchedulerConfig` (absolute warmup, warmdown, final LR floor)
4. `lerobot/src/lerobot/optim/optimizers.py` — extend `AdamWConfig` for per-group lr/wd/betas if needed

**Changes:**
- Add param groups: backbone, encoder_decoder, action_head, embeddings (VAE/encoder proj), norm_bias (no decay)
- Add `optimizer_lr_action_head`, `optimizer_lr_embeddings`, `optimizer_weight_decay_norm_bias` (0 for norm/bias)
- Add scheduler: `warmup_steps`, `warmdown_ratio`, `final_lr_frac`
- Optional: cosine weight decay schedule

**Commit strategy:**
- Commit 1: Refactor `get_optim_params` into explicit groups (backbone, transformer, action_head, embeddings)
- Commit 2: Add config fields for per-group LR
- Commit 3: Add `KarpathyStyleSchedulerConfig`
- Commit 4: Wire ACT config to use new scheduler when enabled

### Phase 3: Attention / Architecture Adaptations

**Only if Phase 2 is stable:**

- **QK post-norm scaling:** Would require custom `MultiheadAttention` or a wrapper. Lower priority; document as future work or add behind a flag with minimal patch.
- **Init scales:** Add `init_scale` multiplier in `_reset_parameters()` for action head (default 1.0, ablation 0.5–0.9).

### Phase 4: Configs and Ablations

**New configs:**
- `faact_karpathy_opt_v1` — optimizer/scheduler only
- `faact_karpathy_optattn_v1` — opt + optional QK scaling
- `faact_karpathy_fullsafe_v1` — all safe changes (opt, scheduler, init)

### Phase 5: Evaluation

**Metrics:**
- Training loss (L1, KL if VAE)
- Validation loss
- Rollout success rate
- Failure predictor AUROC / F1 (on ACT embeddings)
- Gradient norms, NaNs, dead heads

---

## 4. Ablation Order (Safest → Riskiest)

1. **Scheduler only** — Add warmup + warmdown + final LR floor; keep optimizer as-is.
2. **Param groups** — Add action_head, embeddings groups with configurable LR; keep scheduler.
3. **Cosine weight decay** — Optional cosine WD schedule.
4. **Init scale** — Smaller init for action head.
5. **QK post-norm** — Custom attention path (highest risk, most invasive).

---

## 5. Files Reference

| Purpose | Path |
|---------|------|
| ACT model | `lerobot/src/lerobot/policies/act/modeling_act.py` |
| ACT config | `lerobot/src/lerobot/policies/act/configuration_act.py` |
| Optimizer factory | `lerobot/src/lerobot/optim/factory.py` |
| Optimizers | `lerobot/src/lerobot/optim/optimizers.py` |
| Schedulers | `lerobot/src/lerobot/optim/schedulers.py` |
| Train config | `lerobot/src/lerobot/configs/train.py` |
| Training loop | `lerobot/src/lerobot/scripts/lerobot_train.py` |
| Failure predictor | `failure_prediction/models/failure_predictor.py` |

---

## 6. RunPod / Local Commands

**Smoke test (local):**
```bash
lerobot-train --policy.type=act --policy.path=... --dataset.repo_id=... --output_dir=outputs/train/act_smoke --steps=100
```

**RunPod ablation:**
```bash
# TBD: script under scripts/ for karpathy_faact_* experiments
```

---

## 7. Success Criteria

- Training converges without NaNs or divergence
- Validation loss ≤ or better than baseline
- Rollout success rate ≥ baseline
- Failure predictor AUROC stable or improved
- Each change is reversible via config
