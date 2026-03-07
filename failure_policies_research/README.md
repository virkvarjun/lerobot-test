# FIPER-Repro (Runtime Failure Prediction for Chunked Robot Policies)

This repo is an implementation + reproduction of the core ideas from **FIPER: Failure Prediction at Runtime for Generative Robot Policies** (Römer et al., 2025), adapted to a practical robot-learning stack (e.g., ACT-style / chunked policies and real robot execution). The goal is to build a **runtime monitor** that detects **out-of-distribution (OOD)** situations and **imminent failures** while a robot policy is executing action chunks — and to provide a clean baseline that can later be extended into more novel "failure memory + recovery" systems.

## What this project implements (baseline)
This project recreates the FIPER-style runtime failure detection pipeline:

1. **Embedding-space OOD detection**
   - Compute an embedding from the policy (or an encoder aligned with the policy).
   - Train **Random Network Distillation (RND)** on in-distribution rollouts.
   - Use RND prediction error as an **OOD score** at runtime.

2. **Action chunk uncertainty**
   - For chunked/generative policies, compute an uncertainty signal over the **predicted action chunk**.
   - In the original FIPER framing, this is "action-chunk entropy"; in practice this may be:
     - distribution entropy (if the policy exposes it),
     - ensemble variance,
     - reconstruction error (for CVAE-style ACT),
     - or sampling disagreement (for diffusion/flow sampling).

3. **Failure alarm logic**
   - Combine signals (OOD score + uncertainty) with smoothing/windowing.
   - Trigger an alarm when risk is persistently high.

4. **Calibration (optional but recommended)**
   - Threshold calibration using lightweight statistical methods (e.g., conformal-style calibration on successful rollouts) so the alarm is not purely hand-tuned.

> Note: This repo focuses on the **runtime detector baseline**. Recovery/repair policies and failure-mode memory are intentionally out of scope for the first milestone.

---

## Why this matters
Chunked policies are powerful but can be brittle under distribution shift. Runtime failure detection provides:
- safer execution (stop/replan before catastrophic failure),
- better debugging (see risk spike aligned with sensor/policy drift),
- a baseline for more advanced work (failure-mode embeddings, retrieval, recovery skills).

---

## Project milestones

### Milestone 1 (this repo)
✅ Implement FIPER-style runtime failure detection:
- RND OOD score in embedding space
- Chunk uncertainty signal
- Alarm logic + logging + evaluation scripts

### Milestone 2 (future work)
- Failure-mode embedding space (structured clustering of failure types)
- Retrieval-conditioned recovery / chunk repair

---

## Repository structure

```
failure_policies_research/
├── configs/                          # Configuration files
├── data/                             # Datasets and annotations
│   ├── failure_annotations.json      # Failure annotation labels
│   ├── labeled_dataset.parquet       # Labeled training data
│   ├── split_info.json               # Train/val/test splits
│   └── windowed_dataset.npz          # Windowed features for classifier
├── experiments/                      # Experiment logs and outputs
├── models/                           # Trained model checkpoints
│   ├── best_model.pt                 # Best failure-gater checkpoint
│   └── results.json                  # Evaluation results
├── notebooks/                        # Jupyter notebooks for exploration
├── plots/                            # Generated visualizations
│   ├── roc_curve.png                 # ROC curve from evaluation
│   └── training_history.png          # Training loss/accuracy curves
├── scripts/                          # Executable pipeline scripts
│   ├── annotate_failures.py          # Label episodes as success/failure
│   ├── create_windowed_dataset.py    # Build windowed feature dataset
│   ├── debug_cameras.py              # Camera debugging utility
│   ├── gated_eval.py                 # Gated autonomous evaluation
│   ├── gated_teleop.py               # Gated teleoperation with failure detection
│   ├── generate_labels.py            # Generate training labels
│   └── train_classifier.py           # Train the failure classifier
├── src/                              # Source library code
│   ├── data/                         # Data loading and processing
│   ├── models/                       # Model architectures
│   │   └── failure_gater.py          # Failure gater network
│   └── utils/                        # Utility functions
├── tests/                            # Unit tests
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## License

MIT
