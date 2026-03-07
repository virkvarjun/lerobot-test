# ACT Training Pipeline for SO-101

A minimal, reproducible pipeline for training **ACT (Action Chunking with Transformers)** policies on a real SO-101 robot using [Hugging Face LeRobot](https://github.com/huggingface/lerobot).

No custom model code — this project wraps LeRobot's official CLI tools (`lerobot-record`, `lerobot-train`, etc.) with simple shell scripts and a single config file.

---

## Repository structure

```
act_training_pipeline/
├── configs/
│   ├── robot.env.example     # Template — copy to robot.env and fill in
│   └── robot.env             # Your config (git-ignored)
├── scripts/
│   ├── 01_check_env.py       # Verify imports, GPU, lerobot version
│   ├── 02_record_dataset.sh  # Record episodes from SO-101
│   ├── 03_train_act.sh       # Train ACT on recorded dataset
│   └── 04_eval_act.sh        # Run trained policy on real robot
├── data/                     # Recorded datasets (git-ignored)
├── outputs/                  # Training checkpoints & logs (git-ignored)
├── requirements.txt          # Pinned dependencies
├── .gitignore
└── README.md
```

---

## From zero to trained policy

### Step 0: Install

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# If you haven't already, install Feetech motor SDK
pip install feetech-servo-sdk
```

> **Note on Python 3.14:** LeRobot 0.3.2 uses `draccus` which has a known incompatibility with Python 3.14's `argparse`. If you hit `TypeError: str | None is not callable`, either use Python 3.12/3.13, or apply the patch described in the Troubleshooting section below.

### Step 1: Configure

```bash
cp configs/robot.env.example configs/robot.env
```

Edit `configs/robot.env` with your values:

| Variable | What to fill in |
|----------|----------------|
| `ROBOT_PORT_FOLLOWER` | Follower arm USB port (run `lerobot-find-port`) |
| `ROBOT_PORT_LEADER` | Leader arm USB port |
| `ROBOT_ID` / `TELEOP_ID` | Calibration IDs (match filenames in `~/.cache/huggingface/lerobot/calibration/`) |
| `HF_USER` | Your Hugging Face username |
| `DATASET_NAME` | Name for your dataset (e.g. `so101_pick_cube`) |
| `DATASET_TASK` | Natural language task description |
| `DEVICE` | `cuda` (Linux GPU), `mps` (Apple Silicon), or `cpu` |

### Step 2: Verify environment

```bash
python scripts/01_check_env.py
```

This prints Python/PyTorch/LeRobot versions, GPU availability, and detected USB ports.

### Step 3: Record a dataset

Connect both arms, then:

```bash
# Record locally (no upload)
bash scripts/02_record_dataset.sh

# Record and push to Hugging Face Hub
bash scripts/02_record_dataset.sh --push
```

The script will:
- Connect to the leader (teleop) and follower (robot) arms
- Prompt you through calibration if needed (press ENTER to reuse existing calibration)
- Record `DATASET_NUM_EPISODES` episodes, each `DATASET_EPISODE_TIME_S` seconds
- Save to `data/<DATASET_NAME>/`

### Step 4: Train ACT

```bash
bash scripts/03_train_act.sh
```

This trains an ACT policy using LeRobot's built-in training loop. Key defaults:
- **100k steps**, batch size 8, chunk size 100
- Checkpoints saved every 20k steps to `outputs/train/act_<DATASET_NAME>/`
- Override any ACT hyperparameter by uncommenting variables in `configs/robot.env`

To resume from a checkpoint:
```bash
bash scripts/03_train_act.sh --resume
```

### Step 5: Evaluate on the real robot

```bash
# Use the latest checkpoint
bash scripts/04_eval_act.sh

# Use a specific checkpoint
bash scripts/04_eval_act.sh outputs/train/act_so101_pick_cube/checkpoints/040000/pretrained_model
```

This runs the trained policy autonomously on the real robot and records the evaluation episodes to `data/<DATASET_NAME>_eval/`.

---

## Config reference

All configuration lives in `configs/robot.env`. The scripts read from this single file — no need to edit scripts directly.

### ACT hyperparameters (optional overrides)

Uncomment these in `robot.env` to override ACT defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 100 | Action chunk length |
| `DIM_MODEL` | 512 | Transformer hidden dimension |
| `N_HEADS` | 8 | Attention heads |
| `N_ENCODER_LAYERS` | 4 | Encoder layers |
| `N_DECODER_LAYERS` | 1 | Decoder layers |
| `KL_WEIGHT` | 10.0 | VAE KL divergence weight |
| `LEARNING_RATE` | 1e-5 | Optimizer learning rate |

---

## Training on a remote GPU machine

1. Record your dataset locally (macOS)
2. Push to HF Hub: `bash scripts/02_record_dataset.sh --push`
3. On the GPU machine, clone this repo and set `DEVICE=cuda` in `robot.env`
4. The training script will pull the dataset from the Hub automatically when `--dataset.root` points to a directory that doesn't exist yet

Alternatively, `rsync` the `data/` folder to the remote machine and train with the local path.

---

## Troubleshooting

### `TypeError: str | None is not callable` (Python 3.14)

This is a `draccus` + Python 3.14 incompatibility. Patch the installed `draccus`:

```bash
# Find the file
FIELD_WRAPPER=$(python -c "import draccus; print(draccus.__file__.replace('__init__.py', 'wrappers/field_wrapper.py'))")

# The fix: after `tpe = utils.canonicalize_union(tpe)`, add logic to extract the
# non-None type from Optional unions. See the project's draccus patch notes.
```

Or use Python 3.12/3.13 to avoid the issue entirely.

### Calibration prompts during connect

If the robot prompts about calibration mismatch, press **ENTER** to reuse existing calibration. Only type `c` if you need to recalibrate.

### `ModuleNotFoundError: No module named 'scservo_sdk'`

```bash
pip install feetech-servo-sdk
```

---

## References

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot Docs](https://huggingface.co/docs/lerobot)
- [ACT Paper](https://arxiv.org/abs/2304.13705) — Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
- [SO-101 Assembly Guide](https://huggingface.co/docs/lerobot/so101)
