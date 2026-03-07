#!/usr/bin/env python3
"""
Gated Autonomous Evaluation Script

Runs a trained control policy (like ACT) but interrupts with a 
safety freeze if the failure prediction model triggers.
"""

import argparse
import logging
import time
import json
from pathlib import Path
import numpy as np
import torch
from lerobot.configs import parser
from lerobot.robots import make_robot_from_config
from lerobot.policies.factory import make_policy
try:
    from lerobot.utils.control_utils import init_keyboard_listener
except (ImportError, RuntimeError):
    # Fallback for headless/no-perm environments
    def init_keyboard_listener():
        return None, {"stop_recording": False}

from lerobot.utils.robot_utils import precise_sleep
from lerobot.scripts.lerobot_record import RecordConfig

# Import our gater
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.models.failure_gater import FailureGater

import builtins
builtins.input = lambda _: ""

@parser.wrap()
def main(cfg: RecordConfig):
    # 1. Setup Robot
    print(f"DEBUG: robot cameras config: {cfg.robot.cameras}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    # 2. Load Control Policy (ACT)
    # We assume the policy path is provided via command line as --policy.pretrained_path
    if not cfg.policy.pretrained_path:
        # Fallback to the one we are currently training if not specified
        cfg.policy.pretrained_path = Path("/Users/arjunvirk/Desktop/Projects/lerobot/lerobot/outputs/train/act_failure_success/checkpoints/last/pretrained_model")
        print(f"Using default policy path: {cfg.policy.pretrained_path}")

    # For safety, let's check if the path exists
    if not Path(cfg.policy.pretrained_path).exists():
        print(f"❌ Policy path not found: {cfg.policy.pretrained_path}")
        print("Please provide it via --policy.pretrained_path <PATH>")
        sys.exit(1)

    print(f"Loading policy from {cfg.policy.pretrained_path}...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    
    # Load dataset to get stats and meta
    dataset = LeRobotDataset(cfg.dataset.repo_id)
    stats = dataset.meta.stats
    
    # Force ImageNet stats if training used them
    if "observation.images.webcam" in stats:
        print("ℹ️  Overriding webcam stats with ImageNet mean/std")
        # ImageNet mean/std for [0, 1] scaled images
        stats["observation.images.webcam"]["mean"] = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        stats["observation.images.webcam"]["std"] = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    print(f"Loading policy from {cfg.policy.pretrained_path}...")
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    policy.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    policy.to(device)

    # Load pre- and post-processors for normalization
    input_proc, output_proc = make_pre_post_processors(
        cfg.policy, 
        dataset_stats=stats,
        pretrained_path=cfg.policy.pretrained_path
    )
    
    # 3. Load Safety Gater (Classifier)
    results_path = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/models/results.json")
    threshold = 0.29
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
            threshold = data.get("optimal_threshold", 0.29)
            if threshold < 0.001:
                threshold = data.get("recall_at_10pct_fpr_threshold", 0.29)
            print(f"Loaded safety threshold: {threshold:.3f}")

    model_path = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/models/best_model.pt")
    gater = FailureGater(model_path=model_path, threshold=threshold)
    
    # 4. Initialize Loop
    listener, events = init_keyboard_listener()
    fps = cfg.dataset.fps
    freeze_until = 0
    is_frozen = False
    
    joint_keys = [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", 
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
    ]
    
    print("\n🚀 GATED AUTONOMOUS EVALUATION RUNNING")
    print("Policy and Safety Guard active. Press ESC or Q to quit.")
    
    try:
        while not events["stop_recording"]:
            start_loop_t = time.perf_counter()
            
            # A. Get Current State
            obs = robot.get_observation()
            if 'debug_printed' not in locals():
                 print(f"DEBUG: raw robot obs keys: {list(obs.keys())}")
            
            # Prepare state/image dict for processor
            proc_obs = {}
            for k, v in obs.items():
                if k in joint_keys:
                    continue
                else:
                    policy_key = f"observation.images.{k}"
                    img = v
                    if not torch.is_tensor(img):
                        img = torch.from_numpy(img)
                    
                    # Robot observation is [H, W, C], policy expects [C, H, W]
                    if img.ndim == 3:
                        img = img.permute(2, 0, 1)
                    
                    # Convert to float and scale to [0, 1] if uint8
                    if img.dtype == torch.uint8:
                        img = img.float() / 255.0
                    elif img.dtype != torch.float32:
                        img = img.float()
                        
                    proc_obs[policy_key] = img
            
            # Explicitly create state vector in correct order
            state_list = [float(obs[k]) for k in joint_keys]
            state_np = np.array(state_list)
            proc_obs["observation.state"] = torch.from_numpy(state_np).float()
            
            # Apply Pre-processor (Normalization and Scaling)
            # This handles [0, 255] -> [0, 1] for images and MEAN_STD for state
            policy_obs = input_proc(proc_obs)
            
            # Diagnostic: What are the keys?
            if 'debug_printed' not in locals():
                print(f"DEBUG: proc_obs keys: {list(proc_obs.keys())}")
                print(f"DEBUG: policy_obs keys: {list(policy_obs.keys())}")
                debug_printed = True

            # Ensure tensors are on the correct device
            policy_obs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in policy_obs.items()}

            # B. Get Policy Action
            with torch.no_grad():
                # select_action uses policy_obs and returns [action_dim]
                raw_action = policy.select_action(policy_obs)
            
            # Post-process the action (Unnormalization)
            # raw_action shape is (1, 6). output_proc returns (1, 6) tensor.
            processed_action = output_proc(raw_action)[0]
            
            # Extract values for gating and robot command
            action_np = processed_action.cpu().numpy().flatten()
            
            # C. Safety Gating Logic
            is_unsafe, prob = gater.update(state_np, action_np, return_prob=True)
            
            # Map back to dict for robot.send_action
            action_dict = {k: float(v) for k, v in zip(joint_keys, action_np)}
            
            # Diagnostic: Are we actually commanding movement?
            diff = np.abs(action_np - state_np)
            max_diff = diff.max()
            if max_diff > 0.0001:
                # Only print if movement is significant
                print(f"\r[MOVE] Prob: {prob:.3f} | Max Delta: {max_diff:.4f} | Action: {action_np}", end="")
            else:
                print(f"\r[IDLE] Prob: {prob:.3f} | No significant movement commanded", end="")
            
            now = time.time()
            ungated = getattr(cfg, "ungated", False) if hasattr(cfg, "ungated") else ("--ungated" in sys.argv)
            
            if not ungated and now < freeze_until:
                if not is_frozen:
                    print("\n⚠️  SAFETY FREEZE TRIGGERED! ⏸️")
                    is_frozen = True
                freeze_action = {k: float(v) for k, v in zip(joint_keys, state_np)}
                robot.send_action(freeze_action)
            else:
                if is_frozen and not ungated:
                    print("\n✅ Safety reset. Resuming...")
                    is_frozen = False
                    gater.reset()
                
                if not ungated and is_unsafe:
                    print(f"\n🚨 UNSAFE detected! Prob: {prob:.3f} (Threshold: {gater.threshold:.3f}) Counter: {gater.alarm_counter}")
                    freeze_until = now + 1.0 # Freeze for 1.0s safely
                    freeze_action = {k: float(v) for k, v in zip(joint_keys, state_np)}
                    robot.send_action(freeze_action)
                else:
                    robot.send_action(action_dict)
            
            # Sync loop
            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1 / fps - dt_s, 0.0))
            
    finally:
        print("\nStopping...")
        robot.disconnect()
        if listener:
            listener.stop()

if __name__ == "__main__":
    # Filter out --ungated from sys.argv before draccus sees it
    import sys
    ungated_active = "--ungated" in sys.argv
    if ungated_active:
        sys.argv.remove("--ungated")
    main()
