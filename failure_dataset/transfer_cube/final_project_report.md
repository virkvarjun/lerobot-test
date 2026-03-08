# Failure-Aware ACT: Final Project Report
## 1. Checkpoint
- **Checkpoint:** `a23v/act_transfer_cube`
## 2. Dataset Summary
- **Total timesteps:** N/A
- **Episodes:** N/A
- **Success rate (collection):** 0.0%
## 3. Supervised Failure Predictor
- **Test AUROC:** 0.9460
- **Test AUPRC:** 0.1174
- **Test F1:** 0.1803
- **Precision:** 0.1100
- **Recall:** 0.5000
## 4. Threshold and Lead-Time
- **Recommended threshold:** 0.350
- **Reason:** Best F1 on test set
- **Mean lead time:** 99.0 steps
- **Median lead time:** 99.0 steps
- **Failed episodes never alarmed:** 0.0%
- **Success episodes with false alarm:** 0.0%
## 5. FIPER Baseline Summary
- FIPER (RND + ACE) underperformed on this setup.
- Alarm precision/recall were 0; no meaningful failure detection.
- Supervised predictor is the primary path.
## 6. Online Evaluation Summary
### ACT baseline
- **Success rate:** 75.0% (15/20)

### ACT + monitor only
- **Success rate:** 75.0% (15/20)

### ACT + monitor + intervention
- **Success rate:** 75.0% (15/20)
- **Total interventions:** 5
- **Avg interventions/episode:** 0.25
- **Recoveries after intervention:** 3

## 7. Did Intervention Improve ACT?
- Baseline success rate: 75.0%
- Intervention success rate: 75.0%
- **Change:** +0.0 percentage points

## 8. Limitations
- ACT does not natively support multi-sample latent sampling; used observation-noise proxy.
- Candidate scoring uses same-state feature; no chunk-specific embeddings.
- Small evaluation batch (20 ep) in examples; scale for conclusive results.
- FIPER baseline did not detect failures in this setup.

## 9. Recommended Future Work
- Integrate failure predictor into production deployment loop.
- Add multi-sample ACT inference if available for true latent diversity.
- Scale online evaluation for statistical significance.
- Tune threshold per deployment environment.
