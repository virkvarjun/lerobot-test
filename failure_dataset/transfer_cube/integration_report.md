# Failure Prediction Integration Report

## 1. Checkpoint and Commands

- **Checkpoint used:** `a23v/act_transfer_cube`

- **Rollout episodes collected:** 30

## 2. Processed Dataset

- **Total episodes:** 30
- **Successful:** 18
- **Failed:** 12
- **Total timesteps:** 8359
- **failure_within_k positives:** 132

## 3. Available Feature Fields

- `feat_decoder_mean`: {'shape': [8359, 512], 'dtype': 'float32'}
- `feat_encoder_latent_token`: {'shape': [8359, 512], 'dtype': 'float32'}
- `feat_latent_sample`: {'shape': [8359, 32], 'dtype': 'float32'}
- **Chosen default:** `feat_decoder_mean` (or set explicitly)

## 4. Action Chunks

- **Present:** yes
- **Shape:** {'shape': [8359, 100, 14], 'dtype': 'float32'}

## 5. ACE Mode Used

- **Mode:** `chunk_change` (proxy: magnitude of chunk-to-chunk change over rolling window)
- True runtime multi-sample ACE requires ACT to support multiple chunk samples per input.

## 6. Supervised Model Metrics

- **best_val_auroc:** 0.0000
- **test_n_samples:** 1404
- **test_n_positive:** 22
- **test_n_negative:** 1382
- **test_auroc:** 0.9460
- **test_auprc:** 0.1174
- **test_tp:** 11
- **test_tn:** 1293
- **test_fp:** 89
- **test_fn:** 11
- **test_accuracy:** 0.9288
- **test_precision:** 0.1100
- **test_recall:** 0.5000
- **test_f1:** 0.1803

## 7. FIPER Baseline Metrics

- **alarm_precision:** 0.0000
- **alarm_recall:** 0.0000
- **false_alarm_rate:** 0.0951
- **pct_failed_eps_with_alarm:** 0.0000
- **pct_success_eps_false_alarm:** 25.0000
- **lead_time_mean:** 0.0000
- **lead_time_median:** 0.0000

## 8. Caveats and Limitations

- ACE uses chunk_change proxy; true sample-dispersion ACE needs multi-sample inference.
- Calibration assumes successful episodes are nominal; distribution shift may affect thresholds.
- Episode-boundary effects in ACE when concatenating multiple episodes.

## 9. Recommended Next Steps

- Integrate failure predictor into online rollout loop for self-correction.
- Add multi-sample ACT inference if available for true ACE.
- Scale up rollout collection for more robust calibration.
