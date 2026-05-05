# Aggregate (64 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|
| custom_mujoco_action_resampled_v5_halfcheetah | cvae | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.144±0.013 | 0.057±0.006 | - | - |
| custom_mujoco_action_resampled_v5_halfcheetah | cvae | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 0.211±0.001 | 0.096±0.002 | - | - |
| custom_mujoco_action_resampled_v5_halfcheetah | cvae | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.108±0.002 | 0.054±0.001 | - | - |
| custom_mujoco_action_resampled_v5_halfcheetah | cvae | single_shift | 2 | 0 | 0.853±0.147 | 0.853±0.147 | 0.208±0.018 | 0.061±0.007 | - | - |
| custom_mujoco_action_resampled_v5_hopper | cvae | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.136±0.001 | 0.061±0.002 | - | - |
| custom_mujoco_action_resampled_v5_hopper | cvae | generalization | 2 | 0 | 0.769±0.026 | 0.769±0.026 | 1.266±0.038 | 0.388±0.014 | - | - |
| custom_mujoco_action_resampled_v5_hopper | cvae | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 0.883±0.000 | 0.349±0.027 | - | - |
| custom_mujoco_action_resampled_v5_hopper | cvae | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.219±0.013 | 0.140±0.006 | - | - |
| custom_mujoco_action_resampled_v5_hopper | cvae | novel_generalization | 2 | 0 | 0.513±0.017 | 0.769±0.026 | 1.520±0.042 | 0.687±0.033 | - | - |
| custom_mujoco_action_resampled_v5_hopper | cvae | single_shift | 2 | 0 | 0.836±0.164 | 0.836±0.164 | 0.323±0.020 | 0.215±0.001 | - | - |
| custom_mujoco_action_resampled_v5_hopper | cvae | specialization | 2 | 0 | 0.750±0.250 | 0.750±0.250 | 0.379±0.012 | 0.333±0.023 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_diffusion_history_conditioned | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.555±0.012 | 0.298±0.009 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_diffusion_history_conditioned | generalization | 2 | 0 | 0.583±0.032 | 0.583±0.032 | 0.986±0.025 | 0.392±0.016 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_diffusion_history_conditioned | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 1.111±0.001 | 0.934±0.018 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_diffusion_history_conditioned | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.895±0.008 | 0.623±0.002 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_diffusion_history_conditioned | novel_generalization | 2 | 0 | 0.389±0.021 | 0.583±0.032 | 1.179±0.009 | 0.585±0.001 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_diffusion_history_conditioned | single_shift | 2 | 0 | 0.750±0.009 | 0.750±0.009 | 0.800±0.023 | 0.820±0.014 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_diffusion_history_conditioned | specialization | 2 | 0 | 0.605±0.026 | 0.605±0.026 | 0.730±0.013 | 0.972±0.021 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_fitted_latent | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.254±0.002 | 0.156±0.011 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_fitted_latent | generalization | 2 | 0 | 0.756±0.026 | 0.756±0.026 | 1.157±0.042 | 0.481±0.018 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_fitted_latent | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 0.765±0.037 | 0.313±0.013 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_fitted_latent | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.194±0.003 | 0.124±0.001 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_fitted_latent | novel_generalization | 2 | 0 | 0.504±0.017 | 0.756±0.026 | 1.267±0.017 | 0.664±0.002 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_fitted_latent | single_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.335±0.009 | 0.223±0.009 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_fitted_latent | specialization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.396±0.018 | 0.409±0.098 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_history_conditioned | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.127±0.003 | 0.052±0.002 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_history_conditioned | generalization | 2 | 0 | 0.692±0.051 | 0.692±0.051 | 0.919±0.011 | 0.354±0.003 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_history_conditioned | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 0.778±0.048 | 0.302±0.015 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_history_conditioned | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.178±0.003 | 0.114±0.002 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_history_conditioned | novel_generalization | 2 | 0 | 0.462±0.034 | 0.692±0.051 | 1.224±0.012 | 0.599±0.012 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_history_conditioned | single_shift | 2 | 0 | 0.836±0.164 | 0.836±0.164 | 0.310±0.023 | 0.204±0.001 | - | - |
| custom_mujoco_action_resampled_v5_hopper | inr_transformer_history_conditioned | specialization | 2 | 0 | 0.737±0.237 | 0.737±0.237 | 0.357±0.020 | 0.324±0.014 | - | - |