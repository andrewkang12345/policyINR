# Aggregate (56 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|
| custom_mujoco_action_resampled_v4_hopper20x | cvae | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.121±0.002 | 0.044±0.004 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | cvae | generalization | 2 | 0 | 0.724±0.019 | 0.724±0.019 | 1.252±0.117 | 0.392±0.030 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | cvae | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 0.906±0.038 | 0.335±0.002 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | cvae | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.204±0.003 | 0.133±0.000 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | cvae | novel_generalization | 2 | 0 | 0.483±0.013 | 0.724±0.019 | 1.468±0.054 | 0.671±0.036 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | cvae | single_shift | 2 | 0 | 0.871±0.129 | 0.871±0.129 | 0.345±0.032 | 0.257±0.015 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | cvae | specialization | 2 | 0 | 0.763±0.158 | 0.763±0.158 | 0.371±0.023 | 0.313±0.009 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_diffusion_history_conditioned | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.478±0.016 | 0.179±0.013 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_diffusion_history_conditioned | generalization | 2 | 0 | 0.660±0.032 | 0.660±0.032 | 0.968±0.012 | 0.382±0.004 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_diffusion_history_conditioned | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 1.092±0.004 | 0.912±0.018 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_diffusion_history_conditioned | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.871±0.008 | 0.609±0.005 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_diffusion_history_conditioned | novel_generalization | 2 | 0 | 0.440±0.021 | 0.660±0.032 | 1.185±0.001 | 0.585±0.000 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_diffusion_history_conditioned | single_shift | 2 | 0 | 0.672±0.000 | 0.672±0.000 | 0.804±0.008 | 0.874±0.000 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_diffusion_history_conditioned | specialization | 2 | 0 | 0.500±0.000 | 0.500±0.000 | 0.757±0.024 | 0.995±0.021 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_fitted_latent | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.242±0.003 | 0.145±0.002 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_fitted_latent | generalization | 2 | 0 | 0.744±0.000 | 0.744±0.000 | 1.105±0.058 | 0.464±0.027 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_fitted_latent | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 0.765±0.037 | 0.313±0.012 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_fitted_latent | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.194±0.003 | 0.124±0.001 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_fitted_latent | novel_generalization | 2 | 0 | 0.496±0.000 | 0.744±0.000 | 1.211±0.036 | 0.637±0.023 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_fitted_latent | single_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.377±0.012 | 0.330±0.037 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_fitted_latent | specialization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.395±0.017 | 0.398±0.087 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_history_conditioned | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.117±0.000 | 0.036±0.003 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_history_conditioned | generalization | 2 | 0 | 0.603±0.103 | 0.603±0.103 | 0.995±0.031 | 0.400±0.020 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_history_conditioned | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 0.795±0.014 | 0.298±0.015 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_history_conditioned | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.178±0.003 | 0.112±0.001 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_history_conditioned | novel_generalization | 2 | 0 | 0.402±0.068 | 0.603±0.103 | 1.226±0.037 | 0.614±0.030 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_history_conditioned | single_shift | 2 | 0 | 0.836±0.164 | 0.836±0.164 | 0.383±0.060 | 0.315±0.032 | - | - |
| custom_mujoco_action_resampled_v4_hopper20x | inr_transformer_history_conditioned | specialization | 2 | 0 | 0.618±0.118 | 0.618±0.118 | 0.381±0.057 | 0.387±0.031 | - | - |