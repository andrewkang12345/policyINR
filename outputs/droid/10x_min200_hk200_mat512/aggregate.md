# Aggregate (70 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|
| droid_lowdim_full_10x_min200 | cvae | conflation | 2 | 0 | 0.923±0.038 | 0.923±0.038 | 0.066±0.007 | 0.018±0.000 | - | - |
| droid_lowdim_full_10x_min200 | cvae | generalization | 2 | 0 | 0.758±0.061 | 0.758±0.061 | 0.078±0.005 | 0.045±0.005 | - | - |
| droid_lowdim_full_10x_min200 | cvae | new_policy | 2 | 0 | 0.688±0.113 | 0.833±0.136 | 0.072±0.007 | 0.019±0.001 | - | - |
| droid_lowdim_full_10x_min200 | cvae | no_shift | 2 | 0 | 0.833±0.136 | 0.833±0.136 | 0.072±0.005 | 0.019±0.002 | - | - |
| droid_lowdim_full_10x_min200 | cvae | novel_generalization | 2 | 0 | 0.625±0.050 | 0.758±0.061 | 0.077±0.006 | 0.042±0.003 | - | - |
| droid_lowdim_full_10x_min200 | cvae | single_shift | 2 | 0 | 0.865±0.096 | 0.865±0.096 | 0.064±0.005 | 0.020±0.003 | - | - |
| droid_lowdim_full_10x_min200 | cvae | specialization | 2 | 0 | 0.833±0.000 | 0.833±0.000 | 0.067±0.006 | 0.015±0.000 | - | - |
| droid_lowdim_full_10x_min200 | inr_diffusion_history_conditioned | conflation | 2 | 0 | 0.923±0.038 | 0.923±0.038 | 0.163±0.023 | 0.013±0.001 | - | - |
| droid_lowdim_full_10x_min200 | inr_diffusion_history_conditioned | generalization | 2 | 0 | 0.727±0.030 | 0.727±0.030 | 0.249±0.019 | 0.211±0.027 | - | - |
| droid_lowdim_full_10x_min200 | inr_diffusion_history_conditioned | new_policy | 2 | 0 | 0.738±0.013 | 0.894±0.015 | 0.193±0.030 | 0.012±0.003 | - | - |
| droid_lowdim_full_10x_min200 | inr_diffusion_history_conditioned | no_shift | 2 | 0 | 0.894±0.015 | 0.894±0.015 | 0.187±0.028 | 0.013±0.002 | - | - |
| droid_lowdim_full_10x_min200 | inr_diffusion_history_conditioned | novel_generalization | 2 | 0 | 0.600±0.025 | 0.727±0.030 | 0.249±0.018 | 0.200±0.028 | - | - |
| droid_lowdim_full_10x_min200 | inr_diffusion_history_conditioned | single_shift | 2 | 0 | 0.808±0.077 | 0.808±0.077 | 0.174±0.029 | 0.015±0.001 | - | - |
| droid_lowdim_full_10x_min200 | inr_diffusion_history_conditioned | specialization | 2 | 0 | 0.917±0.083 | 0.917±0.083 | 0.228±0.011 | 0.011±0.002 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_fitted_latent | conflation | 2 | 0 | 0.827±0.096 | 0.827±0.096 | 0.077±0.006 | 0.024±0.003 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_fitted_latent | generalization | 2 | 0 | 0.485±0.061 | 0.485±0.061 | 0.134±0.002 | 0.076±0.009 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_fitted_latent | new_policy | 2 | 0 | 0.675±0.050 | 0.818±0.061 | 0.079±0.006 | 0.023±0.001 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_fitted_latent | no_shift | 2 | 0 | 0.818±0.061 | 0.818±0.061 | 0.080±0.005 | 0.023±0.001 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_fitted_latent | novel_generalization | 2 | 0 | 0.400±0.050 | 0.485±0.061 | 0.132±0.002 | 0.071±0.008 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_fitted_latent | single_shift | 2 | 0 | 0.904±0.019 | 0.904±0.019 | 0.074±0.005 | 0.024±0.002 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_fitted_latent | specialization | 2 | 0 | 0.667±0.167 | 0.667±0.167 | 0.070±0.011 | 0.018±0.001 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_history_conditioned | conflation | 2 | 0 | 0.923±0.038 | 0.923±0.038 | 0.029±0.007 | 0.005±0.000 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_history_conditioned | generalization | 2 | 0 | 0.697±0.061 | 0.697±0.061 | 0.156±0.005 | 0.092±0.019 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_history_conditioned | new_policy | 2 | 0 | 0.712±0.013 | 0.864±0.015 | 0.023±0.004 | 0.003±0.000 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_history_conditioned | no_shift | 2 | 0 | 0.864±0.015 | 0.864±0.015 | 0.023±0.004 | 0.003±0.000 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_history_conditioned | novel_generalization | 2 | 0 | 0.575±0.050 | 0.697±0.061 | 0.153±0.003 | 0.086±0.017 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_history_conditioned | single_shift | 2 | 0 | 0.865±0.019 | 0.865±0.019 | 0.023±0.005 | 0.003±0.000 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_history_conditioned | specialization | 2 | 0 | 0.750±0.083 | 0.750±0.083 | 0.018±0.002 | 0.002±0.000 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_infer_latent_maml | conflation | 2 | 0 | 0.904±0.019 | 0.904±0.019 | 0.076±0.005 | 0.022±0.001 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_infer_latent_maml | generalization | 2 | 0 | 0.394±0.000 | 0.394±0.000 | 0.138±0.000 | 0.085±0.019 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_infer_latent_maml | new_policy | 2 | 0 | 0.550±0.000 | 0.667±0.000 | 0.080±0.007 | 0.024±0.000 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_infer_latent_maml | no_shift | 2 | 0 | 0.667±0.000 | 0.667±0.000 | 0.080±0.006 | 0.024±0.000 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_infer_latent_maml | novel_generalization | 2 | 0 | 0.325±0.000 | 0.394±0.000 | 0.137±0.000 | 0.079±0.016 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_infer_latent_maml | single_shift | 2 | 0 | 0.731±0.000 | 0.731±0.000 | 0.074±0.005 | 0.025±0.000 | - | - |
| droid_lowdim_full_10x_min200 | inr_transformer_infer_latent_maml | specialization | 2 | 0 | 0.750±0.083 | 0.750±0.083 | 0.070±0.007 | 0.017±0.000 | - | - |