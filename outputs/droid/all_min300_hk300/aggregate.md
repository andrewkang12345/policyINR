# Aggregate (48 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|
| droid_lowdim_full_all_min300 | cvae | conflation | 2 | 0 | 0.991±0.000 | 0.991±0.000 | 0.052±0.001 | 0.016±0.001 | - | - |
| droid_lowdim_full_all_min300 | cvae | generalization | 2 | 0 | 0.773±0.031 | 0.773±0.031 | 0.072±0.000 | 0.039±0.000 | - | - |
| droid_lowdim_full_all_min300 | cvae | new_policy | 2 | 0 | 0.744±0.028 | 0.843±0.031 | 0.053±0.000 | 0.017±0.001 | - | - |
| droid_lowdim_full_all_min300 | cvae | no_shift | 2 | 0 | 0.867±0.028 | 0.867±0.028 | 0.053±0.000 | 0.017±0.001 | - | - |
| droid_lowdim_full_all_min300 | cvae | novel_generalization | 2 | 0 | 0.694±0.015 | 0.787±0.017 | 0.071±0.000 | 0.038±0.000 | - | - |
| droid_lowdim_full_all_min300 | cvae | single_shift | 2 | 0 | 0.954±0.018 | 0.954±0.018 | 0.049±0.001 | 0.017±0.001 | - | - |
| droid_lowdim_full_all_min300 | cvae | specialization | 2 | 0 | 0.938±0.062 | 0.938±0.062 | 0.053±0.002 | 0.013±0.001 | - | - |
| droid_lowdim_full_all_min300 | inr_diffusion_history_conditioned | conflation | 2 | 0 | 0.986±0.005 | 0.986±0.005 | 0.246±0.006 | 0.034±0.006 | - | - |
| droid_lowdim_full_all_min300 | inr_diffusion_history_conditioned | generalization | 2 | 0 | 0.776±0.014 | 0.776±0.014 | 0.232±0.012 | 0.171±0.001 | - | - |
| droid_lowdim_full_all_min300 | inr_diffusion_history_conditioned | new_policy | 2 | 0 | 0.769±0.022 | 0.871±0.024 | 0.262±0.001 | 0.043±0.004 | - | - |
| droid_lowdim_full_all_min300 | inr_diffusion_history_conditioned | no_shift | 2 | 0 | 0.885±0.003 | 0.885±0.003 | 0.259±0.004 | 0.042±0.006 | - | - |
| droid_lowdim_full_all_min300 | inr_diffusion_history_conditioned | novel_generalization | 2 | 0 | 0.682±0.015 | 0.773±0.017 | 0.233±0.010 | 0.168±0.002 | - | - |
| droid_lowdim_full_all_min300 | inr_diffusion_history_conditioned | single_shift | 2 | 0 | 0.959±0.032 | 0.959±0.032 | 0.253±0.006 | 0.043±0.002 | - | - |
| droid_lowdim_full_all_min300 | inr_diffusion_history_conditioned | specialization | 2 | 0 | 0.875±0.000 | 0.875±0.000 | 0.325±0.002 | 0.048±0.000 | - | - |
| droid_lowdim_full_all_min300 | inr_transformer_fitted_latent | conflation | 2 | 0 | 0.986±0.005 | 0.986±0.005 | 0.055±0.000 | 0.019±0.001 | - | - |
| droid_lowdim_full_all_min300 | inr_transformer_fitted_latent | new_policy | 2 | 0 | 0.710±0.006 | 0.804±0.007 | 0.058±0.001 | 0.020±0.000 | - | - |
| droid_lowdim_full_all_min300 | inr_transformer_fitted_latent | no_shift | 2 | 0 | 0.801±0.003 | 0.801±0.003 | 0.057±0.001 | 0.020±0.000 | - | - |
| droid_lowdim_full_all_min300 | inr_transformer_history_conditioned | conflation | 2 | 0 | 0.991±0.000 | 0.991±0.000 | 0.010±0.000 | 0.001±0.000 | - | - |
| droid_lowdim_full_all_min300 | inr_transformer_history_conditioned | generalization | 2 | 0 | 0.759±0.003 | 0.759±0.003 | 0.147±0.006 | 0.082±0.003 | - | - |
| droid_lowdim_full_all_min300 | inr_transformer_history_conditioned | new_policy | 2 | 0 | 0.744±0.015 | 0.843±0.017 | 0.011±0.000 | 0.001±0.000 | - | - |
| droid_lowdim_full_all_min300 | inr_transformer_history_conditioned | no_shift | 2 | 0 | 0.832±0.028 | 0.832±0.028 | 0.011±0.000 | 0.001±0.000 | - | - |
| droid_lowdim_full_all_min300 | inr_transformer_history_conditioned | novel_generalization | 2 | 0 | 0.670±0.003 | 0.759±0.003 | 0.144±0.005 | 0.080±0.003 | - | - |
| droid_lowdim_full_all_min300 | inr_transformer_history_conditioned | single_shift | 2 | 0 | 0.945±0.028 | 0.945±0.028 | 0.009±0.000 | 0.001±0.000 | - | - |
| droid_lowdim_full_all_min300 | inr_transformer_history_conditioned | specialization | 2 | 0 | 0.812±0.062 | 0.812±0.062 | 0.011±0.002 | 0.001±0.000 | - | - |