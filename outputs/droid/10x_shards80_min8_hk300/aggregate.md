# Aggregate (70 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|
| droid_lowdim_full_10x | cvae | conflation | 2 | 0 | 0.932±0.000 | 0.932±0.000 | 0.078±0.011 | 0.022±0.001 | - | - |
| droid_lowdim_full_10x | cvae | generalization | 2 | 0 | 0.534±0.006 | 0.534±0.006 | 0.082±0.007 | 0.043±0.001 | - | - |
| droid_lowdim_full_10x | cvae | new_policy | 2 | 0 | 0.648±0.013 | 0.837±0.017 | 0.077±0.006 | 0.025±0.001 | - | - |
| droid_lowdim_full_10x | cvae | no_shift | 2 | 0 | 0.865±0.022 | 0.865±0.022 | 0.076±0.005 | 0.025±0.000 | - | - |
| droid_lowdim_full_10x | cvae | novel_generalization | 2 | 0 | 0.417±0.009 | 0.539±0.011 | 0.082±0.006 | 0.044±0.001 | - | - |
| droid_lowdim_full_10x | cvae | single_shift | 2 | 0 | 0.784±0.034 | 0.784±0.034 | 0.072±0.010 | 0.025±0.000 | - | - |
| droid_lowdim_full_10x | cvae | specialization | 2 | 0 | 0.688±0.062 | 0.688±0.062 | 0.066±0.001 | 0.022±0.001 | - | - |
| droid_lowdim_full_10x | inr_diffusion_history_conditioned | conflation | 2 | 0 | 0.932±0.000 | 0.932±0.000 | 0.279±0.011 | 0.025±0.010 | - | - |
| droid_lowdim_full_10x | inr_diffusion_history_conditioned | generalization | 2 | 0 | 0.461±0.034 | 0.461±0.034 | 0.158±0.004 | 0.037±0.005 | - | - |
| droid_lowdim_full_10x | inr_diffusion_history_conditioned | new_policy | 2 | 0 | 0.630±0.004 | 0.815±0.006 | 0.273±0.021 | 0.037±0.005 | - | - |
| droid_lowdim_full_10x | inr_diffusion_history_conditioned | no_shift | 2 | 0 | 0.848±0.006 | 0.848±0.006 | 0.274±0.016 | 0.038±0.006 | - | - |
| droid_lowdim_full_10x | inr_diffusion_history_conditioned | novel_generalization | 2 | 0 | 0.370±0.030 | 0.478±0.039 | 0.158±0.004 | 0.035±0.004 | - | - |
| droid_lowdim_full_10x | inr_diffusion_history_conditioned | single_shift | 2 | 0 | 0.784±0.011 | 0.784±0.011 | 0.274±0.020 | 0.049±0.013 | - | - |
| droid_lowdim_full_10x | inr_diffusion_history_conditioned | specialization | 2 | 0 | 0.719±0.094 | 0.719±0.094 | 0.291±0.010 | 0.044±0.015 | - | - |
| droid_lowdim_full_10x | inr_transformer_fitted_latent | conflation | 2 | 0 | 0.807±0.011 | 0.807±0.011 | 0.090±0.012 | 0.024±0.003 | - | - |
| droid_lowdim_full_10x | inr_transformer_fitted_latent | generalization | 2 | 0 | 0.449±0.034 | 0.449±0.034 | 0.118±0.001 | 0.058±0.000 | - | - |
| droid_lowdim_full_10x | inr_transformer_fitted_latent | new_policy | 2 | 0 | 0.530±0.026 | 0.685±0.034 | 0.084±0.007 | 0.027±0.001 | - | - |
| droid_lowdim_full_10x | inr_transformer_fitted_latent | no_shift | 2 | 0 | 0.708±0.045 | 0.708±0.045 | 0.085±0.008 | 0.027±0.001 | - | - |
| droid_lowdim_full_10x | inr_transformer_fitted_latent | novel_generalization | 2 | 0 | 0.352±0.030 | 0.455±0.039 | 0.116±0.001 | 0.058±0.001 | - | - |
| droid_lowdim_full_10x | inr_transformer_fitted_latent | single_shift | 2 | 0 | 0.591±0.045 | 0.591±0.045 | 0.085±0.015 | 0.027±0.001 | - | - |
| droid_lowdim_full_10x | inr_transformer_fitted_latent | specialization | 2 | 0 | 0.656±0.031 | 0.656±0.031 | 0.073±0.004 | 0.024±0.001 | - | - |
| droid_lowdim_full_10x | inr_transformer_history_conditioned | conflation | 2 | 0 | 0.943±0.011 | 0.943±0.011 | 0.020±0.001 | 0.002±0.000 | - | - |
| droid_lowdim_full_10x | inr_transformer_history_conditioned | generalization | 2 | 0 | 0.421±0.006 | 0.421±0.006 | 0.116±0.004 | 0.054±0.003 | - | - |
| droid_lowdim_full_10x | inr_transformer_history_conditioned | new_policy | 2 | 0 | 0.613±0.013 | 0.792±0.017 | 0.018±0.001 | 0.001±0.000 | - | - |
| droid_lowdim_full_10x | inr_transformer_history_conditioned | no_shift | 2 | 0 | 0.803±0.017 | 0.803±0.017 | 0.019±0.002 | 0.001±0.000 | - | - |
| droid_lowdim_full_10x | inr_transformer_history_conditioned | novel_generalization | 2 | 0 | 0.343±0.004 | 0.444±0.006 | 0.112±0.000 | 0.052±0.000 | - | - |
| droid_lowdim_full_10x | inr_transformer_history_conditioned | single_shift | 2 | 0 | 0.727±0.045 | 0.727±0.045 | 0.017±0.000 | 0.001±0.000 | - | - |
| droid_lowdim_full_10x | inr_transformer_history_conditioned | specialization | 2 | 0 | 0.656±0.094 | 0.656±0.094 | 0.015±0.001 | 0.001±0.000 | - | - |
| droid_lowdim_full_10x | inr_transformer_infer_latent_maml | conflation | 2 | 0 | 0.807±0.102 | 0.807±0.102 | 0.083±0.010 | 0.024±0.001 | - | - |
| droid_lowdim_full_10x | inr_transformer_infer_latent_maml | generalization | 2 | 0 | 0.483±0.011 | 0.483±0.011 | 0.121±0.000 | 0.063±0.003 | - | - |
| droid_lowdim_full_10x | inr_transformer_infer_latent_maml | new_policy | 2 | 0 | 0.574±0.009 | 0.742±0.011 | 0.083±0.004 | 0.028±0.002 | - | - |
| droid_lowdim_full_10x | inr_transformer_infer_latent_maml | no_shift | 2 | 0 | 0.736±0.006 | 0.736±0.006 | 0.084±0.005 | 0.028±0.001 | - | - |
| droid_lowdim_full_10x | inr_transformer_infer_latent_maml | novel_generalization | 2 | 0 | 0.383±0.009 | 0.494±0.011 | 0.118±0.001 | 0.060±0.003 | - | - |
| droid_lowdim_full_10x | inr_transformer_infer_latent_maml | single_shift | 2 | 0 | 0.693±0.011 | 0.693±0.011 | 0.084±0.011 | 0.029±0.001 | - | - |
| droid_lowdim_full_10x | inr_transformer_infer_latent_maml | specialization | 2 | 0 | 0.438±0.000 | 0.438±0.000 | 0.071±0.002 | 0.023±0.002 | - | - |