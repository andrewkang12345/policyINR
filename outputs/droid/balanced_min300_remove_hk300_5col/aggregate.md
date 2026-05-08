# Aggregate (57 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `knn_acc1`/`knn_acc5` = leave-one-out cosine kNN policy accuracy on held-out eval embeddings; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | knn_acc@1 | knn_acc@5 | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| droid_lowdim_full_balanced_min300_remove_5col | cvae | conflation_5p | 2 | 0 | 0.641±0.006 | 0.641±0.006 | 0.494±0.047 | 0.888±0.006 | 0.065±0.010 | 0.021±0.001 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | cvae | generalization_5p | 2 | 0 | 0.326±0.045 | 0.326±0.045 | 0.478±0.073 | 0.848±0.006 | 0.092±0.011 | 0.053±0.001 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | cvae | new_policy_5p | 2 | 0 | 0.478±0.006 | 0.708±0.008 | 0.360±0.022 | 0.831±0.034 | 0.071±0.010 | 0.019±0.000 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | cvae | no_shift_5p | 2 | 0 | 0.601±0.006 | 0.601±0.006 | 0.354±0.073 | 0.826±0.006 | 0.066±0.010 | 0.022±0.002 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | cvae | novel_generalization_5p | 2 | 0 | 0.258±0.034 | 0.383±0.050 | 0.427±0.034 | 0.815±0.006 | 0.105±0.012 | 0.060±0.006 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | cvae | single_shift_5p | 2 | 0 | 0.635±0.000 | 0.635±0.000 | 0.435±0.047 | 0.882±0.024 | 0.066±0.010 | 0.021±0.001 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | cvae | specialization_5p | 2 | 0 | 0.500±0.100 | 0.500±0.100 | 0.100±0.100 | 0.400±0.000 | 0.087±0.013 | 0.023±0.000 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_diffusion_history_conditioned | conflation_5p | 2 | 0 | 0.700±0.006 | 0.700±0.006 | 0.371±0.029 | 0.841±0.029 | 0.257±0.032 | 0.031±0.002 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_diffusion_history_conditioned | generalization_5p | 2 | 0 | 0.382±0.056 | 0.382±0.056 | 0.433±0.006 | 0.826±0.028 | 0.311±0.010 | 0.289±0.015 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_diffusion_history_conditioned | new_policy_5p | 2 | 0 | 0.466±0.017 | 0.692±0.025 | 0.320±0.062 | 0.803±0.017 | 0.287±0.029 | 0.024±0.004 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_diffusion_history_conditioned | no_shift_5p | 2 | 0 | 0.635±0.073 | 0.635±0.073 | 0.343±0.028 | 0.826±0.006 | 0.281±0.033 | 0.050±0.005 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_diffusion_history_conditioned | novel_generalization_5p | 2 | 0 | 0.258±0.022 | 0.383±0.033 | 0.421±0.073 | 0.843±0.045 | 0.309±0.020 | 0.235±0.036 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_diffusion_history_conditioned | single_shift_5p | 2 | 0 | 0.659±0.035 | 0.659±0.035 | 0.376±0.059 | 0.859±0.024 | 0.253±0.005 | 0.041±0.002 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_diffusion_history_conditioned | specialization_5p | 2 | 0 | 0.400±0.000 | 0.400±0.000 | 0.000±0.000 | 0.400±0.000 | 0.291±0.060 | 0.022±0.001 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_fitted_latent | conflation_5p | 2 | 0 | 0.676±0.018 | 0.676±0.018 | 0.453±0.053 | 0.888±0.018 | 0.068±0.008 | 0.026±0.001 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_fitted_latent | generalization_5p | 2 | 0 | 0.315±0.079 | 0.315±0.079 | 0.388±0.039 | 0.798±0.000 | 0.151±0.017 | 0.078±0.009 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_fitted_latent | new_policy_5p | 2 | 0 | 0.455±0.017 | 0.675±0.025 | 0.410±0.028 | 0.865±0.022 | 0.074±0.008 | 0.023±0.000 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_fitted_latent | no_shift_5p | 2 | 0 | 0.573±0.000 | 0.573±0.000 | 0.427±0.022 | 0.854±0.034 | 0.069±0.009 | 0.026±0.003 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_fitted_latent | novel_generalization_5p | 2 | 0 | 0.253±0.017 | 0.375±0.025 | 0.331±0.028 | 0.815±0.017 | 0.172±0.010 | 0.090±0.009 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_fitted_latent | single_shift_5p | 2 | 0 | 0.588±0.012 | 0.588±0.012 | 0.476±0.029 | 0.882±0.012 | 0.068±0.009 | 0.026±0.003 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_fitted_latent | specialization_5p | 2 | 0 | 0.500±0.100 | 0.500±0.100 | 0.100±0.100 | 0.400±0.000 | 0.093±0.016 | 0.028±0.004 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_history_conditioned | conflation_5p | 2 | 0 | 0.606±0.065 | 0.606±0.065 | 0.418±0.053 | 0.835±0.012 | 0.013±0.003 | 0.001±0.000 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_history_conditioned | generalization_5p | 2 | 0 | 0.343±0.006 | 0.343±0.006 | 0.500±0.051 | 0.871±0.017 | 0.184±0.000 | 0.106±0.000 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_history_conditioned | new_policy_5p | 2 | 0 | 0.461±0.011 | 0.683±0.017 | 0.337±0.034 | 0.837±0.006 | 0.016±0.003 | 0.001±0.000 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_history_conditioned | no_shift_5p | 2 | 0 | 0.596±0.045 | 0.596±0.045 | 0.354±0.017 | 0.848±0.017 | 0.013±0.003 | 0.001±0.000 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_history_conditioned | novel_generalization_5p | 2 | 0 | 0.270±0.000 | 0.400±0.000 | 0.466±0.028 | 0.871±0.006 | 0.211±0.011 | 0.121±0.009 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_history_conditioned | single_shift_5p | 2 | 0 | 0.624±0.012 | 0.624±0.012 | 0.412±0.024 | 0.835±0.000 | 0.013±0.003 | 0.001±0.000 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_history_conditioned | specialization_5p | 2 | 0 | 0.300±0.100 | 0.300±0.100 | 0.000±0.000 | 0.400±0.000 | 0.016±0.003 | 0.001±0.000 | - | - |
| droid_lowdim_full_balanced_min300_remove_5col | inr_transformer_infer_latent_maml | new_policy_5p | 1 | 0 | 0.472 | 0.700 | 0.382 | 0.843 | 0.082 | 0.018 | - | - |