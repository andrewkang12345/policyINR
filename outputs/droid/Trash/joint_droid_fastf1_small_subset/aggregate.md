# Aggregate (126 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|
| fastf1_stint_full | cvae | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.496±0.047 | 0.172±0.023 | - | - |
| fastf1_stint_full | cvae | generalization | 2 | 0 | 0.375±0.042 | 0.375±0.042 | 3.023±2.143 | 0.597±0.094 | - | - |
| fastf1_stint_full | cvae | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 0.392±0.032 | 0.093±0.010 | - | - |
| fastf1_stint_full | cvae | no_shift | 2 | 0 | 0.958±0.042 | 0.958±0.042 | 0.363±0.031 | 0.088±0.026 | - | - |
| fastf1_stint_full | cvae | novel_generalization | 2 | 0 | 0.250±0.028 | 0.375±0.042 | 2.788±1.923 | 0.791±0.336 | - | - |
| fastf1_stint_full | cvae | single_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.399±0.006 | 0.129±0.013 | - | - |
| fastf1_stint_full | cvae | specialization | 2 | 0 | 0.500±0.000 | 0.500±0.000 | 0.671±0.091 | 0.213±0.034 | - | - |
| fastf1_stint_full | inr_diffusion_history_conditioned | conflation | 2 | 0 | 0.938±0.062 | 0.938±0.062 | 1.113±0.044 | 0.310±0.112 | - | - |
| fastf1_stint_full | inr_diffusion_history_conditioned | generalization | 2 | 0 | 0.542±0.292 | 0.542±0.292 | 1.239±0.036 | 0.915±0.091 | - | - |
| fastf1_stint_full | inr_diffusion_history_conditioned | new_policy | 2 | 0 | 0.389±0.111 | 0.583±0.167 | 1.022±0.052 | 0.206±0.008 | - | - |
| fastf1_stint_full | inr_diffusion_history_conditioned | no_shift | 2 | 0 | 0.750±0.083 | 0.750±0.083 | 1.083±0.050 | 0.208±0.030 | - | - |
| fastf1_stint_full | inr_diffusion_history_conditioned | novel_generalization | 2 | 0 | 0.306±0.028 | 0.458±0.042 | 1.197±0.082 | 0.985±0.040 | - | - |
| fastf1_stint_full | inr_diffusion_history_conditioned | single_shift | 2 | 0 | 0.688±0.062 | 0.688±0.062 | 1.021±0.021 | 0.219±0.037 | - | - |
| fastf1_stint_full | inr_diffusion_history_conditioned | specialization | 2 | 0 | 0.500±0.000 | 0.500±0.000 | 1.256±0.081 | 0.299±0.055 | - | - |
| fastf1_stint_full | inr_transformer_fitted_latent | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.416±0.070 | 0.083±0.009 | - | - |
| fastf1_stint_full | inr_transformer_fitted_latent | generalization | 2 | 0 | 0.500±0.167 | 0.500±0.167 | 1.316±0.096 | 0.990±0.213 | - | - |
| fastf1_stint_full | inr_transformer_fitted_latent | new_policy | 2 | 0 | 0.667±0.000 | 1.000±0.000 | 0.385±0.050 | 0.072±0.018 | - | - |
| fastf1_stint_full | inr_transformer_fitted_latent | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.373±0.059 | 0.066±0.011 | - | - |
| fastf1_stint_full | inr_transformer_fitted_latent | novel_generalization | 2 | 0 | 0.333±0.111 | 0.500±0.167 | 1.250±0.027 | 1.013±0.233 | - | - |
| fastf1_stint_full | inr_transformer_fitted_latent | single_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.425±0.058 | 0.098±0.017 | - | - |
| fastf1_stint_full | inr_transformer_fitted_latent | specialization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 0.621±0.018 | 0.186±0.035 | - | - |
| fastf1_stint_full | inr_transformer_history_conditioned | conflation | 2 | 0 | 0.938±0.062 | 0.938±0.062 | 0.288±0.063 | 0.032±0.004 | - | - |
| fastf1_stint_full | inr_transformer_history_conditioned | generalization | 2 | 0 | 0.417±0.083 | 0.417±0.083 | 1.243±0.103 | 0.996±0.188 | - | - |
| fastf1_stint_full | inr_transformer_history_conditioned | new_policy | 2 | 0 | 0.528±0.028 | 0.792±0.042 | 0.188±0.012 | 0.013±0.001 | - | - |
| fastf1_stint_full | inr_transformer_history_conditioned | no_shift | 2 | 0 | 0.917±0.000 | 0.917±0.000 | 0.180±0.015 | 0.014±0.001 | - | - |
| fastf1_stint_full | inr_transformer_history_conditioned | novel_generalization | 2 | 0 | 0.278±0.056 | 0.417±0.083 | 1.184±0.005 | 0.971±0.145 | - | - |
| fastf1_stint_full | inr_transformer_history_conditioned | single_shift | 2 | 0 | 0.812±0.062 | 0.812±0.062 | 0.216±0.038 | 0.023±0.001 | - | - |
| fastf1_stint_full | inr_transformer_history_conditioned | specialization | 2 | 0 | 0.250±0.250 | 0.250±0.250 | 0.638±0.151 | 0.123±0.033 | - | - |