# Aggregate (3 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|
| lichess_top3_full | inr_transformer_fitted_latent | conflation | 1 | 0 | 0.723 | 0.723 | - | - | 0.195 | 4.827 |
| lichess_top3_full | inr_transformer_fitted_latent | generalization | 1 | 0 | 0.541 | 0.541 | - | - | 0.071 | 5.805 |
| lichess_top3_full | inr_transformer_fitted_latent | no_shift | 1 | 0 | 0.777 | 0.777 | - | - | 0.225 | 4.341 |