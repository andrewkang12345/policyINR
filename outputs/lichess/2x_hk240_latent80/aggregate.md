# Aggregate (7 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|
| lichess_full_2Xepisode | inr_transformer_fitted_latent | conflation | 1 | 0 | 0.711 | 0.711 | - | - | 0.202 | 4.750 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | generalization | 1 | 0 | 0.496 | 0.496 | - | - | 0.067 | 5.881 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | new_policy | 1 | 0 | 0.437 | 0.651 | - | - | 0.201 | 4.423 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | no_shift | 1 | 0 | 0.654 | 0.654 | - | - | 0.225 | 4.281 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | novel_generalization | 1 | 0 | 0.331 | 0.494 | - | - | 0.067 | 5.907 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | single_shift | 1 | 0 | 0.538 | 0.538 | - | - | 0.210 | 4.422 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | specialization | 1 | 0 | 0.502 | 0.502 | - | - | 0.103 | 5.185 |