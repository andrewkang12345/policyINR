# Aggregate (7 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `knn_acc1`/`knn_acc5` = leave-one-out cosine kNN policy accuracy on held-out eval embeddings; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | knn_acc@1 | knn_acc@5 | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lichess_top3_full | inr_transformer_fitted_latent | conflation | 1 | 0 | 0.723 | 0.723 | 0.729 | 0.819 | - | - | 0.195 | 4.827 |
| lichess_top3_full | inr_transformer_fitted_latent | generalization | 1 | 0 | 0.541 | 0.541 | 0.596 | 0.973 | - | - | 0.071 | 5.805 |
| lichess_top3_full | inr_transformer_fitted_latent | new_policy | 1 | 0 | 0.520 | 0.777 | 0.447 | 0.891 | - | - | 0.203 | 4.500 |
| lichess_top3_full | inr_transformer_fitted_latent | no_shift | 1 | 0 | 0.777 | 0.777 | 0.615 | 0.982 | - | - | 0.225 | 4.341 |
| lichess_top3_full | inr_transformer_fitted_latent | novel_generalization | 1 | 0 | 0.362 | 0.541 | 0.451 | 0.911 | - | - | 0.071 | 5.842 |
| lichess_top3_full | inr_transformer_fitted_latent | single_shift | 1 | 0 | 0.701 | 0.701 | 0.723 | 0.814 | - | - | 0.203 | 4.539 |
| lichess_top3_full | inr_transformer_fitted_latent | specialization | 1 | 0 | 0.499 | 0.499 | 0.479 | 0.973 | - | - | 0.089 | 5.416 |