# Aggregate (7 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `knn_acc1`/`knn_acc5` = leave-one-out cosine kNN policy accuracy on held-out eval embeddings; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | knn_acc@1 | knn_acc@5 | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lichess_full_2Xepisode | inr_transformer_fitted_latent | conflation | 1 | 0 | 0.711 | 0.711 | 0.720 | 0.789 | - | - | 0.202 | 4.749 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | generalization | 1 | 0 | 0.496 | 0.496 | 0.591 | 0.977 | - | - | 0.068 | 5.881 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | new_policy | 1 | 0 | 0.438 | 0.654 | 0.353 | 0.862 | - | - | 0.201 | 4.422 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | no_shift | 1 | 0 | 0.656 | 0.656 | 0.504 | 0.971 | - | - | 0.225 | 4.280 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | novel_generalization | 1 | 0 | 0.331 | 0.494 | 0.414 | 0.922 | - | - | 0.067 | 5.907 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | single_shift | 1 | 0 | 0.538 | 0.538 | 0.725 | 0.795 | - | - | 0.210 | 4.422 |
| lichess_full_2Xepisode | inr_transformer_fitted_latent | specialization | 1 | 0 | 0.502 | 0.502 | 0.517 | 0.971 | - | - | 0.103 | 5.182 |