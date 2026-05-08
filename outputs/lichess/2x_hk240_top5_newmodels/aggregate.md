# Aggregate (42 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `knn_acc1`/`knn_acc5` = leave-one-out cosine kNN policy accuracy on held-out eval embeddings; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | knn_acc@1 | knn_acc@5 | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lichess_top5_full_2Xepisode | cvae_rnn | conflation_5p | 1 | 0 | 0.759 | 0.759 | 0.620 | 0.881 | - | - | 0.170 | 4.186 |
| lichess_top5_full_2Xepisode | cvae_rnn | generalization_5p | 1 | 0 | 0.257 | 0.257 | 0.243 | 0.710 | - | - | 0.071 | 5.411 |
| lichess_top5_full_2Xepisode | cvae_rnn | new_policy_5p | 1 | 0 | 0.671 | 0.850 | 0.669 | 0.908 | - | - | 0.173 | 4.222 |
| lichess_top5_full_2Xepisode | cvae_rnn | no_shift_5p | 1 | 0 | 0.845 | 0.845 | 0.751 | 0.916 | - | - | 0.185 | 3.991 |
| lichess_top5_full_2Xepisode | cvae_rnn | novel_generalization_5p | 1 | 0 | 0.308 | 0.390 | 0.230 | 0.718 | - | - | 0.057 | 5.697 |
| lichess_top5_full_2Xepisode | cvae_rnn | single_shift_5p | 1 | 0 | 0.754 | 0.754 | 0.677 | 0.908 | - | - | 0.170 | 4.071 |
| lichess_top5_full_2Xepisode | cvae_rnn | specialization_5p | 1 | 0 | 0.441 | 0.441 | 0.387 | 0.794 | - | - | 0.113 | 4.550 |
| lichess_top5_full_2Xepisode | inr_diffusion_history_conditioned_shuffle | conflation_5p | 1 | 0 | 0.571 | 0.571 | 0.360 | 0.830 | - | - | 0.001 | 0.000 |
| lichess_top5_full_2Xepisode | inr_diffusion_history_conditioned_shuffle | generalization_5p | 1 | 0 | 0.246 | 0.246 | 0.286 | 0.742 | - | - | 0.001 | 0.000 |
| lichess_top5_full_2Xepisode | inr_diffusion_history_conditioned_shuffle | new_policy_5p | 1 | 0 | 0.413 | 0.523 | 0.284 | 0.722 | - | - | 0.001 | 0.000 |
| lichess_top5_full_2Xepisode | inr_diffusion_history_conditioned_shuffle | no_shift_5p | 1 | 0 | 0.480 | 0.480 | 0.300 | 0.748 | - | - | 0.001 | 0.000 |
| lichess_top5_full_2Xepisode | inr_diffusion_history_conditioned_shuffle | novel_generalization_5p | 1 | 0 | 0.313 | 0.397 | 0.280 | 0.723 | - | - | 0.001 | 0.000 |
| lichess_top5_full_2Xepisode | inr_diffusion_history_conditioned_shuffle | single_shift_5p | 1 | 0 | 0.432 | 0.432 | 0.470 | 0.853 | - | - | 0.001 | 0.000 |
| lichess_top5_full_2Xepisode | inr_diffusion_history_conditioned_shuffle | specialization_5p | 1 | 0 | 0.237 | 0.237 | 0.217 | 0.677 | - | - | 0.000 | 0.000 |
| lichess_top5_full_2Xepisode | inr_transformer_fitted_latent_shuffle | conflation_5p | 1 | 0 | 0.351 | 0.351 | 0.304 | 0.715 | - | - | 0.205 | 3.858 |
| lichess_top5_full_2Xepisode | inr_transformer_fitted_latent_shuffle | generalization_5p | 1 | 0 | 0.239 | 0.239 | 0.243 | 0.724 | - | - | 0.102 | 5.043 |
| lichess_top5_full_2Xepisode | inr_transformer_fitted_latent_shuffle | new_policy_5p | 1 | 0 | 0.300 | 0.381 | 0.225 | 0.703 | - | - | 0.216 | 3.851 |
| lichess_top5_full_2Xepisode | inr_transformer_fitted_latent_shuffle | no_shift_5p | 1 | 0 | 0.362 | 0.362 | 0.206 | 0.684 | - | - | 0.233 | 3.596 |
| lichess_top5_full_2Xepisode | inr_transformer_fitted_latent_shuffle | novel_generalization_5p | 1 | 0 | 0.238 | 0.301 | 0.255 | 0.743 | - | - | 0.096 | 5.259 |
| lichess_top5_full_2Xepisode | inr_transformer_fitted_latent_shuffle | single_shift_5p | 1 | 0 | 0.373 | 0.373 | 0.248 | 0.706 | - | - | 0.213 | 3.703 |
| lichess_top5_full_2Xepisode | inr_transformer_fitted_latent_shuffle | specialization_5p | 1 | 0 | 0.221 | 0.221 | 0.170 | 0.637 | - | - | 0.137 | 4.196 |
| lichess_top5_full_2Xepisode | inr_transformer_history_conditioned_shuffle | conflation_5p | 1 | 0 | 0.765 | 0.765 | 0.629 | 0.849 | - | - | 0.198 | 3.910 |
| lichess_top5_full_2Xepisode | inr_transformer_history_conditioned_shuffle | generalization_5p | 1 | 0 | 0.266 | 0.266 | 0.233 | 0.720 | - | - | 0.101 | 5.102 |
| lichess_top5_full_2Xepisode | inr_transformer_history_conditioned_shuffle | new_policy_5p | 1 | 0 | 0.661 | 0.838 | 0.607 | 0.902 | - | - | 0.207 | 3.907 |
| lichess_top5_full_2Xepisode | inr_transformer_history_conditioned_shuffle | no_shift_5p | 1 | 0 | 0.827 | 0.827 | 0.730 | 0.932 | - | - | 0.224 | 3.639 |
| lichess_top5_full_2Xepisode | inr_transformer_history_conditioned_shuffle | novel_generalization_5p | 1 | 0 | 0.323 | 0.409 | 0.244 | 0.744 | - | - | 0.095 | 5.317 |
| lichess_top5_full_2Xepisode | inr_transformer_history_conditioned_shuffle | single_shift_5p | 1 | 0 | 0.708 | 0.708 | 0.681 | 0.915 | - | - | 0.206 | 3.722 |
| lichess_top5_full_2Xepisode | inr_transformer_history_conditioned_shuffle | specialization_5p | 1 | 0 | 0.355 | 0.355 | 0.241 | 0.712 | - | - | 0.134 | 4.196 |
| lichess_top5_full_2Xepisode | state_action_bem | conflation_5p | 1 | 0 | 0.700 | 0.700 | 0.416 | 0.836 | - | - | 0.017 | 0.000 |
| lichess_top5_full_2Xepisode | state_action_bem | generalization_5p | 1 | 0 | 0.212 | 0.212 | 0.246 | 0.727 | - | - | 0.012 | 0.000 |
| lichess_top5_full_2Xepisode | state_action_bem | new_policy_5p | 1 | 0 | 0.442 | 0.560 | 0.314 | 0.752 | - | - | 0.016 | 0.000 |
| lichess_top5_full_2Xepisode | state_action_bem | no_shift_5p | 1 | 0 | 0.573 | 0.573 | 0.333 | 0.767 | - | - | 0.016 | 0.000 |
| lichess_top5_full_2Xepisode | state_action_bem | novel_generalization_5p | 1 | 0 | 0.231 | 0.293 | 0.247 | 0.717 | - | - | 0.018 | 0.000 |
| lichess_top5_full_2Xepisode | state_action_bem | single_shift_5p | 1 | 0 | 0.477 | 0.477 | 0.378 | 0.814 | - | - | 0.014 | 0.000 |
| lichess_top5_full_2Xepisode | state_action_bem | specialization_5p | 1 | 0 | 0.224 | 0.224 | 0.211 | 0.680 | - | - | 0.006 | 0.000 |
| lichess_top5_full_2Xepisode | vqvae | conflation_5p | 1 | 0 | 0.654 | 0.654 | 0.522 | 0.840 | - | - | 0.162 | 4.251 |
| lichess_top5_full_2Xepisode | vqvae | generalization_5p | 1 | 0 | 0.271 | 0.271 | 0.233 | 0.697 | - | - | 0.063 | 5.461 |
| lichess_top5_full_2Xepisode | vqvae | new_policy_5p | 1 | 0 | 0.626 | 0.793 | 0.586 | 0.880 | - | - | 0.178 | 4.145 |
| lichess_top5_full_2Xepisode | vqvae | no_shift_5p | 1 | 0 | 0.788 | 0.788 | 0.684 | 0.904 | - | - | 0.183 | 4.008 |
| lichess_top5_full_2Xepisode | vqvae | novel_generalization_5p | 1 | 0 | 0.271 | 0.344 | 0.228 | 0.686 | - | - | 0.057 | 5.695 |
| lichess_top5_full_2Xepisode | vqvae | single_shift_5p | 1 | 0 | 0.703 | 0.703 | 0.652 | 0.888 | - | - | 0.169 | 4.095 |
| lichess_top5_full_2Xepisode | vqvae | specialization_5p | 1 | 0 | 0.417 | 0.417 | 0.310 | 0.761 | - | - | 0.116 | 4.486 |