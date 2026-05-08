# Aggregate (60 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `knn_acc1`/`knn_acc5` = leave-one-out cosine kNN policy accuracy on held-out eval embeddings; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | knn_acc@1 | knn_acc@5 | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dmlab_seekavoid_full | cvae_rnn | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.462±0.020 | 1.672±0.084 |
| dmlab_seekavoid_full | cvae_rnn | generalization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.440±0.015 | 1.731±0.055 |
| dmlab_seekavoid_full | cvae_rnn | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.512±0.011 | 1.505±0.033 |
| dmlab_seekavoid_full | cvae_rnn | single_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.459±0.021 | 1.711±0.070 |
| dmlab_seekavoid_full | cvae_rnn | specialization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.373±0.009 | 2.047±0.026 |
| dmlab_seekavoid_full | inr_diffusion_history_conditioned_shuffle | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.186±0.007 | 0.000±0.000 |
| dmlab_seekavoid_full | inr_diffusion_history_conditioned_shuffle | generalization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.140±0.000 | 0.000±0.000 |
| dmlab_seekavoid_full | inr_diffusion_history_conditioned_shuffle | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.187±0.002 | 0.000±0.000 |
| dmlab_seekavoid_full | inr_diffusion_history_conditioned_shuffle | single_shift | 2 | 0 | 0.972±0.028 | 0.972±0.028 | 1.000±0.000 | 1.000±0.000 | - | - | 0.180±0.006 | 0.000±0.000 |
| dmlab_seekavoid_full | inr_diffusion_history_conditioned_shuffle | specialization | 2 | 0 | 0.958±0.042 | 0.958±0.042 | 1.000±0.000 | 1.000±0.000 | - | - | 0.152±0.009 | 0.000±0.000 |
| dmlab_seekavoid_full | inr_transformer_fitted_latent_shuffle | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.455±0.022 | 1.713±0.075 |
| dmlab_seekavoid_full | inr_transformer_fitted_latent_shuffle | generalization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.400±0.006 | 1.837±0.040 |
| dmlab_seekavoid_full | inr_transformer_fitted_latent_shuffle | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.470±0.013 | 1.612±0.033 |
| dmlab_seekavoid_full | inr_transformer_fitted_latent_shuffle | single_shift | 2 | 0 | 0.972±0.028 | 0.972±0.028 | 1.000±0.000 | 1.000±0.000 | - | - | 0.420±0.017 | 1.793±0.056 |
| dmlab_seekavoid_full | inr_transformer_fitted_latent_shuffle | specialization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.317±0.013 | 2.177±0.029 |
| dmlab_seekavoid_full | inr_transformer_history_conditioned_shuffle | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.468±0.018 | 1.670±0.068 |
| dmlab_seekavoid_full | inr_transformer_history_conditioned_shuffle | generalization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.434±0.017 | 1.700±0.050 |
| dmlab_seekavoid_full | inr_transformer_history_conditioned_shuffle | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.520±0.013 | 1.503±0.034 |
| dmlab_seekavoid_full | inr_transformer_history_conditioned_shuffle | single_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.469±0.020 | 1.704±0.066 |
| dmlab_seekavoid_full | inr_transformer_history_conditioned_shuffle | specialization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.347±0.018 | 2.081±0.077 |
| dmlab_seekavoid_full | state_action_bem | conflation | 2 | 0 | 0.917±0.028 | 0.917±0.028 | 0.472±0.028 | 1.000±0.000 | - | - | 0.356±0.008 | 0.000±0.000 |
| dmlab_seekavoid_full | state_action_bem | generalization | 2 | 0 | 0.625±0.083 | 0.625±0.083 | 0.812±0.021 | 0.979±0.021 | - | - | 0.253±0.015 | 0.000±0.000 |
| dmlab_seekavoid_full | state_action_bem | no_shift | 2 | 0 | 0.938±0.021 | 0.938±0.021 | 0.771±0.062 | 0.958±0.000 | - | - | 0.317±0.011 | 0.000±0.000 |
| dmlab_seekavoid_full | state_action_bem | single_shift | 2 | 0 | 0.750±0.083 | 0.750±0.083 | 0.583±0.083 | 1.000±0.000 | - | - | 0.321±0.007 | 0.000±0.000 |
| dmlab_seekavoid_full | state_action_bem | specialization | 2 | 0 | 0.583±0.000 | 0.583±0.000 | 0.458±0.125 | 1.000±0.000 | - | - | 0.237±0.008 | 0.000±0.000 |
| dmlab_seekavoid_full | vqvae | conflation | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.462±0.016 | 1.659±0.076 |
| dmlab_seekavoid_full | vqvae | generalization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.416±0.016 | 1.741±0.037 |
| dmlab_seekavoid_full | vqvae | no_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.520±0.010 | 1.482±0.041 |
| dmlab_seekavoid_full | vqvae | single_shift | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.470±0.021 | 1.682±0.082 |
| dmlab_seekavoid_full | vqvae | specialization | 2 | 0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | - | - | 0.362±0.002 | 2.040±0.008 |