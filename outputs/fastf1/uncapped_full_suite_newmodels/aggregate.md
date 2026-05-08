# Aggregate (84 runs)

Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; `probe_acc_seen` = same probe restricted to training-policy labels; `knn_acc1`/`knn_acc5` = leave-one-out cosine kNN policy accuracy on held-out eval embeddings; `gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); `gen_median_se` = median per-sample squared error; `deg` = degenerate runs (non-finite gen or partial finite fraction).

| data | model | experiment | n | deg | probe_acc | probe_acc_seen | knn_acc@1 | knn_acc@5 | gen_nmse | gen_median_se | gen_acc | gen_nll |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastf1_stint_full_uncapped | cvae_rnn | conflation | 2 | 0 | 0.750±0.083 | 0.750±0.083 | 0.667±0.000 | 0.833±0.000 | 0.323±0.006 | 0.087±0.037 | - | - |
| fastf1_stint_full_uncapped | cvae_rnn | generalization | 2 | 0 | 0.556±0.000 | 0.556±0.000 | 0.389±0.056 | 1.000±0.000 | 0.452±0.077 | 0.068±0.017 | - | - |
| fastf1_stint_full_uncapped | cvae_rnn | new_policy | 2 | 0 | 0.357±0.000 | 0.556±0.000 | 0.250±0.036 | 0.786±0.071 | 0.310±0.031 | 0.041±0.013 | - | - |
| fastf1_stint_full_uncapped | cvae_rnn | no_shift | 2 | 0 | 0.556±0.000 | 0.556±0.000 | 0.333±0.111 | 1.000±0.000 | 0.318±0.058 | 0.039±0.013 | - | - |
| fastf1_stint_full_uncapped | cvae_rnn | novel_generalization | 2 | 0 | 0.357±0.000 | 0.556±0.000 | 0.321±0.107 | 0.893±0.036 | 0.513±0.013 | 0.095±0.004 | - | - |
| fastf1_stint_full_uncapped | cvae_rnn | single_shift | 2 | 0 | 0.500±0.167 | 0.500±0.167 | 0.583±0.083 | 0.833±0.000 | 0.369±0.045 | 0.062±0.015 | - | - |
| fastf1_stint_full_uncapped | cvae_rnn | specialization | 2 | 0 | 0.375±0.125 | 0.375±0.125 | 0.625±0.125 | 0.750±0.000 | 0.293±0.018 | 0.041±0.005 | - | - |
| fastf1_stint_full_uncapped | inr_diffusion_history_conditioned_shuffle | conflation | 2 | 0 | 0.750±0.083 | 0.750±0.083 | 0.750±0.083 | 0.833±0.000 | 1.170±0.029 | 0.305±0.138 | - | - |
| fastf1_stint_full_uncapped | inr_diffusion_history_conditioned_shuffle | generalization | 2 | 0 | 0.333±0.000 | 0.333±0.000 | 0.389±0.167 | 1.000±0.000 | 1.090±0.041 | 0.255±0.042 | - | - |
| fastf1_stint_full_uncapped | inr_diffusion_history_conditioned_shuffle | new_policy | 2 | 0 | 0.357±0.000 | 0.556±0.000 | 0.321±0.036 | 0.786±0.000 | 1.133±0.007 | 0.198±0.057 | - | - |
| fastf1_stint_full_uncapped | inr_diffusion_history_conditioned_shuffle | no_shift | 2 | 0 | 0.556±0.000 | 0.556±0.000 | 0.556±0.111 | 1.000±0.000 | 1.260±0.075 | 0.236±0.093 | - | - |
| fastf1_stint_full_uncapped | inr_diffusion_history_conditioned_shuffle | novel_generalization | 2 | 0 | 0.214±0.000 | 0.333±0.000 | 0.286±0.286 | 0.821±0.036 | 0.921±0.056 | 0.248±0.005 | - | - |
| fastf1_stint_full_uncapped | inr_diffusion_history_conditioned_shuffle | single_shift | 2 | 0 | 0.333±0.000 | 0.333±0.000 | 0.750±0.083 | 0.833±0.000 | 1.218±0.040 | 0.334±0.170 | - | - |
| fastf1_stint_full_uncapped | inr_diffusion_history_conditioned_shuffle | specialization | 2 | 0 | 0.250±0.000 | 0.250±0.000 | 0.500±0.000 | 0.750±0.000 | 1.159±0.007 | 0.374±0.152 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_fitted_latent_shuffle | conflation | 2 | 0 | 0.917±0.083 | 0.917±0.083 | 0.833±0.000 | 0.833±0.000 | 0.537±0.121 | 0.083±0.027 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_fitted_latent_shuffle | generalization | 2 | 0 | 0.833±0.056 | 0.833±0.056 | 0.889±0.111 | 1.000±0.000 | 0.426±0.096 | 0.067±0.017 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_fitted_latent_shuffle | new_policy | 2 | 0 | 0.607±0.036 | 0.944±0.056 | 0.429±0.143 | 0.821±0.036 | 0.407±0.051 | 0.054±0.004 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_fitted_latent_shuffle | no_shift | 2 | 0 | 0.944±0.056 | 0.944±0.056 | 0.500±0.056 | 1.000±0.000 | 0.403±0.106 | 0.046±0.008 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_fitted_latent_shuffle | novel_generalization | 2 | 0 | 0.536±0.036 | 0.833±0.056 | 0.464±0.107 | 0.964±0.036 | 0.480±0.100 | 0.090±0.013 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_fitted_latent_shuffle | single_shift | 2 | 0 | 0.917±0.083 | 0.917±0.083 | 0.750±0.083 | 0.833±0.000 | 0.477±0.092 | 0.053±0.008 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_fitted_latent_shuffle | specialization | 2 | 0 | 0.750±0.250 | 0.750±0.250 | 0.500±0.250 | 0.750±0.000 | 0.437±0.018 | 0.044±0.009 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_history_conditioned_shuffle | conflation | 2 | 0 | 0.833±0.000 | 0.833±0.000 | 0.667±0.000 | 0.833±0.000 | 0.362±0.033 | 0.072±0.037 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_history_conditioned_shuffle | generalization | 2 | 0 | 0.667±0.111 | 0.667±0.111 | 0.556±0.222 | 1.000±0.000 | 0.423±0.086 | 0.063±0.021 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_history_conditioned_shuffle | new_policy | 2 | 0 | 0.357±0.000 | 0.556±0.000 | 0.321±0.179 | 0.821±0.036 | 0.273±0.054 | 0.032±0.010 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_history_conditioned_shuffle | no_shift | 2 | 0 | 0.556±0.000 | 0.556±0.000 | 0.389±0.167 | 1.000±0.000 | 0.315±0.043 | 0.042±0.012 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_history_conditioned_shuffle | novel_generalization | 2 | 0 | 0.429±0.071 | 0.667±0.111 | 0.393±0.107 | 1.000±0.000 | 0.404±0.098 | 0.083±0.012 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_history_conditioned_shuffle | single_shift | 2 | 0 | 0.583±0.083 | 0.583±0.083 | 0.667±0.167 | 0.833±0.000 | 0.377±0.007 | 0.051±0.016 | - | - |
| fastf1_stint_full_uncapped | inr_transformer_history_conditioned_shuffle | specialization | 2 | 0 | 0.375±0.125 | 0.375±0.125 | 0.500±0.000 | 0.750±0.000 | 0.294±0.031 | 0.039±0.021 | - | - |
| fastf1_stint_full_uncapped | state_action_bem | conflation | 2 | 0 | 0.500±0.333 | 0.500±0.333 | 0.667±0.000 | 0.833±0.000 | 0.218±0.100 | 0.066±0.005 | - | - |
| fastf1_stint_full_uncapped | state_action_bem | generalization | 2 | 0 | 0.389±0.056 | 0.389±0.056 | 0.444±0.111 | 1.000±0.000 | 0.177±0.039 | 0.066±0.011 | - | - |
| fastf1_stint_full_uncapped | state_action_bem | new_policy | 2 | 0 | 0.286±0.071 | 0.444±0.111 | 0.357±0.143 | 0.821±0.036 | 0.153±0.002 | 0.048±0.002 | - | - |
| fastf1_stint_full_uncapped | state_action_bem | no_shift | 2 | 0 | 0.444±0.111 | 0.444±0.111 | 0.389±0.167 | 1.000±0.000 | 0.184±0.037 | 0.054±0.001 | - | - |
| fastf1_stint_full_uncapped | state_action_bem | novel_generalization | 2 | 0 | 0.250±0.036 | 0.389±0.056 | 0.321±0.036 | 0.821±0.107 | 0.169±0.019 | 0.078±0.002 | - | - |
| fastf1_stint_full_uncapped | state_action_bem | single_shift | 2 | 0 | 0.333±0.000 | 0.333±0.000 | 0.750±0.083 | 0.833±0.000 | 0.227±0.098 | 0.059±0.003 | - | - |
| fastf1_stint_full_uncapped | state_action_bem | specialization | 2 | 0 | 0.375±0.125 | 0.375±0.125 | 0.250±0.250 | 0.750±0.000 | 0.320±0.016 | 0.060±0.001 | - | - |
| fastf1_stint_full_uncapped | vqvae | conflation | 2 | 0 | 0.833±0.167 | 0.833±0.167 | 0.667±0.000 | 0.833±0.000 | 0.359±0.005 | 0.074±0.012 | - | - |
| fastf1_stint_full_uncapped | vqvae | generalization | 2 | 0 | 0.667±0.000 | 0.667±0.000 | 0.389±0.167 | 0.889±0.111 | 0.438±0.097 | 0.080±0.024 | - | - |
| fastf1_stint_full_uncapped | vqvae | new_policy | 2 | 0 | 0.393±0.036 | 0.611±0.056 | 0.250±0.036 | 0.893±0.036 | 0.299±0.055 | 0.041±0.019 | - | - |
| fastf1_stint_full_uncapped | vqvae | no_shift | 2 | 0 | 0.611±0.056 | 0.611±0.056 | 0.389±0.167 | 1.000±0.000 | 0.324±0.074 | 0.043±0.017 | - | - |
| fastf1_stint_full_uncapped | vqvae | novel_generalization | 2 | 0 | 0.429±0.000 | 0.667±0.000 | 0.393±0.107 | 0.786±0.143 | 0.565±0.016 | 0.109±0.008 | - | - |
| fastf1_stint_full_uncapped | vqvae | single_shift | 2 | 0 | 0.667±0.167 | 0.667±0.167 | 0.750±0.083 | 0.833±0.000 | 0.378±0.032 | 0.057±0.015 | - | - |
| fastf1_stint_full_uncapped | vqvae | specialization | 2 | 0 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.750±0.000 | 0.305±0.007 | 0.049±0.011 | - | - |