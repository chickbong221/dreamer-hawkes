# Event-Episode Logging

During each evaluation phase, the first episode of eval env 0 is captured and
its binary Hawkes events are logged to Weights & Biases under the `event/`
namespace, in the same W&B step as the rest of the eval metrics.

Enabled by `run.eval_event_log: True`, which the `hawkes` preset sets. The
capture is a no-op unless `dyn.typ == hawkes`.

## Division of labour

Two separate views, and it matters which one to read for what.

**Training metrics** (from `HawkesRSSM.loss()`, every `log_every` steps) carry
all the aggregate statistics. These are ordinary scalars, so W&B trends them
across the whole run:

`event_rate_obs`, `event_rate_error`, `event_hard_rate_obs`,
`event_prob_entropy_obs`, `event_prob_std_time_obs`, `event_delta_mag_obs`,
`event_rate_img`, `event_hard_rate_img`, `event_prob_entropy_img`,
`event_delta_mag_img`, `event_prob_std_time_img`, `haw_prior_rate`,
`haw_lam_mean`, `haw_lam_max`, `haw_mbar_std`, `haw_ctx_std`,
`haw_base`, `haw_alpha`, `haw_beta`, `haw_valid_frac`,
`haw_lam_fit_err`.

Event types add `event_type_entropy_sample`, `event_type_entropy_usage`,
`event_type_effective_count`, `event_type_max_occupancy`,
`event_type_min_occupancy`, `event_type_prior_kl`,
`event_type_prob_spread`, `event_type_prob_within_ratio`,
`event_type_usage_obs/<k>`, `event_type_usage_img/<k>`,
`event_type_prob_mean/<k>`, `event_type_detector_mag_mean/<k>`,
`event_type_reward_spread` and `event_type_cont_spread`.

Three more appear only in `report()` (`training=False`), because the
deployment probe costs a second context-network evaluation:
`haw_gap_teacher`, `haw_gap_deploy`, `haw_gap_memory`,
`haw_gap_memory_rel`, `haw_probe_rate`, `haw_delta_mag_prior`.

**Eval event panels** (this module) show only what an aggregate cannot: *where
inside a single episode* events fire. Anything that would merely restate a
training metric is deliberately not logged here.

## Panels

| W&B key | Type | Content |
|---|---|---|
| `event/probs` | Line | detector `q_t` against Hawkes `p_t` |
| `event/spike_trace` | Image | `[1 x T]` strip, colored by assigned type where the detector fired |
| `event/type_grid/<k>` | Image | event frames assigned to type `k`, pooled across every eval env and episode |
| `event/episode_video` | Video | mp4 of the episode with a bar above the frame, red on spike steps |
| `event/hard_count` | Scalar | events in the episode |
| `event/expected_count` | Scalar | `sum(q_t)` |
| `event/expected_count_prior` | Scalar | `sum(p_t)` |

`lambda_t` is not plotted: `pi = 1 - exp(-lambda)` is a monotone transform of
it, so it would be the same curve twice. The learned `b / alpha / beta` are
not plotted either — they are static parameters already logged as training
metrics.

`event/episode_video` uses whichever frames the observation carries: `image`
on RGB tasks (cameras flattened into channels are tiled horizontally) or
`depth_head` / `depth_hand` on mshab depth tasks. If neither is present the
other four panels still appear.

## Capture path

`Agent.policy()` surfaces `haw_prob`, `haw_event`, `haw_prior_prob` and
`haw_type_prob` in `outs`, but only when `mode == 'eval'`. That gating matters: under
`mode='train'` the driver merges `outs` into the transitions written to
replay, and `agent.train()` asserts the per-batch keys equal `self.spaces`
exactly, so extra fields there would break training. `mode` is a
`static_argnums` argument, so the branch is compiled away.

`embodied/run/train.py` pairs those with the `image` /
`depth_head` / `depth_hand` frames from the same steps.

Policy outputs are already host-side (`fetch_async`). Anything read from
`agent.params` is not, and `jax_transfer_guard` is set to `disallow`, so an
implicit `np.asarray` on a device array raises — use `jax.device_get`.

## Usage

```bash
python -m dreamerv3.main \
  --configs maniskill_rgb mshab hawkes \
  --task maniskill_PickSubtaskTrain-v0 \
  --env.maniskill.obs_mode rgb \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task tidy_house
```

`--env.maniskill.mshab_task` is required: it defaults to `none`, and the
`mshab` preset does not set it. Without it the env wrapper never imports
`mshab.envs`, so the `*SubtaskTrain-v0` ids are never registered and
`gym.make` raises `NameNotFound`.

Disable with `--run.eval_event_log False`.

## Reading the panels

Evaluation samples events exactly like training and imagination
(`_dyn_policy_kw` ignores `mode`), so `event/hard_count` should track
`event/expected_count` at roughly `rho * T`. `haw_eval_threshold` survives only
as an explicit diagnostic through `sample_event=False`; it is not used in
normal evaluation.

Cluster ids are permutation-dependent — they carry no meaning across runs or
even across restarts. Read the grouping in `event/type_grid/<k>`, never the
index. Frames in those grids are reservoir sampled among each type's events, so
the mean-confidence caption describes the type rather than its best examples.

Other things worth watching:

- **`detector_q` flat across the episode** means the detector has not
  localized anything — the rate budget is satisfied but no timestep is
  special. Check `event_prob_std_time_obs` in the training metrics.
- **`haw_gap_teacher` small but `haw_gap_deploy` large** is the teacher/
  deployment split: the Hawkes prior fits `q_t` using the posterior latent
  delta and cannot reproduce it from the prior delta alone. That is the
  headline risk of this design, and `event_rate_img` drifting from
  `event_rate_obs` is the same thing seen from imagination.
- **`haw_gap_memory_rel` near zero** means the Hawkes memory contributes
  nothing and `g_eta` carries the whole prediction. Read it together with
  `haw_alpha` and `haw_mbar_std`; `haw_alpha` alone can stay nonzero while
  the network simply ignores variation in `M`.
- **`haw_lam_fit_err` above ~1e-4** means the detached fitting recurrence no
  longer reproduces the live one — a misalignment in the reset mask, the event
  indexing, or the initial carry. See `TestFittingInvariant`.
- **`event_type_effective_count` near 1** is single-cluster collapse. Near `K`
  with indistinguishable grids is the opposite failure: `event_use` splitting
  identical events to fill the budget.
- **`event_type_prob_within_ratio` near 0** means the classifier is simply
  binning the detector probability — all the variation in `q` sits between
  clusters and none within them. Read it with `event_type_prob_spread`; a
  large spread alone is suggestive, the two together are decisive.
- **`event_type_reward_spread` and `event_type_cont_spread` near zero** mean
  the reward and continuation heads are reading the binary event and ignoring
  which type it was, so nothing is giving the classifier semantics.
- **`event_type_usage_img/<k>` far from `event_type_usage_obs/<k>`** means the
  type prior cannot reproduce the posterior from prior-only inputs; check
  `event_type_prior_kl`.

## Dependencies

If `wandb` is missing or no run is active, the builder prints a message and
returns an empty payload.
