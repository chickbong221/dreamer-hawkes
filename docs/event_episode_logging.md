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
| `event/spike_trace` | Image | `[1 x T]` strip, red where the detector fired |
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

`Agent.policy()` surfaces `haw_prob`, `haw_event` and `haw_prior_prob` in
`outs`, but only when `mode == 'eval'`. That gating matters: under
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

`event/spike_trace` will be **empty** whenever `pi_t` stays below
`haw_eval_threshold` (default `0.5`). That is expected while the event model sits
near the target rate `rho` (default `0.05`): eval thresholds deterministically
while training samples. Read `event/probs` and `event/expected_count` for the
event model's actual behavior, and lower `haw_eval_threshold` if you want visible
spikes.

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

## Dependencies

If `wandb` is missing or no run is active, the builder prints a message and
returns an empty payload.
