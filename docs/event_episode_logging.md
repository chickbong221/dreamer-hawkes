# Event-Episode Logging

During each evaluation phase, the first episode of eval env 0 is captured and
its binary Hawkes events are logged to Weights & Biases under the `event/`
namespace, in the same W&B step as the rest of the eval metrics.

Enabled by `run.eval_event_log: True`, which the `hawkes` preset sets. The
capture is a no-op unless `dyn.typ == hawkes`.

## What is captured

`Agent.policy()` surfaces four per-step scalars in `outs`, but only when
`mode == 'eval'`. That gating matters: under `mode='train'` the driver merges
`outs` into the transitions written to replay, and `agent.train()` asserts the
per-batch keys equal `self.spaces` exactly, so extra fields there would break
training. `mode` is a `static_argnums` argument, so the branch is compiled away.

| Key | Meaning |
|---|---|
| `haw_prob` | posterior event probability `pi_t` |
| `haw_event` | hard event `y_t` (thresholded at `haw_eval_threshold` in eval) |
| `haw_prior_prob` | causal Hawkes probability `p_haw_t` |
| `haw_lam` | scalar intensity `lambda_t` |

`embodied/run/train.py` appends these plus `reward`, `action` and any
`depth_head` / `depth_hand` frames, then hands the arrays to
`dreamerv3.log_event_episode.build_event_episode_payload`.

## Panels

| W&B key | Type | Content |
|---|---|---|
| `event/post_prob` | Line | `pi_t` over time |
| `event/prior_prob` | Line | `p_haw_t` over time |
| `event/lambda` | Line | scalar intensity over time |
| `event/post_prior_gap` | Line | `\|pi_t - p_haw_t\|` over time |
| `event/reward_trace` | Line | reward over time |
| `event/spike_trace` | Image | `[1 x T]` strip, bright where an event fired |
| `event/episode_video` | Video | `depth_head \| depth_hand` mp4, red top border on spike frames |
| `event/events` | Table | one row per spike: t, `pi_t`, `p_haw_t`, `lambda_t`, reward, action, thumbnails |
| `event/expected_rate` | Scalar | `mean(pi_t)` |
| `event/hard_rate` | Scalar | fraction of steps with an event |
| `event/rate_error` | Scalar | `hard_rate - expected_rate` |
| `event/expected_count` | Scalar | `sum(pi_t)` |
| `event/hard_count` | Scalar | number of spikes |
| `event/base`, `event/alpha`, `event/beta` | Scalar | learned Hawkes parameters |
| `event/base_prob` | Scalar | `1 - exp(-softplus(b))`, the baseline event probability |

The Hawkes scalars are read from `agent.params` by suffix-matching
`haw_base`, `haw_alpha_raw`, `haw_beta_raw` and applying the same `softplus`
as `hawkes_rssm.py:_haw_params`. If those names change, only these three
panels disappear; the per-step traces keep working since they come from the
model's own forward pass.

## Usage

```bash
python -m dreamerv3.main --configs mshab hawkes --task maniskill_PickSubtaskTrain-v0
```

Disable with `--run.eval_event_log False`.

## Reading the panels

`event/spike_trace` will be **empty** whenever `pi_t` stays below
`haw_eval_threshold` (default `0.5`). That is expected while the detector sits
near the target rate `rho` (default `0.05`): eval thresholds deterministically
while training samples. Read `event/post_prob` for the detector's actual
behavior, and lower `haw_eval_threshold` if you want visible spikes.

Other things worth watching:

- **`event/post_prob` flat at `rho`** means the detector has not localized
  anything — the rate budget is satisfied but no timestep is special. Check
  `haw_prob_std_time` in the training metrics before reaching for a sharpness
  loss.
- **`event/post_prior_gap` large** means the causal prior cannot predict the
  detector, so imagined event timing will be near-random. The prior only sees
  `(M_t, h_{t-1}, a_{t-1})`, so a persistent gap suggests the detector is
  firing on something not visible in the deterministic state.
- **`event/hard_rate` far from `event/expected_rate`** over a whole episode
  points at a sampling or masking bug, not a modelling one.

## Related training metrics

`HawkesRSSM.loss()` logs `haw_rate`, `haw_rate_error`, `haw_event_rate`,
`haw_prior_rate`, `haw_post_prior_gap`, `haw_prob_ent`, `haw_prob_std_time`,
`haw_lam_mean`, `haw_lam_max`, `haw_ctx_std`, `haw_base`, `haw_alpha`,
`haw_beta`, `haw_valid_frac`, and `haw_lam_fit_err`.

`haw_lam_fit_err` is a live guard: the detached fitting recurrence in `loss()`
must reproduce the live recurrence's forward values exactly. Anything above
~1e-4 means a misalignment in the shifted event/deter sequences, the reset
mask, or the initial carry — see `TestFittingInvariant`.

## Dependencies

Matplotlib is optional (colormaps fall back to a hand-rolled gradient). If
`wandb` is missing or no run is active, the builder prints a message and
returns an empty payload.
