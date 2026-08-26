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
`event_delta_mag_img`, `haw_lam_mean`, `haw_lam_max`, `haw_state_mean`,
`haw_state_max`, `haw_ctx_std`, `haw_base`, `haw_alpha`, `haw_beta`,
`haw_valid_frac`.

**Eval event panels** (this module) show only what an aggregate cannot: *where
inside a single episode* events fire. Anything that would merely restate a
training metric is deliberately not logged here.

## Panels

| W&B key | Type | Content |
|---|---|---|
| `event/probs` | Line | `pi_t` over the episode |
| `event/spike_trace` | Image | `[1 x T]` strip, red where an event fired |
| `event/episode_video` | Video | mp4 of the episode with a bar above the frame, red on spike steps |
| `event/hard_count` | Scalar | events in the episode |
| `event/expected_count` | Scalar | `sum(pi_t)` |

`lambda_t` is not plotted: `pi = 1 - exp(-lambda)` is a monotone transform of
it, so it would be the same curve twice. The learned `b / alpha / beta` are
not plotted either — they are static parameters already logged as training
metrics.

`event/episode_video` uses whichever frames the observation carries: `image`
on RGB tasks (cameras flattened into channels are tiled horizontally) or
`depth_head` / `depth_hand` on mshab depth tasks. If neither is present the
other four panels still appear.

## Capture path

`Agent.policy()` surfaces `haw_prob` and `haw_event` in `outs`, but only when `mode == 'eval'`. That gating matters: under
`mode='train'` the driver merges `outs` into the transitions written to
replay, and `agent.train()` asserts the per-batch keys equal `self.spaces`
exactly, so extra fields there would break training. `mode` is a
`static_argnums` argument, so the branch is compiled away.

`embodied/run/train.py` pairs those two scalars with the `image` /
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

- **`event_prob` flat across the episode** means the event model has not
  localized anything — the rate budget is satisfied but no timestep is
  special. This is the constant-rate collapse mode; check
  `event_prob_std_time_obs` in the training metrics, and only then reach for a
  sharpness loss.
- **`event_rate_img` far from `event_rate_obs`** means the prior-to-prior
  latent delta is not in the same regime as the posterior-to-posterior one, so
  imagined event timing drifts from observed. Compare `event_delta_mag_img`
  against `event_delta_mag_obs`; the first intervention is disabling the
  latent-delta channel, not changing the Hawkes recurrence.
- **`haw_alpha` decaying to zero** means the Hawkes memory is not being used
  and `g_eta` is carrying the whole prediction.

## Dependencies

If `wandb` is missing or no run is active, the builder prints a message and
returns an empty payload.
