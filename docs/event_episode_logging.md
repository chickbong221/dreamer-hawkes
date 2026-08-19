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

`haw_rate`, `haw_rate_error`, `haw_event_rate`, `haw_prior_rate`,
`haw_post_prior_gap`, `haw_prob_ent`, `haw_prob_std_time`, `haw_lam_mean`,
`haw_lam_max`, `haw_ctx_std`, `haw_base`, `haw_alpha`, `haw_beta`,
`haw_valid_frac`, `haw_lam_fit_err`.

**Eval event panels** (this module) show only what an aggregate cannot: *where
inside a single episode* events fire. Anything that would merely restate a
training metric is deliberately not logged here.

## Panels

| W&B key | Type | Content |
|---|---|---|
| `event/probs` | Line | `pi_t` and `p_haw_t` over the episode, one chart |
| `event/spike_trace` | Image | `[1 x T]` strip, red where an event fired |
| `event/episode_video` | Video | `depth_head \| depth_hand` mp4, red top border on spike frames |
| `event/hard_count` | Scalar | events in the episode |
| `event/expected_count` | Scalar | `sum(pi_t)` |

`lambda_t` is not plotted: `p_haw = 1 - exp(-lambda)` is a monotone transform
of it, so it would be the same curve twice. The learned `b / alpha / beta` are
not plotted either — they are static parameters already logged as training
metrics.

`event/episode_video` appears only when the observation carries `depth_head`,
which means mshab depth tasks. On plain ManiSkill RGB tasks the other four
panels still appear.

## Capture path

`Agent.policy()` surfaces `haw_prob`, `haw_event` and `haw_prior_prob` in
`outs`, but only when `mode == 'eval'`. That gating matters: under
`mode='train'` the driver merges `outs` into the transitions written to
replay, and `agent.train()` asserts the per-batch keys equal `self.spaces`
exactly, so extra fields there would break training. `mode` is a
`static_argnums` argument, so the branch is compiled away.

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
`haw_eval_threshold` (default `0.5`). That is expected while the detector sits
near the target rate `rho` (default `0.05`): eval thresholds deterministically
while training samples. Read `event/probs` and `event/expected_count` for the
detector's actual behavior, and lower `haw_eval_threshold` if you want visible
spikes.

Other things worth watching:

- **`post_prob` flat across the episode** means the detector has not localized
  anything — the rate budget is satisfied but no timestep is special. Check
  `haw_prob_std_time` in the training metrics before reaching for a sharpness
  loss.
- **`post_prob` and `prior_prob` far apart** means the causal prior cannot
  predict the detector, so imagined event timing will be near-random. The
  prior only sees `(M_t, h_{t-1}, a_{t-1})`, so a persistent gap suggests the
  detector is firing on something not visible in the deterministic state.
- **`haw_lam_fit_err` above ~1e-4** means the detached fitting recurrence in
  `loss()` no longer reproduces the live one — a misalignment in the shifted
  event/deter sequences, the reset mask, or the initial carry. See
  `TestFittingInvariant`.

## Dependencies

If `wandb` is missing or no run is active, the builder prints a message and
returns an empty payload.
