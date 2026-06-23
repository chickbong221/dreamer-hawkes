# Post-Training Event-Episode Logging

This document describes a feature that, **after training completes**, runs one
fresh episode through the trained Hawkes-RSSM agent and logs the per-timestep
event posterior + input frames + per-phase breakdown to Weights & Biases under
the `event/` namespace.

## What it does

When `dyn.typ == hawkes` and `run.post_train_event_log == True`, the training
script runs an extra step right after `cp.save()`:

1. Builds a **fresh single-env wrapper** (`make_env(0, num_envs=1, is_eval=False)`)
   — identical configuration to the training envs but isolated so it does not
   disturb the live training driver state.
2. Initialises a fresh policy carry, resets the env, and rolls one episode out
   to either `event_log_max_steps` or `env.maniskill.max_episode_steps`
   (default 100).
3. At every non-terminal step, it captures:
   - `feat['haw_logit']`  → softmax → `post_probs[t, K]`
   - `feat['haw_lam']`    → Hawkes intensities `λ[t, K]`
   - `depth_head[t]`, `depth_hand[t]` (uint16 mm) frames for env 0
   - The chosen action and the reward
4. Builds and pushes the following panels to W&B under `event/`:

| W&B key | Type | Content |
|---|---|---|
| `event/post_probs` | Image | `[K × T]` heatmap of posterior P(e_t=k) |
| `event/lambda` | Image | `[K × T]` heatmap of Hawkes intensity λ_k(t), row-normalized |
| `event/argmax_trace` | Image | `[1 × T]` strip, color = `argmax_k post_probs[t, k]` |
| `event/argmax_per_step` | Line | Argmax category id over time |
| `event/reward_trace` | Line | Reward over time |
| `event/episode_video` | Video | `depth_head | depth_hand` mp4, top border colored by current argmax category |
| `event/phases` | Table | One row per "phase" (contiguous run of constant argmax): category, start, end, length, mean probability, head + hand thumbnail at the phase midpoint |
| `event/alpha` | Image | `[K × K]` heatmap of learned `α[k, j]` (signed, diverging: red excite / blue inhibit) |
| `event/beta` | Image | `[K × K]` heatmap of learned `β[k, j]` (positive decay) |
| `event/mu` | Bar | `μ_k` baseline intensity per category |

The static `α / β / μ` are extracted from `agent.params` by suffix-matching
the names `haw_alpha`, `haw_beta_raw`, `haw_mu_raw` and applying the same
`softplus` post-processing as `hawkes_rssm.py:_hawkes_params`.

## Files changed

| File | Change |
|---|---|
| `dreamerv3/agent.py` | In `Agent.policy()`, when `config.dyn.typ == 'hawkes'` **and `mode == 'eval'`**, also surface `feat['haw_logit']` and `feat['haw_lam']` in the `outs` dict so the rollout loop can read them per step. Gating on `mode='eval'` is critical: the train-time driver merges `outs` into transitions written to replay, and `agent.train()` asserts the per-batch dict keys exactly equal `self.spaces`. Adding fields under `mode='train'` would break training. `mode` is already a `static_argnums` argument to the JAX policy, so this gating is compiled — there is no runtime overhead and no shared compilation between train and eval paths. |
| `dreamerv3/log_event_episode.py` | **New file.** All logging logic: rollout loop, phase detection, depth → RGB, colormaps (with matplotlib fallback), Hawkes-param fetch, and the `wandb.log` payload. |
| `embodied/run/train.py` | After `cp.save()`, if `args.post_train_event_log` is True, dynamically imports `dreamerv3.log_event_episode.log_event_episode` and runs it. Guarded so a failure cannot prevent the normal shutdown path. |
| `dreamerv3/configs.yaml` | Added `run.post_train_event_log: False` and `run.event_log_max_steps: 0` to the defaults; set `post_train_event_log: True` in both `hawkes` and `hawkes_supervised` presets so combining them with a task preset enables the feature automatically. |

## How to use

Anywhere you would normally combine the `hawkes` (or `hawkes_supervised`)
preset with a task preset, the feature is now on by default — no further
flags needed:

```bash
python -m dreamerv3.main \
    --configs mshab hawkes \
    --task maniskill_PickSubtaskTrain-v0 \
    ...
```

When training reaches its `steps` budget and `cp.save()` runs, the post-train
hook fires once and you should see new panels appear under `event/...` in the
W&B run.

To disable without removing the preset:

```bash
... --run.post_train_event_log False
```

To override the rollout length:

```bash
... --run.event_log_max_steps 200
```

## Design choices and caveats

- **Why a fresh 1-env wrapper?** The training `driver` owns the live vector env
  and is mid-shutdown; reusing its env would require coordinating reset state.
  A fresh `make_env(0, num_envs=1, is_eval=False)` gives a deterministic,
  isolated episode that behaves identically to env 0 of training and adds
  ≪1 % wall-clock to the run.
- **One episode only.** Per the user's spec. To collect more episodes, loop
  the call inside `log_event_episode` — but be aware W&B will overwrite
  same-key images on each call unless you suffix them.
- **Terminal frame is omitted.** The model's posterior is not computed at the
  terminal step (policy isn't queried there), so the frame and features arrays
  always have the same length `T = episode_length − 1` to keep the heatmaps
  aligned with the video.
- **Color mismatches across panels.** All "category id → color" panels
  (`argmax_trace`, video border, phase table) share the same `tab20` palette
  so categories are visually consistent.
- **Matplotlib optional.** Heatmap rendering tries `matplotlib.cm` first and
  falls back to a hand-rolled gradient if matplotlib is missing — no extra
  dependency is required to make the feature run.
- **W&B optional.** If `wandb` is not installed or no run is active, the hook
  prints a message and returns cleanly without touching the env.
- **Hawkes params are read from `agent.params` by name-suffix matching.** If
  the parameter naming convention in `hawkes_rssm.py:_hawkes_params` changes,
  the static `α / β / μ` panels will silently disappear — the per-step
  `post_probs / lambda` panels will continue to work since they go through
  the model's own forward pass.

## Code map

```
embodied/run/train.py
    └── after cp.save():
        └── from dreamerv3.log_event_episode import log_event_episode
            └── log_event_episode(agent, make_env, step, max_steps)
                ├── make_env(0, num_envs=1, is_eval=False)
                ├── _rollout(agent, env, max_steps)
                │   └── per step:
                │       ├── obs = env.step(acts)
                │       ├── strip log/ keys
                │       └── agent.policy(carry, obs, mode='eval')
                │           └── returns outs containing
                │               'haw_logit' and 'haw_lam'
                │               (added in dreamerv3/agent.py:policy)
                ├── _viridis / _diverging / _category_palette
                ├── _detect_phases
                ├── _depth_to_rgb
                ├── _fetch_hawkes_params(agent)
                └── wandb.log({...}, step=step)
```

## Verifying

After a run finishes, search the run's W&B page for keys starting with `event/`.
You should see:

- **`event/post_probs`**: a tall thin heatmap. Vertical bands indicate stable
  categorical assignment over many steps; speckled patterns indicate the model
  is bouncing between categories.
- **`event/episode_video`**: a depth-camera mp4 with a colored top border. The
  border color changes at phase boundaries, making it easy to scrub to the
  frame where (for example) the agent transitions from "approach" to "grasp"
  if the categories ended up corresponding to those phases.
- **`event/phases`**: a tabular summary you can sort by category to see how
  many phases of each type appeared, and click thumbnails for a visual.
- **`event/alpha`**: red off-diagonal cells reveal which categories trigger
  which (e.g. category 3 firing makes category 7 more likely soon after).
  A bright red diagonal means strong self-excitation → long bursts of the
  same category.
