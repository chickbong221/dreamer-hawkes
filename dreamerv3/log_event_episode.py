"""
Post-training event-episode logger.

Rolls out one fresh episode through the trained Hawkes agent on a single
training-style env, captures the per-step Hawkes posterior over K event
categories alongside the raw input frames, and logs everything to W&B under
the `event/` namespace.

Expected runtime: ~1 episode = `max_episode_steps` steps on a 1-wide
ManiSkillVectorEnv. Compute is negligible compared to a training run.

Outputs to wandb (all under `event/`):
  event/post_probs        — [K, T] heatmap of softmax posterior per step
  event/lambda            — [K, T] heatmap of Hawkes intensities per step
  event/argmax_trace      — [1, T] colored strip, color = argmax category id
  event/argmax_per_step   — line plot of argmax category id over time
  event/reward_trace      — line plot of reward over time
  event/episode_video     — [T, H, 2W, 3] depth_head | depth_hand video,
                            with a colored top border keyed to argmax category
  event/phases            — wandb.Table: phase index, category, start, end,
                            length, mean probability, head/hand thumbnails
  event/alpha             — [K, K] heatmap of learned alpha (signed, diverging)
  event/beta              — [K, K] heatmap of learned beta (positive)
  event/mu                — [K] bar chart of learned mu
"""

import io
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def _viridis(values):
  """Apply a viridis-like colormap. Input [..] float in [0,1], output [..,3] u8.

  Falls back to a hand-rolled approximation if matplotlib is unavailable.
  """
  values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
  try:
    import matplotlib.cm as cm
    rgba = cm.viridis(values)
    return (rgba[..., :3] * 255).astype(np.uint8)
  except Exception:
    # Crude hand-rolled gradient: dark blue → green → yellow.
    r = np.clip(2 * values - 0.5, 0, 1)
    g = np.clip(2 * values * (1 - values) * 4, 0, 1)
    b = np.clip(1 - 1.5 * values, 0, 1)
    out = np.stack([r, g, b], axis=-1)
    return (out * 255).astype(np.uint8)


def _diverging(values):
  """Diverging colormap for signed values, centered at 0. Output [..,3] u8."""
  values = np.asarray(values, dtype=np.float32)
  scale = max(float(np.abs(values).max()), 1e-6)
  values = np.clip(values / scale, -1.0, 1.0)
  try:
    import matplotlib.cm as cm
    rgba = cm.coolwarm((values + 1.0) * 0.5)
    return (rgba[..., :3] * 255).astype(np.uint8)
  except Exception:
    pos = np.clip(values, 0, 1)
    neg = np.clip(-values, 0, 1)
    out = np.stack([pos + (1 - pos - neg), 1 - pos - neg, neg + (1 - pos - neg)],
                   axis=-1)
    return (out * 255).clip(0, 255).astype(np.uint8)


def _category_palette(K):
  """Return [K, 3] uint8 RGB palette via the tab20 colormap."""
  try:
    import matplotlib.cm as cm
    rgba = cm.tab20(np.arange(K) / max(K - 1, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)
  except Exception:
    rng = np.random.RandomState(0)
    return (rng.rand(K, 3) * 255).astype(np.uint8)


def _upscale(image, target_h, target_w):
  """Nearest-neighbour upscale a 2D or 3D image to (target_h, target_w[, C])."""
  h, w = image.shape[:2]
  rh = max(target_h // max(h, 1), 1)
  rw = max(target_w // max(w, 1), 1)
  out = np.repeat(np.repeat(image, rh, axis=0), rw, axis=1)
  return out


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

def _detect_phases(argmax):
  """Return list of (cat, start, end_exclusive) for contiguous argmax runs."""
  argmax = np.asarray(argmax)
  if len(argmax) == 0:
    return []
  boundaries = np.where(np.diff(argmax, prepend=argmax[0] - 1) != 0)[0]
  ends = np.append(boundaries[1:], len(argmax))
  return [(int(argmax[s]), int(s), int(e)) for s, e in zip(boundaries, ends)]


# ---------------------------------------------------------------------------
# Depth → RGB
# ---------------------------------------------------------------------------

def _depth_to_rgb(depth_u16, max_depth):
  """Convert depth uint16 mm [H, W, 1] to RGB uint8 [H, W, 3]."""
  depth = np.asarray(depth_u16, dtype=np.float32).squeeze(-1)
  depth = np.clip(depth, 0, max_depth) / max(max_depth, 1.0)
  gray = (depth * 255).astype(np.uint8)
  return np.stack([gray, gray, gray], axis=-1)


# ---------------------------------------------------------------------------
# Hawkes static param fetch
# ---------------------------------------------------------------------------

def _fetch_hawkes_params(agent):
  """Pull mu, alpha, beta from agent.params using softplus on raw scalars.

  Matches the math in hawkes_rssm.py:_hawkes_params. Returns numpy arrays.
  Returns (None, None, None) if the params are not present (e.g. dyn.typ != hawkes).
  """
  try:
    params = agent.params
  except Exception:
    return None, None, None

  def _find(name):
    for k, v in params.items():
      if k.endswith(name):
        return np.asarray(v)
    return None

  mu_raw = _find('haw_mu_raw')
  alpha = _find('haw_alpha')
  beta_raw = _find('haw_beta_raw')
  if mu_raw is None or alpha is None or beta_raw is None:
    return None, None, None

  cfg = agent.config.dyn.hawkes
  init_mu = float(cfg.get('haw_init_mu', 0.1))
  init_beta = float(cfg.get('haw_init_beta', 1.0))

  def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

  mu = softplus(mu_raw + init_mu)
  beta = softplus(beta_raw + init_beta)
  return mu, alpha, beta


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _rollout(agent, env, max_steps):
  """Run one episode with N=1 envs; capture per-step features and obs.

  Returns a dict of numpy arrays keyed by step (concatenated along time).
  """
  obs_keys = list(env.obs_space.keys())
  num_envs = env.num_envs
  assert num_envs == 1, f'Expected single-env wrapper, got {num_envs}'

  acts = {k: np.zeros((num_envs,) + v.shape, v.dtype)
          for k, v in env.act_space.items()}
  acts['reset'] = np.ones(num_envs, bool)

  carry = agent.init_policy(num_envs)
  buf = defaultdict(list)

  # Per-iteration we capture frame + features TOGETHER, so every list in
  # buf has the same length. On the terminal step the agent is not queried
  # (episode is over), so we just break — that step is omitted.
  for t in range(max_steps + 2):
    obs = env.step(acts)
    obs = {k: np.asarray(v) for k, v in obs.items()}

    if t > 0 and bool(obs['is_last'][0]):
      break

    obs_for_policy = {k: v for k, v in obs.items() if not k.startswith('log/')}
    carry, acts_out, outs = agent.policy(carry, obs_for_policy, mode='eval')

    for k in ('depth_head', 'depth_hand'):
      if k in obs:
        buf[k].append(obs[k][0].copy())
    if 'haw_logit' in outs:
      buf['haw_logit'].append(np.asarray(outs['haw_logit'])[0].copy())
      buf['haw_lam'].append(np.asarray(outs['haw_lam'])[0].copy())
    buf['action'].append(np.asarray(acts_out['action'])[0].copy())
    buf['reward'].append(float(obs['reward'][0]))

    acts = {**acts_out, 'reset': np.zeros(num_envs, bool)}
  else:
    print(f'[event-episode] Warning: episode did not finish within '
          f'{max_steps + 2} steps')

  return {k: np.asarray(v) for k, v in buf.items()}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def log_event_episode(agent, make_env, step, max_steps=100, max_depth=None):
  """Roll out one episode, log per-step events + frames + phases to wandb.

  Args:
    agent: trained dreamerv3 Agent (Hawkes dynamics required).
    make_env: bound make_env(index, **overrides) — creates a fresh env.
    step: current global training step, used as the wandb log step.
    max_steps: episode horizon to allocate.
    max_depth: depth max value (mm) for u16 → u8 conversion. If None, read
      from env._max_depth (the actual configured value).
  """
  if getattr(agent, 'config', None) is None or \
     agent.config.dyn.typ != 'hawkes':
    print('[event-episode] dyn.typ != hawkes — skipping.')
    return

  try:
    import wandb
  except ImportError:
    print('[event-episode] wandb not installed — skipping.')
    return
  if wandb.run is None:
    print('[event-episode] no active wandb run — skipping.')
    return

  print('[event-episode] Building fresh 1-env wrapper for rollout...')
  env = make_env(0, num_envs=1, is_eval=False)
  if max_depth is None:
    max_depth = float(getattr(env, '_max_depth', 20000.0))
  try:
    data = _rollout(agent, env, max_steps)
  finally:
    try:
      env.close()
    except Exception as exc:
      print(f'[event-episode] env.close raised: {exc}')

  if 'haw_logit' not in data or len(data['haw_logit']) == 0:
    print('[event-episode] No Hawkes features captured — skipping.')
    return

  haw_logit = data['haw_logit']                       # [T, K]
  haw_lam = data['haw_lam']                           # [T, K]
  post_probs = _softmax(haw_logit, axis=-1)           # [T, K]
  argmax = post_probs.argmax(-1)                      # [T]
  T, K = post_probs.shape
  reward = data['reward'][:T]                         # align: last step may be terminal

  # ---- Heatmaps (K rows × T cols, displayed as wandb images) ----------------
  payload = {}

  # post_probs heatmap: shape [K, T], normalized per-cell to [0, 1].
  pp_img = post_probs.T                                       # [K, T]
  pp_rgb = _viridis(pp_img)                                   # [K, T, 3]
  pp_rgb = _upscale(pp_rgb, target_h=K * 16, target_w=T * 6)
  payload['event/post_probs'] = wandb.Image(
      pp_rgb, caption=f'posterior P(e_t=k) — rows=K(0..{K-1}), cols=t(0..{T-1})')

  # lambda heatmap: normalize per-row (per category) so categories with
  # different scales are comparable.
  lam_norm = haw_lam / np.maximum(haw_lam.max(0, keepdims=True), 1e-6)
  lam_rgb = _viridis(lam_norm.T)
  lam_rgb = _upscale(lam_rgb, target_h=K * 16, target_w=T * 6)
  payload['event/lambda'] = wandb.Image(
      lam_rgb, caption='Hawkes intensity λ_k(t), normalized per row')

  # argmax strip: colored band, one column per timestep, color = category.
  palette = _category_palette(K)
  strip = palette[argmax]                                     # [T, 3]
  strip = strip[None, :, :]                                   # [1, T, 3]
  strip = _upscale(strip, target_h=24, target_w=T * 6)
  payload['event/argmax_trace'] = wandb.Image(
      strip, caption='argmax category per timestep (color = category id)')

  # argmax / reward line plots
  table_trace = wandb.Table(
      data=[[t, int(argmax[t]), float(reward[t])] for t in range(T)],
      columns=['t', 'argmax_category', 'reward'])
  payload['event/argmax_per_step'] = wandb.plot.line(
      table_trace, 't', 'argmax_category',
      title='Argmax event category per timestep')
  payload['event/reward_trace'] = wandb.plot.line(
      table_trace, 't', 'reward', title='Reward per timestep')

  # ---- Per-phase table ------------------------------------------------------
  phases = _detect_phases(argmax)
  has_head = 'depth_head' in data
  has_hand = 'depth_hand' in data
  cols = ['phase', 'category', 'start', 'end', 'length', 'mean_prob']
  if has_head:
    cols.append('head_repr')
  if has_hand:
    cols.append('hand_repr')
  rows = []
  for idx, (cat, s, e) in enumerate(phases):
    mid = (s + e) // 2
    mid = min(mid, T - 1)
    mean_p = float(post_probs[s:e, cat].mean())
    row = [idx, int(cat), int(s), int(e), int(e - s), round(mean_p, 4)]
    if has_head:
      head_rgb = _depth_to_rgb(data['depth_head'][mid], max_depth)
      row.append(wandb.Image(head_rgb, caption=f'phase {idx} cat {cat} t={mid}'))
    if has_hand:
      hand_rgb = _depth_to_rgb(data['depth_hand'][mid], max_depth)
      row.append(wandb.Image(hand_rgb, caption=f'phase {idx} cat {cat} t={mid}'))
    rows.append(row)
  payload['event/phases'] = wandb.Table(data=rows, columns=cols)

  # ---- Episode video with phase-coloured border -----------------------------
  if has_head:
    head = data['depth_head'][:T]
    hand = data['depth_hand'][:T] if has_hand else None
    head_rgb = np.stack([_depth_to_rgb(f, max_depth) for f in head], 0)  # [T,H,W,3]
    if hand is not None:
      hand_rgb = np.stack([_depth_to_rgb(f, max_depth) for f in hand], 0)
      frames = np.concatenate([head_rgb, hand_rgb], axis=2)              # [T,H,2W,3]
    else:
      frames = head_rgb
    # Stamp a 6-px top border in the argmax category's colour.
    border_h = 6
    frames = np.concatenate(
        [np.zeros((T, border_h, frames.shape[2], 3), np.uint8), frames], axis=1)
    for t in range(T):
      frames[t, :border_h] = palette[argmax[t]]
    # wandb.Video expects [T, C, H, W]
    payload['event/episode_video'] = wandb.Video(
        frames.transpose(0, 3, 1, 2), fps=10, format='mp4',
        caption='depth_head | depth_hand; top border = argmax category')

  # ---- Static Hawkes parameters --------------------------------------------
  mu, alpha, beta = _fetch_hawkes_params(agent)
  if alpha is not None:
    payload['event/alpha'] = wandb.Image(
        _upscale(_diverging(alpha), K * 24, K * 24),
        caption='alpha[k,j] — excitation from j to k (red=excite, blue=inhibit)')
  if beta is not None:
    bmax = float(beta.max())
    beta_norm = beta / max(bmax, 1e-6)
    payload['event/beta'] = wandb.Image(
        _upscale(_viridis(beta_norm), K * 24, K * 24),
        caption=f'beta[k,j] — decay (max={bmax:.3f})')
  if mu is not None:
    mu_table = wandb.Table(
        data=[[k, float(mu[k])] for k in range(K)],
        columns=['k', 'mu'])
    payload['event/mu'] = wandb.plot.bar(
        mu_table, 'k', 'mu', title='mu_k — baseline intensity per category')

  # ---- Push to wandb --------------------------------------------------------
  try:
    wandb.log(payload, step=int(step))
    print(f'[event-episode] Logged {len(payload)} panels to wandb '
          f'(T={T}, K={K}, phases={len(phases)}).')
  except Exception as exc:
    print(f'[event-episode] wandb.log failed: {exc}')


def _softmax(x, axis=-1):
  x = x - x.max(axis=axis, keepdims=True)
  ex = np.exp(x)
  return ex / np.maximum(ex.sum(axis=axis, keepdims=True), 1e-8)
