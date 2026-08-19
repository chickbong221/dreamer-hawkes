"""Evaluation event-episode visualizer for the binary-event Hawkes RSSM.

Builds W&B panels from the first episode of evaluation environment index 0.
The evaluation loop supplies already-collected Hawkes features and observation
frames, so this module never creates an environment or runs an extra rollout.

Outputs (all under `event/`):
  post_prob        pi_t over time
  prior_prob       p_haw_t over time
  spike_trace      hard events as a binary strip
  lambda           scalar intensity over time
  post_prior_gap   |pi_t - p_haw_t| over time
  reward_trace     reward over time
  expected_rate    mean pi_t
  hard_rate        mean hard event
  rate_error       hard_rate - expected_rate
  expected_count   sum pi_t
  hard_count       number of spikes
  events           one table row per spike
  episode_video    depth_head | depth_hand, red border on spike frames
  base/alpha/beta  learned scalar Hawkes parameters
"""

import numpy as np


# ---------------------------------------------------------------------------
# Colormap and image helpers
# ---------------------------------------------------------------------------

def _viridis(values):
  """Apply a viridis-like colormap. Input [..] float in [0,1] -> [..,3] u8."""
  values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
  try:
    import matplotlib.cm as cm
    rgba = cm.viridis(values)
    return (rgba[..., :3] * 255).astype(np.uint8)
  except Exception:
    r = np.clip(2 * values - 0.5, 0, 1)
    g = np.clip(2 * values * (1 - values) * 4, 0, 1)
    b = np.clip(1 - 1.5 * values, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _upscale(image, target_h, target_w):
  """Nearest-neighbour upscale a 2D or 3D image to (target_h, target_w[, C])."""
  h, w = image.shape[:2]
  rh = max(target_h // max(h, 1), 1)
  rw = max(target_w // max(w, 1), 1)
  return np.repeat(np.repeat(image, rh, axis=0), rw, axis=1)


def _depth_to_rgb(depth_u16, max_depth):
  """Convert depth uint16 mm [H, W, 1] to RGB uint8 [H, W, 3]."""
  depth = np.asarray(depth_u16, dtype=np.float32).squeeze(-1)
  depth = np.clip(depth, 0, max_depth) / max(max_depth, 1.0)
  gray = (depth * 255).astype(np.uint8)
  return np.stack([gray, gray, gray], axis=-1)


def _softplus(x):
  return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _host(value):
  """Device -> host. jax_transfer_guard is 'disallow', which blocks implicit
  transfers such as np.asarray on a device array; device_get is explicit and
  stays allowed."""
  try:
    import jax
    return np.asarray(jax.device_get(value))
  except ImportError:
    return np.asarray(value)


def _fetch_hawkes_params(agent):
  """Pull scalar (b, alpha, beta) from agent.params. Mirrors _haw_params."""
  try:
    params = agent.params

    def find(name):
      for k, v in params.items():
        if k.endswith(name):
          return float(_host(v).reshape(()))
      return None

    base = find('haw_base')
    alpha_raw = find('haw_alpha_raw')
    beta_raw = find('haw_beta_raw')
  except Exception as exc:
    print(f'[event-episode] Hawkes params unavailable: {exc!r}')
    return None
  if base is None or alpha_raw is None or beta_raw is None:
    return None
  return base, float(_softplus(alpha_raw)), float(_softplus(beta_raw))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_event_episode_payload(agent, data, max_depth=None):
  """Build binary-event panels from an episode captured by the eval loop.

  Args:
    agent: Dreamer agent with Hawkes dynamics.
    data: arrays captured from evaluation environment index 0. Required keys
      are haw_prob, haw_event, haw_prior_prob, haw_lam and reward. Action and
      depth frames are optional.
    max_depth: depth maximum in millimeters for uint16 visualization.

  Returns:
    A dict of W&B media objects ready to merge into the eval-step payload.
  """
  if getattr(agent, 'config', None) is None or \
     agent.config.dyn.typ != 'hawkes':
    print('[event-episode] dyn.typ != hawkes — skipping.')
    return {}

  try:
    import wandb
  except ImportError:
    print('[event-episode] wandb not installed — skipping.')
    return {}
  if wandb.run is None:
    print('[event-episode] no active wandb run — skipping.')
    return {}

  if 'haw_prob' not in data or len(data['haw_prob']) == 0:
    print('[event-episode] No Hawkes features captured — skipping.')
    return {}

  data = {key: np.asarray(value) for key, value in data.items()}
  max_depth = float(max_depth or 20000.0)

  prob = np.asarray(data['haw_prob'], np.float32).reshape(-1)
  T = len(prob)
  event = np.asarray(data['haw_event'], np.float32).reshape(-1)[:T]
  prior = np.asarray(data['haw_prior_prob'], np.float32).reshape(-1)[:T]
  lam = np.asarray(data['haw_lam'], np.float32).reshape(-1)[:T]
  reward = np.asarray(data['reward'], np.float32).reshape(-1)[:T]
  gap = np.abs(prob - prior)
  spikes = np.nonzero(event > 0.5)[0]

  payload = {}

  # ---- Per-step traces ------------------------------------------------------
  trace = wandb.Table(
      data=[[t, float(prob[t]), float(prior[t]), float(lam[t]),
             float(event[t]), float(gap[t]), float(reward[t])]
            for t in range(T)],
      columns=['t', 'post_prob', 'prior_prob', 'lambda', 'hard_event',
               'post_prior_gap', 'reward'])
  for key, col, title in (
      ('post_prob', 'post_prob', 'Posterior event probability pi_t'),
      ('prior_prob', 'prior_prob', 'Causal Hawkes probability p_haw_t'),
      ('lambda', 'lambda', 'Hawkes intensity lambda_t'),
      ('post_prior_gap', 'post_prior_gap', '|pi_t - p_haw_t|'),
      ('reward_trace', 'reward', 'Reward per timestep')):
    payload[f'event/{key}'] = wandb.plot.line(trace, 't', col, title=title)

  # ---- Spike strip: white where an event fired ------------------------------
  strip = _upscale((event > 0.5).astype(np.float32)[None, :], 24, T * 6)
  payload['event/spike_trace'] = wandb.Image(
      _viridis(strip), caption=f'hard events ({len(spikes)} of {T} steps)')

  # ---- Scalar summaries -----------------------------------------------------
  hard_rate = float(event.mean()) if T else 0.0
  expected_rate = float(prob.mean()) if T else 0.0
  payload['event/expected_rate'] = expected_rate
  payload['event/hard_rate'] = hard_rate
  payload['event/rate_error'] = hard_rate - expected_rate
  payload['event/expected_count'] = float(prob.sum())
  payload['event/hard_count'] = int(len(spikes))
  payload['event/post_prior_gap_mean'] = float(gap.mean()) if T else 0.0

  # ---- One table row per spike ---------------------------------------------
  has_head = 'depth_head' in data
  has_hand = 'depth_hand' in data
  has_act = 'action' in data
  cols = ['t', 'post_prob', 'prior_prob', 'lambda', 'reward']
  if has_act:
    cols.append('action')
  if has_head:
    cols.append('head')
  if has_hand:
    cols.append('hand')
  rows = []
  for t in spikes:
    t = int(t)
    row = [t, round(float(prob[t]), 4), round(float(prior[t]), 4),
           round(float(lam[t]), 4), round(float(reward[t]), 4)]
    if has_act:
      row.append(np.array2string(
          np.asarray(data['action'][t]).reshape(-1), precision=3))
    if has_head:
      row.append(wandb.Image(
          _depth_to_rgb(data['depth_head'][t], max_depth),
          caption=f'spike t={t}'))
    if has_hand:
      row.append(wandb.Image(
          _depth_to_rgb(data['depth_hand'][t], max_depth),
          caption=f'spike t={t}'))
    rows.append(row)
  payload['event/events'] = wandb.Table(data=rows, columns=cols)

  # ---- Episode video, red border on spike frames ----------------------------
  if has_head:
    head = np.stack(
        [_depth_to_rgb(f, max_depth) for f in data['depth_head'][:T]], 0)
    if has_hand:
      hand = np.stack(
          [_depth_to_rgb(f, max_depth) for f in data['depth_hand'][:T]], 0)
      frames = np.concatenate([head, hand], axis=2)
    else:
      frames = head
    pad = 6
    frames = np.concatenate(
        [np.zeros((T, pad, frames.shape[2], 3), np.uint8), frames], axis=1)
    frames[event > 0.5, :pad] = np.array([255, 0, 0], np.uint8)
    payload['event/episode_video'] = wandb.Video(
        frames.transpose(0, 3, 1, 2), fps=10, format='mp4',
        caption='depth_head | depth_hand; red top border = event')

  # ---- Learned scalar Hawkes parameters ------------------------------------
  params = _fetch_hawkes_params(agent)
  if params is not None:
    base, alpha, beta = params
    payload['event/base'] = base
    payload['event/alpha'] = alpha
    payload['event/beta'] = beta
    payload['event/base_prob'] = float(-np.expm1(-_softplus(base)))

  return payload
