"""Evaluation event-episode visualizer for the binary-event Hawkes RSSM.

Builds W&B panels from the first episode of evaluation environment index 0.
The evaluation loop supplies already-collected Hawkes features and observation
frames, so this module never creates an environment or runs an extra rollout.

Scope is deliberately narrow. Aggregate Hawkes statistics (rate, prior gap,
intensity, learned b/alpha/beta) are already logged as training metrics by
`HawkesRSSM.loss()` every `log_every` steps, where they trend properly. What
those cannot show is *where inside one episode* events fire, so that is all
this module produces:

  event/probs           pi_t and p_haw_t over the episode
  event/spike_trace     binary strip of hard events
  event/episode_video   frames with a red border on spike steps
  event/hard_count      events in the episode
  event/expected_count  sum of pi_t (still meaningful when the eval
                        threshold suppresses every hard event)
"""

import numpy as np


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


def _rgb_to_frames(images):
  """uint8 [T, H, W, C] -> [T, H, W*views, 3], tiling cameras horizontally."""
  x = np.asarray(images, np.uint8)
  c = x.shape[-1]
  if c == 1:
    return np.repeat(x, 3, axis=-1)
  if c == 3:
    return x
  if c % 3 == 0:  # several cameras flattened into channels
    t, h, w, _ = x.shape
    x = x.reshape(t, h, w, c // 3, 3).transpose(0, 1, 3, 2, 4)
    return x.reshape(t, h, (c // 3) * w, 3)
  return x[..., :3]


def _episode_frames(data, T, max_depth):
  """Best available RGB frames for the episode, or None."""
  if 'image' in data:
    return _rgb_to_frames(np.asarray(data['image'])[:T])
  if 'depth_head' in data:
    head = np.stack(
        [_depth_to_rgb(f, max_depth) for f in data['depth_head'][:T]], 0)
    if 'depth_hand' in data:
      hand = np.stack(
          [_depth_to_rgb(f, max_depth) for f in data['depth_hand'][:T]], 0)
      return np.concatenate([head, hand], axis=2)
    return head
  return None


def build_event_episode_payload(agent, data, max_depth=None):
  """Build binary-event panels from an episode captured by the eval loop.

  Args:
    agent: Dreamer agent with Hawkes dynamics.
    data: arrays captured from evaluation environment index 0. Required keys
      are haw_prob, haw_event and haw_prior_prob. Depth frames are optional.
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

  max_depth = float(max_depth or 20000.0)
  prob = np.asarray(data['haw_prob'], np.float32).reshape(-1)
  T = len(prob)
  event = np.asarray(data['haw_event'], np.float32).reshape(-1)[:T]
  prior = np.asarray(data['haw_prior_prob'], np.float32).reshape(-1)[:T]
  spikes = event > 0.5

  payload = {
      'event/hard_count': int(spikes.sum()),
      'event/expected_count': float(prob.sum()),
  }

  # Posterior against causal prior, one chart. lambda_t is omitted on purpose:
  # p_haw = 1 - exp(-lambda) is a monotone transform of it, so plotting both
  # shows the same curve twice.
  payload['event/probs'] = wandb.plot.line_series(
      xs=list(range(T)),
      ys=[prob.tolist(), prior.tolist()],
      keys=['post_prob', 'prior_prob'],
      title='Event probability over the episode', xname='t')

  # Binary strip: red where an event fired, dark otherwise.
  strip = np.zeros((1, T, 3), np.uint8)
  strip[0, spikes] = np.array([255, 60, 60], np.uint8)
  strip[0, ~spikes] = np.array([30, 30, 40], np.uint8)
  payload['event/spike_trace'] = wandb.Image(
      _upscale(strip, 24, T * 6),
      caption=f'{int(spikes.sum())} events in {T} steps')

  # Episode video with an event bar above the frame: dark grey normally, red
  # on the steps where an event fired.
  frames = _episode_frames(data, T, max_depth)
  if frames is not None:
    bar = 8
    frames = np.concatenate(
        [np.zeros((T, bar, frames.shape[2], 3), np.uint8), frames], axis=1)
    frames[:, :bar] = np.array([40, 40, 48], np.uint8)
    frames[spikes, :bar] = np.array([255, 40, 40], np.uint8)
    payload['event/episode_video'] = wandb.Video(
        frames.transpose(0, 3, 1, 2), fps=10, format='mp4',
        caption=f'red bar = event ({int(spikes.sum())} of {T} steps)')

  return payload
