"""
OmniGibson (BEHAVIOR-1K) environment wrapper for DreamerV3.

Wraps a single OmniGibson Environment instance and converts its
Gymnasium-style API into the embodied.Env interface expected by DreamerV3.

Observation layout
------------------
  image : uint8 [H, W, 3]   — first RGB camera found in flattened obs
  state : float32 [D]        — all non-image obs concatenated
  reward, is_first, is_last, is_terminal : scalars

Action layout
-------------
  action : float32 [A]  — flattened, normalized to [-1, 1] by OmniGibson
  reset  : bool

Usage
-----
  env = OmniGibson(
      activity_name='picking_up_trash',
      config_path='embodied/envs/behavior1k_cfg/default.yaml',
      image_size=64,
  )
"""

import functools
import pathlib

import elements
import embodied
import numpy as np


_HERE = pathlib.Path(__file__).parent
_DEFAULT_CFG = _HERE / 'behavior1k_cfg' / 'default.yaml'


class OmniGibson(embodied.Env):

    def __init__(
        self,
        activity_name,
        config_path=None,
        activity_definition_id=0,
        activity_instance_id=0,
        image_size=64,
        seed=0,
        **kwargs,
    ):
        import omnigibson as og
        from omnigibson.utils.config_utils import parse_config

        cfg_path = str(config_path or _DEFAULT_CFG)
        cfg = parse_config(cfg_path)

        # Override task params
        cfg['task']['activity_name'] = activity_name
        cfg['task']['activity_definition_id'] = int(activity_definition_id)
        cfg['task']['activity_instance_id'] = int(activity_instance_id)

        # Override image resolution on all robot vision sensors
        _set_nested(cfg, ['robots'], self._patch_image_size, image_size)

        # Always flatten so we get a simple dict + 1D action
        cfg.setdefault('env', {})
        cfg['env']['flatten_action_space'] = True
        cfg['env']['flatten_obs_space'] = True
        cfg['env']['automatic_reset'] = False

        # Apply any extra overrides passed as kwargs
        for key, value in kwargs.items():
            cfg[key] = value

        self._og_env = og.Environment(configs=cfg)
        self._image_size = image_size
        self._done = True

        # Discover obs/act structure from a warm reset
        raw_obs, _ = self._og_env.reset()
        self._img_key, self._state_keys = self._discover_keys(raw_obs)
        self._act_dim = int(self._og_env.action_space.shape[0])
        self._state_dim = self._compute_state_dim(raw_obs)

    # ------------------------------------------------------------------
    # embodied.Env interface
    # ------------------------------------------------------------------

    @functools.cached_property
    def obs_space(self):
        spaces = {
            'image': elements.Space(np.uint8, (self._image_size, self._image_size, 3)),
            'state': elements.Space(np.float32, (self._state_dim,)),
            'reward': elements.Space(np.float32, ()),
            'is_first': elements.Space(bool, ()),
            'is_last': elements.Space(bool, ()),
            'is_terminal': elements.Space(bool, ()),
        }
        return spaces

    @functools.cached_property
    def act_space(self):
        return {
            'action': elements.Space(np.float32, (self._act_dim,), low=-1.0, high=1.0),
            'reset': elements.Space(bool, ()),
        }

    def step(self, action):
        import torch as th
        if action['reset'] or self._done:
            self._done = False
            raw_obs, _ = self._og_env.reset()
            return self._make_obs(raw_obs, reward=0.0, terminated=False,
                                  truncated=False, is_first=True)

        act = np.asarray(action['action'], dtype=np.float32)
        act_tensor = th.tensor(act)
        raw_obs, reward, terminated, truncated, _info = self._og_env.step(act_tensor)
        self._done = bool(terminated or truncated)
        return self._make_obs(raw_obs, float(reward), terminated, truncated,
                              is_first=False)

    def close(self):
        try:
            self._og_env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_keys(self, raw_obs):
        """Split flat obs keys into one image key and remaining state keys."""
        img_key = None
        state_keys = []
        for k, v in raw_obs.items():
            v = _to_numpy(v)
            if v.ndim == 3 and v.shape[-1] in (3, 4):
                if img_key is None:
                    img_key = k
            elif np.issubdtype(v.dtype, np.floating) or np.issubdtype(v.dtype, np.integer):
                state_keys.append(k)
        if img_key is None:
            raise RuntimeError(
                f'No RGB observation found in OmniGibson obs keys: {list(raw_obs.keys())}. '
                'Make sure the robot config includes obs_modalities: [rgb].')
        return img_key, state_keys

    def _compute_state_dim(self, raw_obs):
        total = 0
        for k in self._state_keys:
            v = _to_numpy(raw_obs[k])
            total += int(np.prod(v.shape))
        return max(total, 1)

    def _extract_image(self, raw_obs):
        import torch as th
        img = raw_obs[self._img_key]
        if isinstance(img, th.Tensor):
            img = img.cpu().numpy()
        img = np.asarray(img)
        # Take only RGB, drop alpha if present
        img = img[..., :3]
        # Resize to target resolution if needed
        h, w = img.shape[:2]
        if (h, w) != (self._image_size, self._image_size):
            img = _resize_image(img, self._image_size)
        return img.astype(np.uint8)

    def _extract_state(self, raw_obs):
        if not self._state_keys:
            return np.zeros(1, dtype=np.float32)
        parts = []
        for k in self._state_keys:
            v = _to_numpy(raw_obs[k]).astype(np.float32).ravel()
            parts.append(v)
        return np.concatenate(parts, axis=0)

    def _make_obs(self, raw_obs, reward, terminated, truncated, is_first):
        return {
            'image': self._extract_image(raw_obs),
            'state': self._extract_state(raw_obs),
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': bool(terminated or truncated),
            'is_terminal': bool(terminated),
        }

    @staticmethod
    def _patch_image_size(robots_cfg, image_size):
        for robot in robots_cfg:
            robot.setdefault('sensor_config', {})
            robot['sensor_config'].setdefault('VisionSensor', {})
            robot['sensor_config']['VisionSensor'].setdefault('sensor_kwargs', {})
            robot['sensor_config']['VisionSensor']['sensor_kwargs']['image_height'] = image_size
            robot['sensor_config']['VisionSensor']['sensor_kwargs']['image_width'] = image_size


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------

def _set_nested(cfg, keys, fn, *args):
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    fn(node[keys[-1]], *args)


def _to_numpy(v):
    import torch as th
    if isinstance(v, th.Tensor):
        return v.cpu().numpy()
    return np.asarray(v)


def _resize_image(img, size):
    try:
        from PIL import Image
        return np.asarray(
            Image.fromarray(img).resize((size, size), Image.BILINEAR))
    except ImportError:
        pass
    # Fallback: nearest-neighbour without PIL
    h, w = img.shape[:2]
    row = (np.arange(size) * h // size).astype(int)
    col = (np.arange(size) * w // size).astype(int)
    return img[np.ix_(row, col)]
