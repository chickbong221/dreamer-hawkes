import collections
from functools import partial as bind
from time import time

import elements
import embodied
import numpy as np


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)

  start_time = time()
  train_vector_episode_done = [False]
  train_episode_idx = [0]
  eval_stat_next = [True]
  eval_video_next = [False]

  def _arg(name, default):
    if hasattr(args, 'get'):
      return args.get(name, default)
    return getattr(args, name, default)

  # TD-MPC2 defaults: num_eval_envs=4, eval_episodes_per_env=2.
  # Total eval episodes per call = eval_envs * eval_eps; per-episode metrics
  # are averaged in `all_metrics` below, so any positive eval_eps is supported.
  eval_freq = int(float(_arg('eval_freq', 50000)))
  eval_video_freq = int(float(_arg('eval_video_freq', 0)))
  eval_num_envs = int(_arg('eval_envs', 4))
  eval_episodes_per_env = max(int(_arg('eval_eps', 1)), 1)
  dyn_config = getattr(getattr(agent, 'config', None), 'dyn', None)
  dyn_type = getattr(dyn_config, 'typ', None)
  eval_event_log = bool(_arg('eval_event_log', False)) and dyn_type == 'hawkes'

  # ManiSkill episode-level metrics that TD-MPC2 logs once per completed
  # training vector episode batch as mean over cfg.num_envs.
  maniskill_metric_keys = (
      'log/success_once',
      'log/success_at_end',
      'log/fail_once',
  )
  maniskill_batch = {
      'count': 0,
      'values': collections.defaultdict(list),
  }

  def _to_numpy(value):
    try:
      import torch
      if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    except Exception:
      pass
    return np.asarray(value)

  def _flush_logger_before_direct_wandb():
    # Direct wandb.log(step=...) must not overtake queued Dreamer summaries.
    # Flush first so W&B sees monotonically increasing steps.
    try:
      logger.write()
    except Exception as exc:
      print(f'Could not flush Dreamer logger before direct W&B log: {exc}')

  def _wandb_log_direct(payload, step_value):
    if not payload:
      return
    try:
      import wandb
      if wandb.run is not None:
        wandb.log(payload, step=int(step_value))
    except Exception as exc:
      print(f'Could not log directly to W&B: {exc}')

  def log_maniskill_vector_batch():
    values = maniskill_batch['values']

    payload = {}
    for key, vals in values.items():
      if vals:
        # log/success_once -> train/success_once
        payload[f'train/{key[len("log/"):]}'] = float(np.mean(vals))

    if payload:
      # Do not call wandb.log() directly for train metrics. Dreamer's logger may
      # still hold queued summaries for earlier callback steps. Logging through
      # the same logger stream prevents W&B step-order warnings.
      logger.add(payload)
      logger.write()

    values.clear()
    maniskill_batch['count'] = 0
    train_vector_episode_done[0] = True

  def _tile_images(images, nrows):
    try:
      from mani_skill.utils.visualization.misc import tile_images
      return tile_images(images, nrows=nrows)
    except Exception:
      images = np.asarray(images)
      n = len(images)
      nrows = max(1, int(nrows))
      ncols = int(np.ceil(n / nrows))
      h, w, c = images.shape[1:]
      canvas = np.zeros((nrows * h, ncols * w, c), dtype=images.dtype)
      for idx, image in enumerate(images):
        r, col = divmod(idx, ncols)
        canvas[r * h:(r + 1) * h, col * w:(col + 1) * w] = image
      return canvas

  def _log_wandb_eval_video(video_frames, step_value, fps=15,
                            key='videos/eval_video'):
    """TD-MPC2-style W&B video logging under videos/eval_video.

    video_frames is a list of arrays [num_eval_envs, height, width, 3], one per
    eval timestep. We tile vector-env frames per timestep, convert to W&B's
    [T, C, H, W] layout, and log at the current training step.
    """
    if not video_frames:
      return
    try:
      import wandb
      if wandb.run is None:
        return
      frames = [_to_numpy(frame) for frame in video_frames]
      frames = [frame[None] if frame.ndim == 3 else frame for frame in frames]
      frames = np.stack(frames, axis=0)  # [T, N, H, W, 3]
      nrows = max(1, int(np.sqrt(frames.shape[1])))
      tiled = [_tile_images(rgbs, nrows=nrows) for rgbs in frames]
      tiled = np.stack(tiled, axis=0)  # [T, H, W, 3]
      _wandb_log_direct({
          key: wandb.Video(
              tiled.transpose(0, 3, 1, 2), fps=fps, format='mp4')},
          step_value)
    except Exception as exc:
      print(f'Could not log eval video to W&B: {exc}')

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')

      elif key.startswith('log/') and tran['is_last']:
        # TD-MPC2 logs ManiSkill success metrics immediately once per completed
        # vector episode batch. Do not also average them through Dreamer's
        # epstats path.
        if getattr(driver, 'batched', False) and key in maniskill_metric_keys:
          continue

        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key, value, agg='avg')

    vector_batch_complete = False
    if getattr(driver, 'batched', False) and tran['is_last']:
      for key in maniskill_metric_keys:
        if key not in tran:
          continue
        value = np.asarray(tran[key])
        assert value.ndim == 0, (key, value.shape, value.dtype)
        maniskill_batch['values'][key].append(float(value))

      maniskill_batch['count'] += 1
      vector_batch_complete = maniskill_batch['count'] >= driver.length

    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

    # Log after episode summaries for the final worker have been queued, so
    # logger.write() flushes the whole completed vector batch in order.
    if vector_batch_complete:
      log_maniskill_vector_batch()

  # Keep the original Driver construction. In your ManiSkill main.py,
  # embodied.Driver is monkey-patched so this creates one Dreamer env object
  # containing env.maniskill.num_envs GPU sub-envs for training.
  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  # TD-MPC2 creates a separate eval env once at startup and reuses it. It uses
  # cfg.num_eval_envs, not cfg.num_envs. Here eval_num_envs comes from
  # run.eval_envs, which should be 4 for TD-MPC2 default comparability.
  eval_env = None
  if getattr(driver, 'batched', False):
    eval_env = make_env(
        0,
        num_envs=eval_num_envs,
        is_eval=True,
        eval_reconfiguration_frequency=int(_arg(
            'eval_reconfiguration_frequency', 1)),
    )

  def _zero_actions(env, num_envs):
    acts = {
        k: np.zeros((num_envs,) + v.shape, v.dtype)
        for k, v in env.act_space.items()}
    acts['reset'] = np.ones(num_envs, bool)
    return acts

  def _mask(value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _mean_episode_metrics(metrics):
    out = {}
    for key, value in metrics.items():
      arr = _to_numpy(value).astype(np.float32)
      if arr.size:
        out[key] = float(np.mean(arr))
    return out

  @elements.timer.section('evalfn')
  def evalfn(do_video=False):
    if eval_env is None:
      return

    # Ensure direct eval/* and videos/eval_video logs do not advance W&B past
    # queued Dreamer summaries from earlier steps.
    _flush_logger_before_direct_wandb()

    eval_policy = lambda *xs: agent.policy(*xs, mode='eval')
    horizon = int(getattr(eval_env, '_max_episode_steps', 10000))
    all_metrics = collections.defaultdict(list)
    video_frames = []
    # Filled only from episode 0, environment index 0. This reuses the
    # existing evaluation rollout and never creates an event-specific env.
    event_data = collections.defaultdict(list) if eval_event_log else None

    for eval_episode_idx in range(eval_episodes_per_env):
      carry = agent.init_policy(eval_num_envs)
      acts = _zero_actions(eval_env, eval_num_envs)
      fallback_return = np.zeros(eval_num_envs, np.float32)
      fallback_len = np.zeros(eval_num_envs, np.float32)
      final_metrics = None

      for t in range(horizon + 2):
        obs = eval_env.step(acts)
        obs = {k: np.asarray(v) for k, v in obs.items()}

        if do_video and hasattr(eval_env, 'render'):
          try:
            frame = np.asarray(eval_env.render())
            video_frames.append(frame[:4].copy())
          except Exception as exc:
            if t == 0:
              print(f'Could not render eval video frames: {exc}')

        not_first = ~np.asarray(obs['is_first'], dtype=bool)
        fallback_return += obs['reward'].astype(np.float32) * not_first
        fallback_len += not_first.astype(np.float32)

        done = np.asarray(obs['is_last'], dtype=bool)
        if bool(done[0]):
          # Prefer full ManiSkill final_info['episode'], matching TD-MPC2's
          # final_info_metrics(). Fall back to values available in obs.
          if hasattr(eval_env, 'last_episode_metrics'):
            final_metrics = eval_env.last_episode_metrics()
          if not final_metrics:
            final_metrics = {
                'return': fallback_return,
                'episode_len': fallback_len,
                'reward': fallback_return / np.maximum(fallback_len, 1.0),
            }
            for name in ('success_once', 'success_at_end',
                         'fail_once', 'fail_at_end'):
              key = f'log/{name}'
              if key in obs:
                final_metrics[name] = obs[key]
          break

        # log/ keys are intentionally excluded from policy input.
        obs_for_policy = {k: v for k, v in obs.items()
                          if not k.startswith('log/')}
        carry, acts, outs = eval_policy(carry, obs_for_policy)
        assert all(k not in acts for k in outs), (
            list(outs.keys()), list(acts.keys()))

        if event_data is not None and eval_episode_idx == 0 and \
           'haw_prob' in outs:
          for key in ('haw_prob', 'haw_event', 'haw_prior_prob',
                      'haw_lam'):
            event_data[key].append(float(_to_numpy(outs[key])[0]))
          event_data['reward'].append(float(obs['reward'][0]))
          if 'action' in acts:
            event_data['action'].append(_to_numpy(acts['action'])[0].copy())
          for key in ('depth_head', 'depth_hand'):
            if key in obs:
              event_data[key].append(obs[key][0].copy())

        if done.any():
          mask = ~done
          acts = {k: _mask(v, mask) for k, v in acts.items()}
        acts = {**acts, 'reset': done.copy()}
      else:
        raise RuntimeError(
            f'Eval episode did not finish within {horizon + 2} steps.')

      for key, value in _mean_episode_metrics(final_metrics).items():
        all_metrics[key].append(value)

    eval_metrics = {
        key: float(np.mean(values))
        for key, values in all_metrics.items()}
    eval_metrics.update({
        'step': int(step),
        'episode': int(train_episode_idx[0]),
        'total_time': time() - start_time,
    })

    # TD-MPC2 Logger.log(eval_metrics, 'eval') writes eval/<key> and uses
    # W&B step=d['step']. Merge event panels into this same call so all media
    # from an evaluation phase shares one W&B step.
    eval_payload = {f'eval/{k}': v for k, v in eval_metrics.items()}
    if event_data:
      try:
        from dreamerv3.log_event_episode import build_event_episode_payload
        captured = {key: np.asarray(values)
                    for key, values in event_data.items()}
        max_depth = float(getattr(eval_env, '_max_depth', 20000.0))
        eval_payload.update(build_event_episode_payload(
            agent, captured, max_depth=max_depth))
      except Exception as exc:
        import traceback
        print(f'[event-episode] eval logging failed: {exc!r}')
        traceback.print_exc()
    _wandb_log_direct(eval_payload, step)

    if do_video:
      _log_wandb_eval_video(video_frames, step, fps=15,
                            key='videos/eval_video')

    shown = ', '.join(
        f'{k}: {v:.4g}' for k, v in eval_metrics.items()
        if isinstance(v, (int, float, np.integer, np.floating)))
    suffix = ' [+video]' if do_video else ''
    print(f'Eval metrics{suffix} | {shown}')

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  # cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  # Only resume from the run's own ckpt dir if it already exists; do not
  # auto-save a fresh checkpoint at start. End-of-training cp.save() still runs.
  if (logdir / 'ckpt').exists():
    cp.load()

  print('Start training loop')
  policy = lambda *xs: agent.policy(*xs, mode='train')
  driver.reset(agent.init_policy)

  # evaluate once at step 0 before the first train rollout
  if eval_env is not None and eval_stat_next[0]:
    do_video = eval_video_freq > 0
    eval_video_next[0] = False
    evalfn(do_video=do_video)
    eval_stat_next[0] = False

  while step < args.steps:

    train_vector_episode_done[0] = False
    driver(policy, steps=10)

    if eval_env is not None and eval_freq > 0:
      if int(step) > 0 and int(step) % eval_freq < driver.length:
        eval_stat_next[0] = True
      if eval_video_freq > 0 and int(step) > 0 and int(step) % eval_video_freq < driver.length:
        eval_video_next[0] = True
      if train_vector_episode_done[0] and eval_stat_next[0]:
        evalfn(do_video=eval_video_next[0])
        eval_stat_next[0] = False
        eval_video_next[0] = False

    if train_vector_episode_done[0]:
      train_episode_idx[0] += 1

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      report_mets = agg.result()
      video_mets = {
          k: v for k, v in report_mets.items()
          if isinstance(v, np.ndarray) and v.ndim == 4 and v.dtype == np.uint8}
      scalar_mets = {k: v for k, v in report_mets.items() if k not in video_mets}
      logger.add(scalar_mets, prefix='report')
      if video_mets:
        try:
          import wandb
          if wandb.run is not None:
            _flush_logger_before_direct_wandb()
            payload = {}
            for k, v in video_mets.items():
              # v shape: [T, H, W, C] uint8 — wandb.Video expects [T, C, H, W]
              payload[f'report/{k}'] = wandb.Video(
                  v.transpose(0, 3, 1, 2),
                  fps=int(_arg('report_fps', 4)),
                  format='mp4')
            _wandb_log_direct(payload, step)
        except Exception as exc:
          print(f'Could not log report videos to W&B as mp4: {exc}')

    if should_log(step):
      logger.add(train_agg.result())
      epstats.result()
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

  # save_every < 0 disables checkpointing entirely. There is no periodic
  # save in this loop; this is the only cp.save() call.
  if args.save_every >= 0:
    cp.save()

  if eval_env is not None:
    eval_env.close()
  driver.close()
  logger.close()
