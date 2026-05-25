#!/usr/bin/env python3
"""
Multi-task evaluation of a DreamerV3 checkpoint.

Loads a checkpoint, runs eval episodes on each specified task, and prints a
per-task metrics table. Optionally logs to Weights & Biases and saves a
results.json to --logdir.

Constraint: all eval tasks must share the same obs/act space as --train_task
(same robot, same control_mode, same obs_mode). Mismatched tasks are skipped
with a warning rather than crashing.

Usage (from repo root):
  python sequential/eval_multitask.py \\
    --checkpoint ~/logdir/sequential/run1/phase2/pretrained/ckpt \\
    --train_task PlugCharger-v1 \\
    --tasks PickCube-v1 PushCube-v1 StackCube-v1 PlugCharger-v1 \\
    --configs maniskill_rgb \\
    --episodes 5 \\
    --eval_envs 4 \\
    --logdir ~/logdir/sequential/run1/eval/pretrained
"""

import argparse
import collections
import json
import os
import pathlib
import sys
from functools import partial as bind

# Must be set before JAX initialises — same as dreamerv3/main.py.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# sequential/ lives one level below the repo root.
REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import elements
import numpy as np
import ruamel.yaml as yaml


# ── Config helpers ────────────────────────────────────────────────────────────

def _build_config(task_name, config_names):
    """Construct a DreamerV3 Config for a given task, same as main.py does."""
    raw = elements.Path(REPO / 'dreamerv3' / 'configs.yaml').read()
    all_cfg = yaml.YAML(typ='safe').load(raw)
    config = elements.Config(all_cfg['defaults'])
    for name in config_names:
        config = config.update(all_cfg[name])
    config = config.update(task=f'maniskill_{task_name}')
    # Throwaway logdir — nothing is written to disk during eval.
    config = config.update(logdir='/tmp/dreamer_eval_tmp')
    return config


# ── Eval loop (copied from embodied/run/train.py evalfn) ─────────────────────

def _run_eval_episodes(agent, eval_env, num_envs, episodes_per_env):
    """
    Run eval episodes and return mean metrics across all episodes.

    Logic is copied directly from evalfn() in embodied/run/train.py so that
    metric calculation stays consistent with training-time eval. Differences:
      - No W&B video logging
      - No logger.write() / step counter — metrics returned as a plain dict
      - episodes_per_env episodes are run in sequence
    """
    try:
        import torch
        def _to_numpy(v):
            return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    except ImportError:
        def _to_numpy(v):
            return np.asarray(v)

    def _zero_actions(env, n):
        acts = {k: np.zeros((n,) + sp.shape, sp.dtype)
                for k, sp in env.act_space.items()}
        acts['reset'] = np.ones(n, bool)
        return acts

    def _mask(value, mask):
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        return value * mask.astype(value.dtype)

    def _mean_metrics(metrics):
        out = {}
        for k, v in metrics.items():
            arr = _to_numpy(v).astype(np.float32)
            if arr.size:
                out[k] = float(np.mean(arr))
        return out

    eval_policy = lambda *xs: agent.policy(*xs, mode='eval')
    horizon = int(getattr(eval_env, '_max_episode_steps', 10000))
    all_metrics = collections.defaultdict(list)

    for ep_idx in range(episodes_per_env):
        carry = agent.init_policy(num_envs)
        acts = _zero_actions(eval_env, num_envs)
        fallback_return = np.zeros(num_envs, np.float32)
        fallback_len   = np.zeros(num_envs, np.float32)
        final_metrics  = None

        for t in range(horizon + 2):
            obs = eval_env.step(acts)
            obs = {k: np.asarray(v) for k, v in obs.items()}

            not_first = ~np.asarray(obs['is_first'], dtype=bool)
            fallback_return += obs['reward'].astype(np.float32) * not_first
            fallback_len   += not_first.astype(np.float32)

            done = np.asarray(obs['is_last'], dtype=bool)
            if bool(done[0]):
                # Prefer full ManiSkill final_info['episode'] metrics, matching
                # the evalfn() path in train.py. Fall back to obs-level values.
                if hasattr(eval_env, 'last_episode_metrics'):
                    final_metrics = eval_env.last_episode_metrics()
                if not final_metrics:
                    final_metrics = {
                        'return':      fallback_return,
                        'episode_len': fallback_len,
                        'reward':      fallback_return / np.maximum(fallback_len, 1.0),
                    }
                    for name in ('success_once', 'success_at_end',
                                 'fail_once', 'fail_at_end'):
                        key = f'log/{name}'
                        if key in obs:
                            final_metrics[name] = obs[key]
                break

            obs_for_policy = {k: v for k, v in obs.items()
                              if not k.startswith('log/')}
            carry, acts, outs = eval_policy(carry, obs_for_policy)
            assert all(k not in acts for k in outs), (list(outs), list(acts))

            if done.any():
                acts = {k: _mask(v, ~done) for k, v in acts.items()}
            acts = {**acts, 'reset': done.copy()}

        else:
            raise RuntimeError(
                f'Episode {ep_idx} did not finish within {horizon + 2} steps.')

        for k, v in _mean_metrics(final_metrics).items():
            all_metrics[k].append(v)

    return {k: float(np.mean(vs)) for k, vs in all_metrics.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a DreamerV3 checkpoint across multiple ManiSkill tasks.')
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint directory (e.g. .../phase2/pretrained/ckpt)')
    parser.add_argument('--train_task', required=True,
                        help='Task the checkpoint was last trained on. '
                             'Sets the obs/act space the agent expects.')
    parser.add_argument('--tasks', nargs='+', required=True,
                        help='Task IDs to evaluate. Must share obs/act space with --train_task.')
    parser.add_argument('--configs', nargs='+', default=['maniskill_rgb'],
                        help='DreamerV3 config block names, same as used during training.')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Eval episodes per task (default: 5).')
    parser.add_argument('--eval_envs', type=int, default=4,
                        help='Parallel GPU eval envs per task (default: 4).')
    parser.add_argument('--logdir', default='~/logdir/sequential/eval',
                        help='Directory to write results.json.')
    parser.add_argument('--wandb_project', default='',
                        help='W&B project for logging eval metrics.')
    parser.add_argument('--wandb_entity',  default='', help='W&B entity.')
    parser.add_argument('--wandb_name',    default='', help='W&B run name.')
    args = parser.parse_args()

    logdir    = pathlib.Path(args.logdir).expanduser()
    ckpt_path = pathlib.Path(args.checkpoint).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    assert ckpt_path.exists(), f'Checkpoint not found: {ckpt_path}'

    # ── Build agent with train_task obs/act space + load weights ─────────────
    from dreamerv3.main import make_agent, make_env
    train_config = _build_config(args.train_task, args.configs)
    agent = make_agent(train_config)
    elements.checkpoint.load(
        str(ckpt_path),
        {'agent': bind(agent.load, regex=None)})
    print(f'Loaded checkpoint: {ckpt_path}  (train_task={args.train_task})')

    # Reference obs/act spaces (from train_task) used for compatibility checks.
    notlog = lambda k: not k.startswith('log/')
    ref_obs = {k: v for k, v in agent.obs_space.items() if notlog(k)}
    ref_act = agent.act_space

    # ── W&B (optional) ───────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=args.wandb_name or f'eval_{ckpt_path.parent.name}',
                dir=str(logdir),
                config={
                    'train_task': args.train_task,
                    'eval_tasks': args.tasks,
                    'checkpoint': str(ckpt_path),
                },
            )
        except Exception as exc:
            print(f'W&B init failed: {exc}')

    # ── Evaluate each task ────────────────────────────────────────────────────
    all_results = {}

    for task in args.tasks:
        print(f'\n{"=" * 60}\nEvaluating: {task}\n{"=" * 60}')

        eval_config = _build_config(task, args.configs)
        eval_env = make_env(eval_config, 0,
                            num_envs=args.eval_envs, is_eval=True)

        # ── Compatibility check (obs/act shapes must match train_task) ────────
        eval_obs = {k: v for k, v in eval_env.obs_space.items() if notlog(k)}
        eval_act = {k: v for k, v in eval_env.act_space.items() if k != 'reset'}

        obs_mismatch = [
            f'{k}: agent={ref_obs[k].shape} env={eval_obs[k].shape}'
            for k in ref_obs if k in eval_obs
            and ref_obs[k].shape != eval_obs[k].shape]
        act_mismatch = [
            f'{k}: agent={ref_act[k].shape} env={eval_act[k].shape}'
            for k in ref_act if k in eval_act
            and ref_act[k].shape != eval_act[k].shape]

        if obs_mismatch or act_mismatch:
            print(f'WARNING: {task} has incompatible spaces — skipping.')
            if obs_mismatch:
                print(f'  obs mismatch: {obs_mismatch}')
            if act_mismatch:
                print(f'  act mismatch: {act_mismatch}')
            eval_env.close()
            continue

        # ── Run eval ─────────────────────────────────────────────────────────
        try:
            metrics = _run_eval_episodes(
                agent, eval_env, args.eval_envs, args.episodes)
        finally:
            eval_env.close()

        all_results[task] = metrics
        shown = ', '.join(f'{k}: {v:.4g}' for k, v in metrics.items())
        print(f'{task} | {shown}')

        if wandb_run:
            import wandb
            wandb.log({f'{task}/{k}': v for k, v in metrics.items()})

    # ── Save results.json ─────────────────────────────────────────────────────
    results_path = logdir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {results_path}')

    # ── Print summary table ───────────────────────────────────────────────────
    if all_results:
        all_keys = sorted({k for m in all_results.values() for k in m})
        col = 20
        header = f"{'Task':<32}" + ''.join(f'{k:<{col}}' for k in all_keys)
        sep = '-' * len(header)
        print(f'\n{sep}\n{header}\n{sep}')
        for task, m in all_results.items():
            row = f'{task:<32}' + ''.join(
                f'{m.get(k, float("nan")):<{col}.4g}' for k in all_keys)
            print(row)
        print(sep)

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
