import importlib
import os
import pathlib
import sys
from functools import partial as bind

# Must be set before JAX initialises to share GPU with ManiSkill cleanly.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml

def main(argv=None):
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=os.path.expanduser(
      config.logdir.format(timestamp=elements.timestamp())))

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  suite = config.task.split('_', 1)[0]

  if suite == 'behavior1k':
    args = args.update(envs=1, eval_envs=1, debug=False)
    if config.script == 'train':
      embodied.run.train(
          bind(make_agent, config),
          bind(make_replay, config, 'replay'),
          bind(make_env, config),
          bind(make_stream, config),
          bind(make_logger, config),
          args)
    else:
      raise NotImplementedError(
          f'behavior1k only supports script=train, got {config.script}')
    return

  if suite == 'maniskill':
    num_envs = dict(config.env.get('maniskill', {})).get('num_envs', 32)
    args = args.update(envs=1, debug=False)  # one env call, no subprocess

    _OrigDriver = embodied.Driver

    class _ManiSkillDriver(_OrigDriver):
      def __init__(self, make_env_fns, parallel=True, **kwargs):
        # fns[0]() returns the ManiSkill batched env instance
        env = make_env_fns[0]()
        _OrigDriver.__init__(
            self, env,
            batched=True,
            num_envs=num_envs,
            **kwargs,
        )

    embodied.Driver = _ManiSkillDriver
    try:
      if config.script == 'train':
        embodied.run.train(
            bind(make_agent, config),
            bind(make_replay, config, 'replay'),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args)
      else:
        raise NotImplementedError(
            f'ManiSkill only supports script=train, got {config.script}')
    finally:
      embodied.Driver = _OrigDriver
    return

  if config.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from .agent import Agent
  suite = config.task.split('_', 1)[0]
  # For ManiSkill: build with num_envs=1 just for obs/act space discovery.
  # This avoids spinning up 32 GPU envs twice.
  if suite == 'maniskill':
    env = make_env(config, 0, num_envs=1)
  else:
    env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []

  # Keep ManiSkill metrics visible to terminal / normal logger outputs.
  suite = config.task.split('_', 1)[0]
  log_filter = config.logger.filter
  if suite == 'maniskill':
    log_filter = log_filter + '|episode/score|epstats/log/|fps/'

  outputs.append(elements.logger.TerminalOutput(log_filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))

    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))

    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))

    elif output == 'wandb':
      import os
      import wandb as _wandb
      lc = config.logger
      run_name = lc.get('wandb_name') or '/'.join(logdir.split('/')[-4:])
      os.environ['WANDB_PROJECT'] = lc.get('wandb_project', 'dreamerv3')
      if lc.get('wandb_entity'):
        os.environ['WANDB_ENTITY'] = lc.get('wandb_entity')
      os.environ['WANDB_NAME'] = run_name
      if lc.get('wandb_group'):
        os.environ['WANDB_RUN_GROUP'] = lc.get('wandb_group')
      os.environ['WANDB_RESUME'] = 'allow'
      os.environ['WANDB_DIR'] = logdir
      # Init wandb first with the desired name so that WandBOutput reuses
      # this run rather than starting a new one with the logdir path as name.
      _wandb.init(
          project=lc.get('wandb_project', 'dreamerv3'),
          entity=lc.get('wandb_entity') or None,
          name=run_name,
          group=lc.get('wandb_group') or None,
          dir=logdir,
          config=dict(config),
          resume='allow',
      )
      outputs.append(elements.logger.WandBOutput(
          '/'.join(logdir.split('/')[-4:])))
      _wandb.run.config.update(dict(config), allow_val_change=True)

    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))

    else:
      raise NotImplementedError(output)

  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy':     'embodied.envs.dummy:Dummy',
      'gym':       'embodied.envs.from_gym:FromGym',
      'dm':        'embodied.envs.from_dmenv:FromDM',
      'crafter':   'embodied.envs.crafter:Crafter',
      'dmc':       'embodied.envs.dmc:DMC',
      'atari':     'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab':     'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav':   'embodied.envs.loconav:LocoNav',
      'pinpad':    'embodied.envs.pinpad:PinPad',
      'langroom':  'embodied.envs.langroom:LangRoom',
      'procgen':   'embodied.envs.procgen:ProcGen',
      'bsuite':    'embodied.envs.bsuite:BSuite',
      'memmaze':   lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
      # ManiSkill: builds the batched GPU env directly.
      # Skips wrap_env — batched obs are not compatible with scalar wrappers.
      'maniskill': 'embodied.envs.maniskill:ManiSkill',
      'behavior1k': 'embodied.envs.omnigibson:OmniGibson',
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = dict(config.env.get(suite, {}))   # mutable copy
  kwargs.update(overrides)                   # e.g. num_envs=1 for space discovery
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
  env = ctor(task, **kwargs)
  if suite in ('maniskill', 'behavior1k'):
    return env   # no scalar wrappers: obs spaces are already correct
  return wrap_env(env, config)


def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)
  return stream


if __name__ == '__main__':
  print("Using Hawkes")
  main()