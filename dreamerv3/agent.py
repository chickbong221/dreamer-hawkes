import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm
from . import hawkes_rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype in (np.uint8, np.uint16) and len(s.shape) == 3


def _dyn_policy_kw(hawkes, mode):
  """Dynamics kwargs for a policy step.

  `mode` deliberately does not appear: evaluation samples events exactly like
  training, so every recurrent path sees the same event distribution. The 0.5
  threshold survives only as an explicit diagnostic via `sample_event=False`.
  """
  del mode
  kw = dict(training=False, single=True)
  if hawkes:
    kw['sample_event'] = True
  return kw


def _bern_ent(p):
  """Mean Bernoulli entropy in nats; low means the events are sharp."""
  p = jnp.clip(f32(p), 1e-6, 1.0 - 1e-6)
  return -(p * jnp.log(p) + (1.0 - p) * jnp.log1p(-p)).mean()


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    self.hawkes = config.dyn.typ == 'hawkes'

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    dyn_cls = {
        'rssm': rssm.RSSM,
        'hawkes': hawkes_rssm.HawkesRSSM,
    }[config.dyn.typ]
    if config.dyn.typ == 'hawkes':
      self.dyn = dyn_cls(
          act_space, obs_dim=self.enc.output_size,
          **config.dyn[config.dyn.typ], name='dyn')
    else:
      self.dyn = dyn_cls(
          act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')
    # Auxiliary long-horizon reward head. An ordinary MLP -- no Hawkes
    # process, no recurrence, no intensity, no memory, and it never sees
    # p_t, M_t or lam_t. It exists only under dyn.typ == hawkes because its
    # inputs (y_t and sg(y_t) c^ST_t) exist only there. Same output
    # distribution as `rew`, separate parameters.
    self.future_rew = embodied.jax.MLPHead(
        scalar, **config.rewhead, name='future_rew') if self.hawkes else None

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    if self.hawkes:
      self.modules.append(self.future_rew)
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    if not self.hawkes:
      for key in ('haw', 'event_rate', 'event_conf', 'event_use',
                  'event_type_prior', 'event_future_rew'):
        scales.pop(key, None)
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **_dyn_policy_kw(self.hawkes, mode))
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    # Surface Hawkes per-step features so the evaluation event logger
    # (dreamerv3/log_event_episode.py) can capture them. Gated on mode='eval'
    # because the train-mode driver feeds `outs` into the replay buffer,
    # and the train assertion `data.keys() == self.spaces.keys()` rejects
    # any extra fields. eval-mode outs are consumed locally and discarded.
    if self.hawkes and mode == 'eval':
      for key in ('haw_prob', 'haw_event', 'haw_prior_prob',
                  'haw_type_prob'):
        out[key] = feat[key]
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)

    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training=training)

    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self._headfeat(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    # Plain [h, z]: ManiSkill never signals task termination, so the target
    # is effectively constant and any event input to this head would shape
    # event placement and typing against a signal that carries nothing.
    # Continuation still trains the ordinary RSSM features.
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      if isimage(space):
        scale = 255.0 if space.dtype == np.uint8 else float(
            np.asarray(space.high).max())
        target = f32(value) / scale
      else:
        target = value
      losses[key] = recon.loss(sg(target))

    if self.hawkes:
      los, mets = self._future_loss(repfeat, obs)
      losses['event_future_rew'] = los
      metrics.update(mets)

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    if self.hawkes:
      # Imagined counterparts of the observed metrics. The observed/imagined
      # rate gap is the deployment diagnostic: observed events come from the
      # detector, imagined ones from the Hawkes prior.
      metrics.update({
          'event_rate_img': f32(imgfeat['haw_prob']).mean(),
          'event_hard_rate_img': f32(imgfeat['haw_event']).mean(),
          'event_prob_entropy_img': _bern_ent(imgfeat['haw_prob']),
          'event_prob_std_time_img': f32(imgfeat['haw_prob']).std(1).mean(),
          'event_delta_mag_img': f32(imgfeat['haw_delta_mag']).mean()})
      cbar = f32(imgfeat['haw_type_prior_prob']).mean((0, 1))
      metrics.update({
          f'event_type_usage_img/{k}': cbar[k]
          for k in range(cbar.shape[-1])})
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    rewmix, rewspread = self._headmix(
        self.rew, imgfeat, lambda o: o.pred(), spread=True)
    # No marginalization here: the continuation head reads [h, z] only, so
    # there is nothing to marginalize over.
    conmix = self.con(inp, 2).prob(1)
    if self.hawkes:
      # Near zero means the reward head is reading the binary event and
      # ignoring which type it was.
      metrics['event_type_reward_spread'] = rewspread
    los, imgloss_out, mets = imag_loss(
        imgact,
        rewmix,
        conmix,
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def _headfeat(self, feat):
    """Base feature, plus the binary event and its typed one-hot.

    Both are read back out of `haw_head_mix`, which is one-hot on the realized
    outcome while observing: y from channel 0, y*c from the rest. y carries
    gradient to the detector; y*c carries gradient to the classifier only,
    because the binary factor there is stop-gradded.
    """
    x = self.feat2tensor(feat)
    if not self.hawkes:
      return x
    mix = nn.cast(feat['haw_head_mix'])
    return jnp.concatenate([x, 1.0 - mix[..., :1], mix[..., 1:]], -1)

  def _futurefeat(self, feat):
    """[sg(z_t), y_t, sg(y_t) c^ST_t] -- the auxiliary head's input.

    No h_t and a detached z_t: this head must not be able to reshape the RSSM
    to make a long-horizon target easier to hit. What it does reach is the
    event itself, which is the point:

      L_future -> y_t          -> D_psi   (binary, straight through)
      L_future -> sg(y_t) c^ST -> C_phi   (type, only on frames that fired)

    Both channels are read back out of `haw_head_mix` exactly as `_headfeat`
    reads them, so the two heads see the same event encoding.
    """
    z = nn.cast(sg(feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1))))
    mix = nn.cast(feat['haw_head_mix'])
    return jnp.concatenate([z, 1.0 - mix[..., :1], mix[..., 1:]], -1)

  def _future_loss(self, repfeat, obs):
    """Auxiliary H-step future-return regression, plus its probes.

    Supervised only on frames with a complete, reset-free horizon, and
    normalized by that count, so the loss does not shrink when a batch
    happens to hold fewer of them. Returns a [B, T] broadcast of the scalar
    so it drops into `losses` like every other term.

    Report-only, never fed back: this prediction is not added to the imagined
    reward, not shown to actor or critic, not an intrinsic reward, not part
    of any value target, and it does not touch the Hawkes memory.
    """
    H = int(self.config.event_future_horizon)
    disc = 1 - 1 / self.config.horizon
    target, umask = future_return_target(
        obs['reward'], obs['is_first'], H, disc)
    out = self.future_rew(self._futurefeat(repfeat), 2)
    total = umask.sum()
    denom = jnp.maximum(total, 1.0)
    # A batch with no full-horizon frame takes the constant branch, so the
    # loss is an exact zero with an exact zero gradient, not 0/eps.
    value = jnp.where(
        total > 0, (umask * out.loss(sg(target))).sum() / denom, 0.0)
    metrics = {
        'event_future_loss': value,
        'event_future_target_mean': (umask * target).sum() / denom,
        'event_future_pred_mean': (umask * f32(out.pred())).sum() / denom,
        'event_future_full_horizon_frac': umask.mean()}
    metrics.update(self._future_probes(repfeat))
    return jnp.broadcast_to(value, umask.shape), metrics

  def _future_probes(self, repfeat):
    """Does the auxiliary head read the event, and does it read the type.

    S_y = E|F(z, 1, c) - F(z, 0, 0)| over frames that fired, and
    S_c = E[max_k F(z, 1, e_k) - min_k F(z, 1, e_k)] over valid frames.
    Both fall out of the same K + 1 evaluations, because `haw_head_mix`
    channels 1: are exactly onehot(c_t) wherever y_t = 1. Detached: a metric
    must not open a gradient route.
    """
    z = nn.cast(sg(repfeat['stoch'].reshape(
        (*repfeat['stoch'].shape[:-2], -1))))
    mix = f32(sg(repfeat['haw_head_mix']))
    K = mix.shape[-1] - 1
    zeros = jnp.zeros((*z.shape[:-1], 1 + K), z.dtype)
    at = lambda chan: f32(
        self.future_rew(jnp.concatenate([z, chan], -1), 2).pred())
    noevent = at(zeros)
    typed = jnp.stack(
        [at(zeros.at[..., 0].set(1).at[..., 1 + k].set(1)) for k in range(K)],
        -1)
    fired = 1.0 - mix[..., 0]
    valid = f32(repfeat['haw_valid'])
    sel = (mix[..., 1:] * typed).sum(-1)
    return {
        'event_future_sens_binary': (
            (fired * jnp.abs(sel - noevent)).sum() /
            jnp.maximum(fired.sum(), 1.0)),
        'event_future_sens_type': (
            (valid * (typed.max(-1) - typed.min(-1))).sum() /
            jnp.maximum(valid.sum(), 1.0)),
    }

  def _headmix(self, head, feat, readout, spread=False):
    """E[readout(head)] over the K + 1 event outcomes.

    Evaluates the head at no-event and at each typed event, so it never leaves
    the support it was trained on. `haw_head_mix` is one-hot on the realized
    outcome for observed steps and the joint (1 - p, p c_k) for imagined ones,
    so this is exact in both cases. `spread` additionally returns the range of
    the head across the K typed outcomes -- near zero means the head is
    ignoring the type.

    Unrolled rather than folded into one batched call: the K + 1 copies of
    `base` would be the largest activation in the imagination path.

    Only the reward head uses this. The continuation head reads [h, z], so
    there is no event to marginalize out of it.
    """
    if not self.hawkes:
      out = readout(head(self.feat2tensor(feat), 2))
      return (out, jnp.zeros((), f32)) if spread else out
    base = self.feat2tensor(feat)
    w = f32(feat['haw_head_mix'])
    K = w.shape[-1] - 1
    zeros = jnp.zeros((*base.shape[:-1], 1 + K), base.dtype)
    outs = []
    for k in range(K + 1):
      # Channel 0 is y; channels 1: are y * onehot(type).
      chan = zeros if k == 0 else zeros.at[..., 0].set(1).at[..., k].set(1)
      outs.append(f32(readout(head(jnp.concatenate([base, chan], -1), 2))))
    stack = jnp.stack(outs, 0)
    mixed = (jnp.moveaxis(w, -1, 0) * stack).sum(0)
    if spread:
      ev = stack[1:]
      return mixed, (ev.max(0) - ev.min(0)).mean()
    return mixed

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    metrics.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    def to_report_rgb(x):
      # ManiSkill can flatten multiple RGB cameras into the channel dimension
      # (for example, 2 cameras -> 6 channels). The logger/report video path
      # expects displayable RGB, so tile camera views horizontally for summaries
      # while leaving the model input/reconstruction tensors unchanged.
      channels = x.shape[-1]
      if channels == 3:
        return x
      if channels % 3 == 0:
        views = channels // 3
        B_, T_, H_, W_, _ = x.shape
        x = x.reshape((B_, T_, H_, W_, views, 3))
        x = x.transpose((0, 1, 2, 4, 3, 5))
        return x.reshape((B_, T_, H_, views * W_, 3))
      if channels == 1:
        return jnp.repeat(x, 3, axis=-1)
      return x[..., :3]

    for key in self.dec.imgkeys:
      space = self.obs_space[key]
      assert obs[key].dtype in (jnp.uint8, jnp.uint16)
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      # `pred` is the decoder's sigmoid output in [0, 1] regardless of dtype.
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      if true.dtype == jnp.uint16:
        # uint16 depth in mm → rescale by space.high before quantising to uint8.
        hi = float(np.asarray(space.high).max())
        true = jnp.clip(true.astype(jnp.float32) * (255.0 / hi),
                        0, 255).astype(jnp.uint8)
      true = to_report_rgb(true)
      pred = to_report_rgb(pred)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      T_vid = video.shape[1]
      border = jnp.full((T_vid, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T_vid // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def future_return_target(rew, first, horizon, disc):
  """Discount-normalized H-step future return and its full-horizon mask.

    G_t = sum_{k=1..H} disc^(k-1) r_{t+k} / sum_{k=1..H} disc^(k-1)
    u_t = 1[t + H < T] * prod_{j=1..H} (1 - is_first_{t+j})

  Strict on purpose. A frame is supervised only when all H of its future
  rewards are present in this replay sequence and none of them belongs to a
  later episode. The horizon is never shortened at the tail, missing rewards
  are never zero filled, and `is_terminal` plays no part in availability --
  a truncated window is a different, easier target that would drift with the
  batch length. The sum starts at r_{t+1}, not r_t.
  """
  assert horizon >= 1, horizon
  rew, first = f32(rew), f32(first)
  B, T = rew.shape
  H = int(horizon)
  if H >= T:
    zeros = jnp.zeros((B, T), f32)
    return zeros, zeros
  w = np.asarray([disc ** k for k in range(H)], np.float32)
  num = jnp.zeros((B, T - H), f32)
  alive = jnp.ones((B, T - H), f32)
  for k in range(1, H + 1):
    num += w[k - 1] * rew[:, k: T - H + k]
    alive *= 1.0 - first[:, k: T - H + k]
  pad = jnp.zeros((B, H), f32)
  return (
      jnp.concatenate([num / w.sum(), pad], 1),
      jnp.concatenate([alive, pad], 1))


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
