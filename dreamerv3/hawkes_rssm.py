"""Binary-event Hawkes RSSM for DreamerV3. Selected via `--dyn.typ hawkes`.

Observed causal chain (x_t never reaches h_t, only h_{t+1}):

  M_t = exp(-beta) M_{t-1} + alpha y_{t-1}
  lam_t = softplus(b + M_t + g_eta(h_{t-1}, a_{t-1}))
  h_t = core(h_{t-1}, z_{t-1}, a_{t-1}, omega([M_t, log1p(lam_t)]))
  z_t = RSSM posterior
  pi_t = sigmoid(D_psi(a_{t-1}, sg(u_t - u_{t-1})))
  y_t = straight-through Bernoulli(pi_t) -> reward, cont, Hawkes at t+1

Losses: Dreamer picks where events fire; `event_rate` is a two-sided KL
budget on mean(pi); `haw` fits b/alpha/beta/eta to detached detector
statistics and has zero gradient into the detector or the world model.

Hawkes scalars are float32 (the recurrence accumulates); activations stay in
COMPUTE_DTYPE. The carry is mixed precision -- never blanket-cast it.
"""

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient


def _inv_softplus(y):
  y = float(y)
  assert y > 0.0, y
  return float(y + np.log(-np.expm1(-y)))


def _logit_of(p):
  p = float(p)
  assert 0.0 < p < 1.0, p
  return float(np.log(p) - np.log1p(-p))


def _const_init(value):
  """Constant initializer matching the nn.Linear winit/binit protocol."""
  def init(shape, dtype=f32, fshape=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return jnp.full(shape, value, dtype)
  return init


class HawkesRSSM(nj.Module):

  # Inherited RSSM hyperparameters (same defaults as rssm.RSSM).
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  # Hawkes hyperparameters.
  obs_dim: int = 0                    # flattened encoder token width
  haw_hidden: int = 256
  haw_embed: int = 32
  haw_context_hidden: int = 64
  haw_target_rate: float = 0.05       # rho
  haw_rate_clip: float = 1e-3
  haw_eval_threshold: float = 0.5
  haw_init_alpha: float = 0.1
  haw_init_beta: float = 1.0
  haw_detector_outscale: float = 0.1  # keeps init pi tight around rho

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    assert self.obs_dim > 0, self.obs_dim
    assert 0.0 < self.haw_target_rate < 1.0, self.haw_target_rate
    self.act_space = act_space
    self.kw = kw
    rho = float(self.haw_target_rate)
    self._init_base = _inv_softplus(-np.log1p(-rho))  # 1-exp(-softplus(b))=rho
    self._init_alpha_raw = _inv_softplus(self.haw_init_alpha)
    self._init_beta_raw = _inv_softplus(self.haw_init_beta)
    self._init_det_bias = _logit_of(rho)

  # ---------------------------------------------------------------- carry --

  @property
  def entry_space(self):
    # `haw_prev_obs` is deliberately absent: obs_dim floats per stored step
    # would add tens of GB to replay. Restored chunks mask their first
    # transition instead.
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)),
        haw_state=elements.Space(np.float32),
        haw_prev=elements.Space(np.float32))

  def initial(self, bsize):
    # Mixed precision on purpose; do not wrap in nn.cast().
    return dict(
        deter=nn.cast(jnp.zeros([bsize, self.deter], f32)),
        stoch=nn.cast(jnp.zeros([bsize, self.stoch, self.classes], f32)),
        haw_state=jnp.zeros([bsize], f32),
        haw_prev=jnp.zeros([bsize], f32),
        haw_prev_obs=nn.cast(jnp.zeros([bsize, self.obs_dim], f32)),
        haw_prev_valid=jnp.zeros([bsize], bool))

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    assert carry is not None, 'HawkesRSSM.truncate needs the live carry'
    out = jax.tree.map(lambda x: x[:, -1], entries)
    out['deter'] = nn.cast(out['deter'])
    out['stoch'] = nn.cast(out['stoch'])
    out['haw_state'] = f32(out['haw_state'])
    out['haw_prev'] = f32(out['haw_prev'])
    out['haw_prev_obs'] = jnp.zeros_like(carry['haw_prev_obs'])
    out['haw_prev_valid'] = jnp.zeros_like(carry['haw_prev_valid'])
    return out

  def starts(self, entries, carry, nlast):
    # Imagination never reads haw_prev_obs, so it is not materialized here.
    B = len(jax.tree.leaves(carry)[0])
    keys = ('deter', 'stoch', 'haw_state', 'haw_prev')
    return {k: entries[k][:, -nlast:].reshape(
        (B * nlast, *entries[k].shape[2:])) for k in keys}

  # --------------------------------------------------------------- Hawkes --

  def _haw_params(self):
    """Scalar (b, alpha, beta) in float32; alpha, beta > 0."""
    base = self.value('haw_base', _const_init(self._init_base), ())
    alpha_raw = self.value(
        'haw_alpha_raw', _const_init(self._init_alpha_raw), ())
    beta_raw = self.value(
        'haw_beta_raw', _const_init(self._init_beta_raw), ())
    return base, jax.nn.softplus(alpha_raw), jax.nn.softplus(beta_raw)

  def _context(self, deter, action):
    """g_eta -> scalar f32. Accepts [B, ...] or [B, T, ...]."""
    x = jnp.concatenate([nn.cast(deter), nn.cast(action)], -1)
    x = self.sub('ctx0', nn.Linear, self.haw_context_hidden, **self.kw)(x)
    x = nn.act(self.act)(self.sub('ctx0norm', nn.Norm, self.norm)(x))
    kw = dict(**self.kw, outscale=0.0)  # g_eta output starts at zero
    x = self.sub('ctxout', nn.Linear, 1, **kw)(x)
    return f32(x[..., 0])

  def _hawkes_step(self, state, event, deter, action):
    base, alpha, beta = self._haw_params()
    state = jnp.exp(-beta) * state + alpha * event
    lam = jax.nn.softplus(base + state + self._context(deter, action))
    return state, lam, -jnp.expm1(-lam)

  def _hawkes_embed(self, state, lam):
    """omega([M_t, log1p(lam_t)]) -> [..., haw_embed] in compute dtype.

    No norm on the first projection: its input is 2-D and RMS norm is scale
    invariant, so it would divide out the intensity magnitude. Later norms
    act on wide vectors, where the informative direction survives.
    """
    x = nn.cast(jnp.stack([state, jnp.log1p(lam)], -1))
    x = nn.act(self.act)(
        self.sub('hemb0', nn.Linear, self.haw_hidden, **self.kw)(x))
    x = self.sub('hemb1', nn.Linear, self.haw_hidden, **self.kw)(x)
    x = nn.act(self.act)(self.sub('hemb1norm', nn.Norm, self.norm)(x))
    return self.sub('hembout', nn.Linear, self.haw_embed, **self.kw)(x)

  def _detector(self, action, delta):
    """pi_t from the previous action and the detached encoder delta only.

    symlog keeps the delta scale-preserving (an RMS norm would erase the
    magnitude that distinguishes an event); `mag` restores one unsquashed
    scale channel.
    """
    delta = sg(f32(delta))
    mag = jnp.log1p(jnp.sqrt(jnp.square(delta).mean(-1) + 1e-8))
    x = jnp.concatenate([
        nn.cast(action),
        nn.cast(nn.symlog(delta)),
        nn.cast(mag)[..., None]], -1)
    x = self.sub('det0', nn.Linear, self.haw_hidden, **self.kw)(x)
    x = nn.act(self.act)(self.sub('det0norm', nn.Norm, self.norm)(x))
    kw = dict(
        **self.kw, outscale=self.haw_detector_outscale,
        binit=_const_init(self._init_det_bias))
    logit = self.sub('detout', nn.Linear, 1, **kw)(x)
    return jax.nn.sigmoid(f32(logit[..., 0]))

  # ------------------------------------------------- observe / imagine ----

  def observe(self, carry, tokens, action, reset, training, single=False,
              sample_event=None):
    sample_event = training if sample_event is None else sample_event
    carry = dict(carry)
    carry['deter'] = nn.cast(carry['deter'])
    carry['stoch'] = nn.cast(carry['stoch'])
    carry['haw_prev_obs'] = nn.cast(carry['haw_prev_obs'])
    carry['haw_state'] = f32(carry['haw_state'])
    carry['haw_prev'] = f32(carry['haw_prev'])
    tokens, action = nn.cast((tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training, sample_event)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda c, inputs: self._observe(
              c, *inputs, training, sample_event),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training, sample_event):
    keep = ~reset
    valid = carry['haw_prev_valid'] & keep
    deter, stoch, haw_state, haw_prev = nn.mask(
        (carry['deter'], carry['stoch'],
         carry['haw_state'], carry['haw_prev']), keep)
    action = nn.mask(action, keep)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, keep)

    # Hawkes block: strictly t-1 information.
    haw_state, lam, prior_prob = self._hawkes_step(
        haw_state, haw_prev, deter, action)
    h_haw = self._hawkes_embed(haw_state, lam)

    deter = self._core(deter, stoch, action, h_haw)
    tokens_flat = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens_flat if self.absolute else jnp.concatenate(
        [deter, tokens_flat], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    # Detector sees o_t but only reaches h_{t+1}.
    delta = nn.mask(tokens_flat - carry['haw_prev_obs'], valid)
    prob = jnp.where(valid, self._detector(action, delta), 0.0)
    if sample_event:
      draw = f32(jax.random.bernoulli(nj.seed(), prob))
      event = sg(draw - prob) + prob
    else:
      event = f32(prob >= self.haw_eval_threshold)
    event = jnp.where(valid, event, 0.0)

    carry = dict(
        deter=deter, stoch=stoch,
        haw_state=haw_state, haw_prev=event,
        haw_prev_obs=tokens_flat,
        # Unconditional: episode and replay boundaries are handled by `reset`
        # and by truncate() respectively.
        haw_prev_valid=jnp.ones_like(carry['haw_prev_valid']))
    entry = dict(
        deter=deter, stoch=stoch, haw_state=haw_state, haw_prev=event)
    feat = dict(
        deter=deter, stoch=stoch, logit=logit,
        haw_prob=prob, haw_event=event, haw_head_event=event,
        haw_prior_prob=prior_prob, haw_lam=lam, haw_valid=f32(valid))
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      haw_state, lam, prior_prob = self._hawkes_step(
          carry['haw_state'], carry['haw_prev'], carry['deter'], actemb)
      h_haw = self._hawkes_embed(haw_state, lam)
      deter = self._core(carry['deter'], carry['stoch'], actemb, h_haw)
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      # Hard draw keeps the imagined event history on-distribution. No
      # straight-through: imagined features are detached before actor/critic.
      event = f32(jax.random.bernoulli(nj.seed(), prior_prob))
      carry = dict(
          deter=deter, stoch=stoch, haw_state=haw_state, haw_prev=event)
      feat = dict(
          deter=deter, stoch=stoch, logit=logit,
          haw_prob=prior_prob, haw_event=event,
          # Marginalization weight for the reward/cont heads: keeps their
          # input on the {0,1} support they were trained on, without the
          # return noise of a hard sample.
          haw_head_event=prior_prob,
          haw_prior_prob=prior_prob, haw_lam=lam,
          haw_valid=jnp.ones_like(prior_prob))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      # Drop the observation-only fields: report() hands us a full observe
      # carry, and the scan carry structure has to match what _observe of the
      # imagined step returns.
      carry = dict(
          deter=nn.cast(carry['deter']),
          stoch=nn.cast(carry['stoch']),
          haw_state=f32(carry['haw_state']),
          haw_prev=f32(carry['haw_prev']))
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            carry, (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            carry, nn.cast(policy), length, unroll=unroll, axis=1)
      return carry, feat, action

  # ------------------------------------------------------------------ loss --

  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    # Snapshot before observe() advances the carry; the fitting recurrence
    # below must start from the same place.
    init_state = f32(carry['haw_state'])
    init_event = f32(carry['haw_prev'])
    init_deter = nn.cast(carry['deter'])

    carry, entries, feat = self.observe(
        carry, tokens, acts, reset, training, sample_event=training)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()

    valid = feat['haw_valid']            # [B, T] f32 in {0, 1}
    prob = feat['haw_prob']              # [B, T] f32
    nvalid = jnp.maximum(1.0, valid.sum())

    # Event-rate budget. Two-sided: KL(Ber(rho) || Ber(mean pi)) diverges as
    # the rate goes to zero, so a collapsing detector is pushed back up.
    rho = float(self.haw_target_rate)
    clip = float(self.haw_rate_clip)
    rate = (valid * prob).sum() / nvalid
    rate_c = jnp.clip(rate, clip, 1.0 - clip)
    rate_kl = (
        rho * (np.log(rho) - jnp.log(rate_c)) +
        (1.0 - rho) * (np.log1p(-rho) - jnp.log1p(-rate_c)))
    losses['event_rate'] = jnp.broadcast_to(rate_kl, prob.shape)

    # Detached fitting recurrence: same forward values as the live one, but
    # event history and deter are detached so this trains only b/alpha/beta/eta.
    acts_m = nn.mask(acts, ~reset)
    actemb = nn.DictConcat(self.act_space, 1)(acts_m)
    actemb = nn.mask(actemb, ~reset)
    prepend = lambda init, x: jnp.concatenate([init[:, None], x[:, :-1]], 1)
    prev_event = sg(prepend(init_event, feat['haw_event']))
    prev_deter = sg(prepend(init_deter, feat['deter']))
    # Must reuse the live path's ~reset mask on M, y, h and a, or lambda_fit
    # and lambda_model diverge at every episode boundary inside the batch.
    ctx = self._context(nn.mask(prev_deter, ~reset), actemb)

    base, alpha, beta = self._haw_params()
    decay = jnp.exp(-beta)

    def fit_step(state, xs):
      y_prev, k = xs
      state = jnp.where(k, state, 0.0)
      y_prev = jnp.where(k, y_prev, 0.0)
      state = decay * state + alpha * y_prev
      return state, state

    _, mfit = jax.lax.scan(
        fit_step, init_state,
        (prev_event.swapaxes(0, 1), (~reset).swapaxes(0, 1)))
    mfit = mfit.swapaxes(0, 1)
    lam_fit = jax.nn.softplus(base + mfit + ctx)
    p_fit = -jnp.expm1(-lam_fit)

    q = jnp.clip(sg(prob), 1e-6, 1.0 - 1e-6)
    p = jnp.clip(p_fit, 1e-6, 1.0 - 1e-6)
    losses['haw'] = valid * (
        q * (jnp.log(q) - jnp.log(p)) +
        (1.0 - q) * (jnp.log1p(-q) - jnp.log1p(-p)))

    wmean = lambda x: (valid * x).sum() / nvalid
    metrics['haw_rate'] = rate
    metrics['haw_rate_error'] = rate - rho
    metrics['haw_event_rate'] = wmean(feat['haw_event'])
    metrics['haw_prior_rate'] = wmean(feat['haw_prior_prob'])
    metrics['haw_post_prior_gap'] = wmean(
        jnp.abs(prob - feat['haw_prior_prob']))
    metrics['haw_prob_ent'] = wmean(
        -(q * jnp.log(q) + (1.0 - q) * jnp.log1p(-q)))
    metrics['haw_prob_std_time'] = prob.std(1).mean()
    metrics['haw_lam_mean'] = wmean(feat['haw_lam'])
    metrics['haw_lam_max'] = feat['haw_lam'].max()
    metrics['haw_ctx_std'] = ctx.std()
    metrics['haw_base'] = base
    metrics['haw_alpha'] = alpha
    metrics['haw_beta'] = beta
    metrics['haw_valid_frac'] = valid.mean()
    # Live guard on the fitting invariant; nonzero means a misalignment in
    # prepend, the reset mask, or the initial carry.
    metrics['haw_lam_fit_err'] = (
        valid * jnp.abs(lam_fit - feat['haw_lam'])).max()

    return carry, entries, losses, feat, metrics

  # ------------------------------------------------------------------ core --

  def _core(self, deter, stoch, action, h_haw=None):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))

    streams = [x0, x1, x2]
    if h_haw is not None:
      # Norm kept here: input is `hidden`-dimensional, so it balances this
      # stream against the other three without erasing the direction.
      x3 = self.sub('dynin3', nn.Linear, self.hidden, **self.kw)(h_haw)
      x3 = nn.act(self.act)(self.sub('dynin3norm', nn.Norm, self.norm)(x3))
      streams.append(x3)

    x = jnp.concatenate(streams, -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out
