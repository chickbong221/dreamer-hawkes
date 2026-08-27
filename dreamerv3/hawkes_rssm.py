"""Binary-event Hawkes RSSM for DreamerV3. Selected via `--dyn.typ hawkes`.

Two networks with separate jobs. A posterior detector picks the events from
what actually happened; a Hawkes prior, placed after the world-model update,
learns to predict them and is what imagination deploys.

  e^H_{t-1} = omega(M_{t-1}, log1p(lam_{t-1}))
  h_t       = core(h_{t-1}, z_{t-1}, a_{t-1}, e^H_{t-1})
  l_t       = posterior(h_t, x_t) observing, prior(h_t) imagining
  z_t       ~ Categorical(l_t)
  r_t       = log[(1 - u) softmax(l_t) + u / C]   canonical, shift invariant
  d_t       = sg(r_t - r_{t-1})                   posterior delta observing,
                                                  prior delta imagining
  Mbar_t    = exp(-beta) M_{t-1}
  lam_t     = softplus(b + Mbar_t + g_eta(sg(h_t), symlog(d_t), m_t))
  p_t       = 1 - exp(-lam_t)                     the Hawkes prediction
  q_t       = sigmoid(D_psi(a_{t-1}, symlog(sg(dx_t)), m^x_t))   the detector
  y_t       = straight-through Bernoulli(q_t) observing, Bernoulli(p_t) imagining
  M_t       = Mbar_t + alpha y_t                  -> h_{t+1}

y_t reaches h_{t+1}, never h_t. Observed events always come from the detector,
imagined events always from the Hawkes prior.

Teacher forcing on purpose: g_eta reads the *posterior* delta while observing
and the *prior* delta while imagining, the way Dreamer's reward head trains on
posterior features and runs on prior ones. The resulting input gap is measured
rather than designed away -- see `_probes`, report-only.

No sampled z_t enters the Hawkes context, in either path. The event therefore
does not depend on the particular current categorical sample, only on how the
distribution moved.

Two masks, kept logically separate:
  keep  = ~reset                    masks the incoming h, z, M, lam.
  valid = keep & haw_prev_valid     gates the detector delta, q_t, y_t, the
                                    rate loss and the Hawkes KL. False at a
                                    restored replay boundary, where there is no
                                    previous encoder token -- but the restored
                                    Hawkes state is still used, not cleared.

Losses: Dreamer's reward/cont heads and the delayed y_t -> M_t -> h_{t+1} path
decide where events fire; `event_rate` is a two-sided KL budget on mean(q);
`haw` fits b/alpha/beta/g_eta to detached detector statistics through an
explicit recurrence, and has zero gradient into the detector or the RSSM.

Hawkes scalars and the latent log-probs are float32 -- the recurrence
accumulates, and differencing log-probs near log(u/C) underflows bfloat16.
Activations stay in COMPUTE_DTYPE. The carry is mixed precision: never
blanket-cast it.
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
  haw_context_hidden: int = 256
  haw_target_rate: float = 0.05       # rho
  haw_rate_clip: float = 1e-3
  haw_eval_threshold: float = 0.5
  haw_init_alpha: float = 0.1
  haw_init_beta: float = 1.0
  haw_detector_outscale: float = 0.1  # nonzero: frames must differ at init

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    assert self.obs_dim > 0, self.obs_dim
    assert 0.0 < self.haw_target_rate < 1.0, self.haw_target_rate
    # _event_repr takes a log; the unimix floor is what keeps it finite.
    assert self.unimix > 0.0, self.unimix
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
    # `haw_prev_obs` and `haw_prev_repr` are deliberately absent: obs_dim and
    # stoch * classes floats per stored step would add tens of GB to replay.
    # Restored chunks mask their first transition instead, and starts() rebuilds
    # the representation from the prior.
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)),
        haw_state=elements.Space(np.float32),
        haw_lam=elements.Space(np.float32))

  def initial(self, bsize):
    # Mixed precision on purpose; do not wrap in nn.cast().
    return dict(
        deter=nn.cast(jnp.zeros([bsize, self.deter], f32)),
        stoch=nn.cast(jnp.zeros([bsize, self.stoch, self.classes], f32)),
        haw_state=jnp.zeros([bsize], f32),
        haw_lam=jnp.zeros([bsize], f32),
        haw_prev_obs=nn.cast(jnp.zeros([bsize, self.obs_dim], f32)),
        haw_prev_repr=jnp.zeros([bsize, self.stoch, self.classes], f32),
        haw_prev_valid=jnp.zeros([bsize], bool))

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    assert carry is not None, 'HawkesRSSM.truncate needs the live carry'
    out = jax.tree.map(lambda x: x[:, -1], entries)
    out['deter'] = nn.cast(out['deter'])
    out['stoch'] = nn.cast(out['stoch'])
    # The restored Hawkes state is kept: an unavailable encoder delta
    # invalidates the detector, not the event history.
    out['haw_state'] = f32(out['haw_state'])
    out['haw_lam'] = f32(out['haw_lam'])
    out['haw_prev_obs'] = jnp.zeros_like(carry['haw_prev_obs'])
    out['haw_prev_repr'] = jnp.zeros_like(carry['haw_prev_repr'])
    out['haw_prev_valid'] = jnp.zeros_like(carry['haw_prev_valid'])
    return out

  def starts(self, entries, carry, nlast):
    # Imagination never reads haw_prev_obs, so it is not materialized here.
    B = len(jax.tree.leaves(carry)[0])
    keys = ('deter', 'stoch', 'haw_state', 'haw_lam')
    out = {k: entries[k][:, -nlast:].reshape(
        (B * nlast, *entries[k].shape[2:])) for k in keys}
    # Imagination is prior-to-prior throughout, so rebuilding r from the prior
    # is the consistent choice here, not merely the cheap one.
    out['haw_prev_repr'] = self._event_repr(self._prior(nn.cast(out['deter'])))
    out['haw_prev_valid'] = jnp.ones(B * nlast, bool)
    return out

  # --------------------------------------------------------------- Hawkes --

  def _haw_params(self):
    """Scalar (b, alpha, beta) in float32; alpha, beta > 0."""
    base = self.value('haw_base', _const_init(self._init_base), ())
    alpha_raw = self.value(
        'haw_alpha_raw', _const_init(self._init_alpha_raw), ())
    beta_raw = self.value(
        'haw_beta_raw', _const_init(self._init_beta_raw), ())
    return base, jax.nn.softplus(alpha_raw), jax.nn.softplus(beta_raw)

  def _event_repr(self, logit):
    """Canonical log-probabilities of the unimixed categorical, float32.

    softmax(l) == softmax(l + c), so raw logit differences would report a
    change where the distribution is identical. Kept in f32 because the
    downstream difference of two values near log(unimix / classes) loses most
    of its precision in bfloat16.
    """
    prob = jax.nn.softmax(f32(logit), -1)
    prob = (1.0 - self.unimix) * prob + self.unimix / self.classes
    return jnp.log(prob)

  def _delta(self, repr_t, prev_repr, valid):
    """Detached canonical delta and its per-group magnitude, both float32.

    An invalid delta and a genuinely zero one give the same inputs on purpose:
    both mean "no latent movement to report".
    """
    delta = sg(repr_t - f32(prev_repr))
    delta = jnp.where(valid[:, None, None], delta, 0.0)
    return delta, self._mag(delta)

  def _mag(self, delta):
    """Per-group scale of a delta, preserved through the later norms."""
    return jnp.log1p(jnp.sqrt(jnp.square(delta).mean(-1) + 1e-8))

  def _context(self, deter, delta, mag):
    """g_eta -> scalar f32. Accepts [B, ...] or [B, T, ...].

    Inputs are detached: the Hawkes KL reads world-model values but must not
    reshape the RSSM to make its own target easier. No sampled z_t here, in
    either path -- only how the categorical distribution moved.
    """
    flat = lambda x: x.reshape((*x.shape[:-2], -1))
    x = jnp.concatenate([
        nn.cast(sg(deter)),
        nn.cast(nn.symlog(flat(sg(delta)))),
        nn.cast(mag)], -1)
    x = self.sub('ctx0', nn.Linear, self.haw_context_hidden, **self.kw)(x)
    x = nn.act(self.act)(self.sub('ctx0norm', nn.Norm, self.norm)(x))
    kw = dict(**self.kw, outscale=0.0)  # g_eta output starts at zero
    x = self.sub('ctxout', nn.Linear, 1, **kw)(x)
    return f32(x[..., 0])

  def _hawkes_probs(self, state, ctx):
    """Decay the memory, then score: (Mbar_t, lam_t, p_t)."""
    base, _, beta = self._haw_params()
    mbar = jnp.exp(-beta) * state
    lam = jax.nn.softplus(base + mbar + ctx)
    return mbar, lam, -jnp.expm1(-lam)

  def _hawkes_update(self, mbar, event):
    _, alpha, _ = self._haw_params()
    return mbar + alpha * event

  def _detector(self, action, delta):
    """q_t from the previous action and the detached encoder delta only.

    symlog keeps the delta scale-preserving (an RMS norm would erase the
    magnitude that distinguishes an event); `mag` restores one unsquashed
    scale channel. The output layer is deliberately not zero-initialized:
    different visual transitions must score differently from step one, or the
    rate budget is already satisfied by a constant and nothing breaks the
    symmetry. `logit(rho)` is the layer's bias -- do not add it again outside.
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

  def _hawkes_embed(self, state, lam):
    """omega([M_{t-1}, log1p(lam_{t-1})]) -> [..., haw_embed], compute dtype.

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

  # ------------------------------------------------- observe / imagine ----

  def observe(self, carry, tokens, action, reset, training, single=False,
              sample_event=None):
    sample_event = training if sample_event is None else sample_event
    carry = dict(carry)
    carry['deter'] = nn.cast(carry['deter'])
    carry['stoch'] = nn.cast(carry['stoch'])
    carry['haw_prev_obs'] = nn.cast(carry['haw_prev_obs'])
    carry['haw_state'] = f32(carry['haw_state'])
    carry['haw_lam'] = f32(carry['haw_lam'])
    carry['haw_prev_repr'] = f32(carry['haw_prev_repr'])
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
    deter, stoch, haw_state, haw_lam = nn.mask(
        (carry['deter'], carry['stoch'],
         carry['haw_state'], carry['haw_lam']), keep)
    action = nn.mask(action, keep)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, keep)

    # omega(0, 0) is the learned no-history token; a reset lands on it because
    # both carried scalars were just masked.
    h_haw = self._hawkes_embed(haw_state, haw_lam)
    deter = self._core(deter, stoch, action, h_haw)

    tokens_flat = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens_flat if self.absolute else jnp.concatenate(
        [deter, tokens_flat], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    # Hawkes prediction, teacher forced on the posterior delta.
    repr_t = self._event_repr(logit)
    delta, mag = self._delta(repr_t, carry['haw_prev_repr'], valid)
    ctx = self._context(deter, delta, mag)
    mbar, lam, prior_prob = self._hawkes_probs(haw_state, ctx)

    # Detector sees o_t but only reaches h_{t+1}.
    dx = nn.mask(tokens_flat - carry['haw_prev_obs'], valid)
    prob = jnp.where(valid, self._detector(action, dx), 0.0)
    if sample_event:
      draw = f32(jax.random.bernoulli(nj.seed(), prob))
      event = sg(draw - prob) + prob
    else:
      event = f32(prob >= self.haw_eval_threshold)
    event = jnp.where(valid, event, 0.0)
    haw_state = self._hawkes_update(mbar, event)

    carry = dict(
        deter=deter, stoch=stoch, haw_state=haw_state, haw_lam=lam,
        haw_prev_obs=tokens_flat,
        # r_t is stored unconditionally, including at a reset, so the next step
        # has a real delta. Replay boundaries are handled by truncate().
        haw_prev_repr=repr_t,
        haw_prev_valid=jnp.ones_like(carry['haw_prev_valid']))
    entry = dict(
        deter=deter, stoch=stoch, haw_state=haw_state, haw_lam=lam)
    feat = dict(
        deter=deter, stoch=stoch, logit=logit,
        haw_prob=prob, haw_event=event, haw_head_event=event,
        haw_prior_prob=prior_prob, haw_lam=lam, haw_ctx=ctx,
        haw_delta_mag=mag.mean(-1), haw_valid=f32(valid))
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      h_haw = self._hawkes_embed(carry['haw_state'], carry['haw_lam'])
      deter = self._core(carry['deter'], carry['stoch'], actemb, h_haw)
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      repr_t = self._event_repr(logit)
      delta, mag = self._delta(
          repr_t, carry['haw_prev_repr'], carry['haw_prev_valid'])
      ctx = self._context(deter, delta, mag)
      mbar, lam, prior_prob = self._hawkes_probs(carry['haw_state'], ctx)
      # Hard draw keeps the imagined event history on-distribution. No
      # straight-through: imagined features are detached before actor/critic.
      event = f32(jax.random.bernoulli(nj.seed(), prior_prob))
      haw_state = self._hawkes_update(mbar, event)
      carry = dict(
          deter=deter, stoch=stoch, haw_state=haw_state, haw_lam=lam,
          haw_prev_repr=repr_t,
          haw_prev_valid=jnp.ones_like(carry['haw_prev_valid']))
      feat = dict(
          deter=deter, stoch=stoch, logit=logit,
          haw_prob=prior_prob, haw_event=event,
          # Marginalization weight for the reward/cont heads: keeps their
          # input on the {0, 1} support they were trained on, without the
          # return noise of a hard sample.
          haw_head_event=prior_prob,
          haw_prior_prob=prior_prob, haw_lam=lam, haw_ctx=ctx,
          haw_delta_mag=mag.mean(-1), haw_valid=jnp.ones_like(prior_prob))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      # Drop the observation-only field: report() hands us a full observe
      # carry, and the scan carry structure has to match what the imagined
      # step returns.
      carry = dict(
          deter=nn.cast(carry['deter']),
          stoch=nn.cast(carry['stoch']),
          haw_state=f32(carry['haw_state']),
          haw_lam=f32(carry['haw_lam']),
          haw_prev_repr=f32(carry['haw_prev_repr']),
          haw_prev_valid=carry['haw_prev_valid'].astype(bool))
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

    # valid = keep & haw_prev_valid. Gates the detector-derived terms only;
    # the recurrent state is masked separately, by keep alone.
    valid = feat['haw_valid']            # [B, T] f32 in {0, 1}
    prob = feat['haw_prob']              # [B, T] f32, the detector
    nvalid = jnp.maximum(1.0, valid.sum())
    wmean = lambda x: (valid * x).sum() / nvalid

    # Event-rate budget on the detector. Two-sided: KL(Ber(rho) || Ber(mean q))
    # diverges as the rate goes to zero, so a collapsing detector is pushed up.
    rho = float(self.haw_target_rate)
    clip = float(self.haw_rate_clip)
    rate = (valid * prob).sum() / nvalid
    rate_c = jnp.clip(rate, clip, 1.0 - clip)
    rate_kl = (
        rho * (np.log(rho) - jnp.log(rate_c)) +
        (1.0 - rho) * (np.log1p(-rho) - jnp.log1p(-rate_c)))
    losses['event_rate'] = jnp.broadcast_to(rate_kl, prob.shape)

    # Detached fitting recurrence: same forward values as the live one, but the
    # event history is detached so this trains only b/alpha/beta/g_eta. It
    # cannot be replaced by sg(Mbar) -- alpha and beta reach the KL only
    # through this recurrence. Post-WM ordering: score with Mbar_t, then
    # consume y_t; there is no prepend.
    base, alpha, beta = self._haw_params()
    decay = jnp.exp(-beta)
    ctx = feat['haw_ctx']  # reused, not recomputed and not detached

    def fit_step(state, xs):
      y_t, k = xs
      state = jnp.where(k, state, 0.0)
      mbar = decay * state
      return mbar + alpha * y_t, mbar

    _, mbar_fit = jax.lax.scan(
        fit_step, init_state,
        (sg(feat['haw_event']).swapaxes(0, 1), (~reset).swapaxes(0, 1)))
    mbar_fit = mbar_fit.swapaxes(0, 1)
    lam_fit = jax.nn.softplus(base + mbar_fit + ctx)
    p_fit = -jnp.expm1(-lam_fit)

    q = jnp.clip(sg(prob), 1e-6, 1.0 - 1e-6)
    p = jnp.clip(p_fit, 1e-6, 1.0 - 1e-6)
    losses['haw'] = valid * (
        q * (jnp.log(q) - jnp.log(p)) +
        (1.0 - q) * (jnp.log1p(-q) - jnp.log1p(-p)))

    metrics['event_rate_obs'] = rate
    metrics['event_rate_error'] = rate - rho
    metrics['event_hard_rate_obs'] = wmean(feat['haw_event'])
    metrics['event_prob_entropy_obs'] = wmean(
        -(q * jnp.log(q) + (1.0 - q) * jnp.log1p(-q)))
    metrics['event_prob_std_time_obs'] = prob.std(1).mean()
    metrics['event_delta_mag_obs'] = wmean(feat['haw_delta_mag'])
    metrics['haw_prior_rate'] = wmean(feat['haw_prior_prob'])
    metrics['haw_lam_mean'] = wmean(feat['haw_lam'])
    metrics['haw_lam_max'] = feat['haw_lam'].max()
    metrics['haw_mbar_std'] = mbar_fit.std()
    metrics['haw_ctx_std'] = ctx.std()
    metrics['haw_base'] = base
    metrics['haw_alpha'] = alpha
    metrics['haw_beta'] = beta
    metrics['haw_valid_frac'] = valid.mean()
    # Live guard on the fitting invariant; nonzero means a misalignment in the
    # reset mask, the event indexing, or the initial carry.
    metrics['haw_lam_fit_err'] = (
        valid * jnp.abs(lam_fit - feat['haw_lam'])).max()

    if not training:
      metrics.update(self._probes(
          init_deter, prior, feat, reset, mbar_fit, valid, nvalid))
    return carry, entries, losses, feat, metrics

  def _probes(self, init_deter, prior, feat, reset, mbar_fit, valid, nvalid):
    """Report-only diagnostics. Never part of any loss.

    `gap_deploy` isolates the teacher/deployment input gap: same detached event
    history, prior canonical delta instead of the posterior one. `gap_memory`
    ablates the *variation* in Hawkes memory while holding its level, so it
    does not confound with a base-rate shift the way zeroing Mbar would.
    """
    wmean = lambda x: (valid * x).sum() / nvalid
    base, _, _ = self._haw_params()
    keep = ~reset

    # r^-_t from the prior already computed for the dyn KL; r^-_{-1} needs one
    # extra prior evaluation on the incoming deterministic state.
    r_cur = self._event_repr(prior)
    r_init = self._event_repr(self._prior(init_deter))
    r_prev = jnp.concatenate([r_init[:, None], r_cur[:, :-1]], 1)
    d_neg = jnp.where(keep[..., None, None], r_cur - r_prev, 0.0)
    ctx_probe = self._context(feat['deter'], d_neg, self._mag(d_neg))
    p_probe = -jnp.expm1(-jax.nn.softplus(base + mbar_fit + ctx_probe))

    mu_m = (valid * mbar_fit).sum() / nvalid
    p_mean_m = -jnp.expm1(
        -jax.nn.softplus(base + mu_m + feat['haw_ctx']))

    q, p_plus = feat['haw_prob'], feat['haw_prior_prob']
    gap_memory = wmean(jnp.abs(p_plus - p_mean_m))
    return {
        'haw_gap_teacher': wmean(jnp.abs(q - p_plus)),
        'haw_gap_deploy': wmean(jnp.abs(q - p_probe)),
        'haw_gap_memory': gap_memory,
        'haw_gap_memory_rel': gap_memory / (wmean(p_plus) + 1e-8),
        'haw_probe_rate': wmean(p_probe),
        'haw_delta_mag_prior': wmean(self._mag(d_neg).mean(-1)),
    }

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
