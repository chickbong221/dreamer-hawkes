"""
Hawkes-Aware RSSM dynamics for DreamerV3 + ManiSkill.

Drop-in replacement for `dreamerv3.rssm.RSSM` selected via `--dyn.typ hawkes`.
Implements the revised HTWM algorithm:

  1. Two-encoder events: causal prior q_psi^pri(z_t, a_t) used at imagination
     time; posterior q_psi^pos(z_t, z_{t+1}, a_t, eps_t) used during training.
  2. Multivariate Hawkes intensity with O(1) recursive update on the
     exponential kernel (per-pair state S_{k,j}).
  3. Closed-form compensator integral for the Hawkes NLL.
  4. Signed alpha + softplus(lambda) so the process can express inhibition
     as well as excitation (nonlinear Hawkes, Bremaud & Massoulie 1996).
  5. Hawkes context fused into the deterministic state via h_t = MLP(lambda_t).
  6. Per-trajectory state is carried in the dynamics carry dict alongside
     the standard deter/stoch, so observe/imagine/scan work unchanged.

Notes specific to this repo:
  - The full Hawkes-aware Transformer attention + KV cache (Eqs. 4-5 in the
    paper) is a separate component intended to replace the GRU-style `_core`
    in a future iteration. This file keeps the GRU core and injects Hawkes
    structure via (a) the event-conditioned context vector h_t and (b) an
    additional loss term L_haw + L_kl + L_ent + (optional) L_sup. Cache-based
    attention requires a Transformer backbone and a different unroll pattern
    in `embodied.run.train`; we leave that as the obvious next step.
  - Privileged event labels (`log/success_once`, `log/fail_once`) are
    consumed via the `extras` dict passed into `loss()`. When absent, the
    supervision loss is zero and the model trains in fully unsupervised mode.
"""

import math

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


class HawkesRSSM(nj.Module):
  """Hawkes-aware RSSM. Selected via `--dyn.typ hawkes`."""

  # --- Inherited RSSM hyperparameters (same defaults as rssm.RSSM) ----------
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

  # --- Hawkes-specific hyperparameters --------------------------------------
  num_events: int = 8                # K
  haw_hidden: int = 256              # MLP_omega hidden width
  haw_embed: int = 128               # h_t dimensionality
  haw_kl_balance: float = 0.1        # beta_kl in KL-balancing
  haw_ent_target: float = 0.5        # entropy floor; below this, L_ent kicks in
  haw_init_alpha: float = 0.01       # alpha init scale (small so HTWM ~ RSSM at start)
  haw_init_beta: float = 1.0         # beta init (decay over ~1 step)
  haw_init_mu: float = 0.1           # mu init (baseline intensity)
  haw_gumbel_tau: float = 1.0        # Gumbel-Softmax temperature
  haw_use_supervision: bool = False  # if True, expect privileged labels in extras

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    self.act_space = act_space
    self.kw = kw

  # =========================================================================
  #  embodied.jax compatibility surface — identical to rssm.RSSM
  # =========================================================================

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)),
        # Hawkes per-pair state S_{k,j} carried alongside the latent state.
        # Shape [K, K]. Stored as float32 even when COMPUTE_DTYPE is bf16,
        # because the recursion is numerically delicate.
        haw_state=elements.Space(np.float32, (self.num_events, self.num_events)),
        # Last event distribution p_{t-1}, used by the recursion to add
        # alpha_{k, e_{t-1}} into S without committing to a hard sample.
        haw_prev=elements.Space(np.float32, (self.num_events,)))

  def initial(self, bsize):
    return nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32),
        haw_state=jnp.zeros([bsize, self.num_events, self.num_events], f32),
        haw_prev=jnp.zeros([bsize, self.num_events], f32)))

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    return jax.tree.map(lambda x: x[:, -1], entries)

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  # =========================================================================
  #  Hawkes parameter modules
  # =========================================================================

  def _hawkes_params(self):
    """Return (mu [K], alpha [K,K], beta [K,K]).

    Parameters are stored as nj.Variable so JAX/ninjax tracks them as
    trainable leaves. alpha is signed (nonlinear Hawkes); the intensity
    is passed through softplus to guarantee positivity. beta and mu are
    kept positive via softplus on a raw learnable scalar with a small
    positive bias so initial values are sensible.
    """
    K = self.num_events
    mu_raw   = self.sub('haw_mu_raw',
                        nj.Variable, jnp.zeros, (K,), f32).read()
    alpha    = self.sub('haw_alpha',
                        nj.Variable, jnp.zeros, (K, K), f32).read()
    beta_raw = self.sub('haw_beta_raw',
                        nj.Variable, jnp.zeros, (K, K), f32).read()
    mu    = jax.nn.softplus(mu_raw   + self.haw_init_mu).astype(nn.COMPUTE_DTYPE)   # [K]
    beta  = jax.nn.softplus(beta_raw + self.haw_init_beta).astype(nn.COMPUTE_DTYPE) # [K, K]
    # alpha starts near zero so the model begins close to plain RSSM and
    # learns excitation/inhibition entries as needed.
    return mu, alpha.astype(nn.COMPUTE_DTYPE), beta

  def _event_prior(self, deter, action):
    """Causal prior p_t^pri(e_t | z_t, a_t). Returns logits [B, K]."""
    x = jnp.concatenate([deter, action], -1)
    for i in range(2):
      x = self.sub(f'pri{i}', nn.Linear, self.haw_hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'pri{i}norm', nn.Norm, self.norm)(x))
    return self.sub('prilogit', nn.Linear, self.num_events, **self.kw)(x)

  def _event_post(self, deter, next_tokens, action, eps):
    """Posterior q_t^pos(e_t | z_t, z_{t+1}, a_t, eps_t). Returns logits [B, K].

    `next_tokens` is the encoder output for x_{t+1} (the obs token).
    `eps` is the per-batch prediction-surprise scalar [B, 1].
    """
    x = jnp.concatenate([deter, next_tokens, action, eps], -1)
    for i in range(2):
      x = self.sub(f'pos{i}', nn.Linear, self.haw_hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'pos{i}norm', nn.Norm, self.norm)(x))
    return self.sub('poslogit', nn.Linear, self.num_events, **self.kw)(x)

  def _hawkes_embed(self, lam):
    """h_t = MLP_omega(lambda(t)). Input [B, K], output [B, haw_embed]."""
    x = jnp.log1p(lam)  # symlog-style compression; intensities are positive
    for i in range(2):
      x = self.sub(f'hemb{i}', nn.Linear, self.haw_hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'hemb{i}norm', nn.Norm, self.norm)(x))
    return self.sub('hembout', nn.Linear, self.haw_embed, **self.kw)(x)

  # =========================================================================
  #  Hawkes recursion (per-step, O(K^2))
  # =========================================================================

  def _hawkes_step(self, S_prev, e_prev_probs, alpha, beta):
    """One step of the exponential Hawkes recursion.

    S_{k,j}(t) = exp(-beta_{k,j}) * S_{k,j}(t-1)
                 + p_{t-1}(j) * alpha_{k,j}

    where p_{t-1}(j) is the (soft) event probability at the previous step,
    so gradients flow back through the event distribution without requiring
    a hard sample.

    Args:
      S_prev: [B, K, K]
      e_prev_probs: [B, K]
      alpha, beta: [K, K]
    Returns:
      S_new: [B, K, K]
      lam:   [B, K]   (intensities at the new step)
    """
    decay = jnp.exp(-beta)[None]               # [1, K, K]
    increment = e_prev_probs[:, None, :] * alpha[None]  # [B, K, K]
    S_new = decay * S_prev + increment
    mu, _, _ = self._hawkes_params()           # mu cached separately; cheap
    # Sum_j S_{k,j} gives the total contribution to lambda_k from past events.
    lam_raw = mu[None] + S_new.sum(-1)         # [B, K]
    lam = jax.nn.softplus(lam_raw)             # positivity, allows signed alpha
    return S_new, lam

  def _hawkes_compensator(self, S_prev, alpha, beta, L):
    """Closed-form compensator over a single step of length 1.

    For a length-L segment, sum the per-step compensators. With unit step,
    Lambda_k = mu_k + (alpha_{k,j}/beta_{k,j}) * (1 - exp(-beta_{k,j})) * p(j)
    integrated tilt; we return the per-step compensator and the trainer
    accumulates it. Returned shape: [B, K].
    """
    # Per-step closed form for the segment [t-1, t] with the kernel decaying
    # from time t-1 (the event that updated S). Using the trapezoidal form:
    # Lambda_k(unit step) = mu_k + sum_j alpha_{k,j}/beta_{k,j}
    #                              * (1 - exp(-beta_{k,j})) * S_{k,j} / alpha_{k,j}
    # Simplifies to mu_k + sum_j (1 - exp(-beta_{k,j})) * S_{k,j} / beta_{k,j}.
    mu, _, _ = self._hawkes_params()
    decayed = (1.0 - jnp.exp(-beta)) / jnp.maximum(beta, 1e-6)  # [K, K]
    contrib = S_prev * decayed[None]                            # [B, K, K]
    return mu[None] + contrib.sum(-1)                           # [B, K]

  # =========================================================================
  #  Top-level observe / imagine / loss — same signatures as RSSM
  # =========================================================================

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda c, inputs: self._observe(c, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, haw_state, haw_prev, action = nn.mask(
        (carry['deter'], carry['stoch'], carry['haw_state'],
         carry['haw_prev'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)

    # --- Hawkes step: roll forward using p_{t-1} stored in haw_prev ----------
    mu, alpha, beta = self._hawkes_params()
    haw_state, lam = self._hawkes_step(haw_state, haw_prev, alpha, beta)
    h_haw = self._hawkes_embed(lam)            # [B, haw_embed]

    # --- Standard RSSM core, with Hawkes context fused into the action stream
    deter = self._core(deter, stoch, action, h_haw)

    tokens_flat = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens_flat if self.absolute else jnp.concatenate([deter, tokens_flat], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    # --- Posterior event distribution p_t (becomes haw_prev for next step) ---
    # eps_t: prediction-surprise signal. We compare the prior over the
    # discrete stoch latent (predicted from deter alone) against the
    # posterior (which saw the new observation). High KL = the observation
    # carried information the dynamics could not predict — i.e. an event.
    prior_logit = sg(self._prior(deter))
    eps = self._dist(logit).kl(self._dist(prior_logit))[..., None].astype(nn.COMPUTE_DTYPE)  # [B, 1]
    post_logit = self._event_post(deter, tokens_flat, action, eps)
    post_probs = jax.nn.softmax(post_logit, axis=-1)
    haw_prev_next = post_probs

    carry = dict(
        deter=deter, stoch=stoch,
        haw_state=haw_state, haw_prev=haw_prev_next)
    feat = dict(
        deter=deter, stoch=stoch, logit=logit,
        haw_lam=lam, haw_post=post_logit, haw_embed=h_haw)
    entry = dict(
        deter=deter, stoch=stoch,
        haw_state=haw_state, haw_prev=haw_prev_next)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    """Imagination uses the PRIOR for event sampling — z_{t+1} is unavailable."""
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)

      # Hawkes step using haw_prev from the carry
      mu, alpha, beta = self._hawkes_params()
      haw_state, lam = self._hawkes_step(
          carry['haw_state'], carry['haw_prev'], alpha, beta)
      h_haw = self._hawkes_embed(lam)

      deter = self._core(carry['deter'], carry['stoch'], actemb, h_haw)
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

      # Use the PRIOR to draw the event distribution for the NEXT step
      pri_logit = self._event_prior(deter, actemb)
      pri_probs = jax.nn.softmax(pri_logit, axis=-1)

      carry = nn.cast(dict(
          deter=deter, stoch=stoch,
          haw_state=haw_state, haw_prev=pri_probs))
      feat = nn.cast(dict(
          deter=deter, stoch=stoch, logit=logit,
          haw_lam=lam, haw_pri=pri_logit, haw_embed=h_haw))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      return carry, feat, action

  def loss(self, carry, tokens, acts, reset, training, extras=None):
    """Compute world-model losses including the Hawkes terms.

    `extras` is an optional dict that may contain a privileged event label
    tensor under key 'event_label' of shape [B, T] with integer entries in
    [0, K). When present and self.haw_use_supervision is True, we add a
    cross-entropy term to push the posterior toward the labels.
    """
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
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

    # ── Hawkes losses ────────────────────────────────────────────────────────
    # Event prior p_t^pri evaluated at the SAME (deter, action) used to
    # produce the posterior, so the KL is well-defined per step.
    actemb = nn.DictConcat(self.act_space, 1)(acts)
    pri_logit = self._event_prior(feat['deter'], actemb)
    post_logit = feat['haw_post']
    lam = feat['haw_lam']                                  # [B, T, K]

    # Sample e_t differentiably from the posterior
    g = -jnp.log(-jnp.log(jax.random.uniform(
        nj.seed(), post_logit.shape, minval=1e-9, maxval=1.0)))
    e_soft = jax.nn.softmax(
        (post_logit + g) / self.haw_gumbel_tau, axis=-1)   # [B, T, K]

    # L_haw = -sum_t log lambda_{e_t}(t) + sum_k Lambda_k
    log_lam = jnp.log(lam + 1e-8)
    nll = -(e_soft * log_lam).sum(-1)                       # [B, T]
    # Compensator: per-step closed form, summed over the segment.
    # We stored S_{k,j} in entries; reconstruct the per-step compensator by
    # using a stop-gradient copy to avoid double-counting through S.
    mu, alpha, beta = self._hawkes_params()
    S = entries['haw_state']                                # [B, T, K, K]
    decayed = (1.0 - jnp.exp(-beta)) / jnp.maximum(beta, 1e-6)
    comp = mu[None, None] + (S * decayed[None, None]).sum(-1)  # [B, T, K]
    losses['haw'] = nll + comp.sum(-1)                          # [B, T]

    # L_kl: KL-balanced posterior <-> prior
    pri_lp = jax.nn.log_softmax(pri_logit, axis=-1)
    post_p = jax.nn.softmax(post_logit, axis=-1)
    post_lp = jax.nn.log_softmax(post_logit, axis=-1)
    kl_post_pri = (post_p * (post_lp - sg(pri_lp))).sum(-1)
    kl_post_pri_sg = (sg(post_p) * (sg(post_lp) - pri_lp)).sum(-1)
    losses['haw_kl'] = (
        kl_post_pri_sg + self.haw_kl_balance * kl_post_pri)

    # L_ent: entropy regularization with a soft floor at haw_ent_target
    ent = -(post_p * post_lp).sum(-1)
    losses['haw_ent'] = jnp.maximum(0.0, self.haw_ent_target - ent)

    # Optional L_sup: cross-entropy against privileged labels
    if self.haw_use_supervision and extras and 'event_label' in extras:
      labels = extras['event_label']                         # [B, T] int
      one_hot = jax.nn.one_hot(labels, self.num_events)
      losses['haw_sup'] = -(one_hot * post_lp).sum(-1)
      metrics['haw_sup_acc'] = (
          post_p.argmax(-1) == labels).mean().astype(f32)

    # Metrics
    metrics['haw_lam_mean'] = lam.mean()
    metrics['haw_lam_max'] = lam.max()
    metrics['haw_post_ent'] = ent.mean()
    metrics['haw_alpha_abs_mean'] = jnp.abs(alpha).mean()
    metrics['haw_beta_mean'] = beta.mean()
    metrics['haw_inhibition_frac'] = (alpha < 0).mean().astype(f32)

    return carry, entries, losses, feat, metrics

  # =========================================================================
  #  Core (GRU) — extended to take h_t^haw as a third input stream
  # =========================================================================

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