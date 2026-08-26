"""Invariant tests for the latent-conditioned Hawkes RSSM.

Run with:  python -m pytest dreamerv3/tests/test_hawkes_rssm.py -v

COMPUTE_DTYPE is forced to float32 here so the numeric assertions mean
something; production runs use bfloat16 for activations only.
"""

import elements
import embodied.jax
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

import embodied.jax.nets as nn

nn.COMPUTE_DTYPE = jnp.float32

from dreamerv3.hawkes_rssm import HawkesRSSM  # noqa: E402

B, T, OBS, ACT = 3, 6, 12, 4
DETER, STOCH, CLASSES, UNIMIX = 32, 4, 4, 0.01
TYPES = 4
f32 = jnp.float32


def make_dyn(**kw):
  base = dict(
      deter=DETER, hidden=16, stoch=STOCH, classes=CLASSES, blocks=4,
      act='silu', norm='rms', unimix=UNIMIX, haw_hidden=16, haw_embed=8,
      haw_context_hidden=8, haw_target_rate=0.1, haw_types=TYPES)
  base.update(kw)
  act_space = {'action': elements.Space(np.float32, (ACT,), -1, 1)}
  return HawkesRSSM(act_space, **base, name='dyn')


def make_batch(seed=0, resets=(0,)):
  rng = np.random.RandomState(seed)
  tokens = jnp.asarray(rng.normal(0, 1, (B, T, OBS)), f32)
  acts = {'action': jnp.asarray(rng.uniform(-1, 1, (B, T, ACT)), f32)}
  reset = np.zeros((B, T), bool)
  for t in resets:
    reset[:, t] = True
  return tokens, acts, jnp.asarray(reset)


def init_and_run(fn, *args, seed=0):
  """nj.init + nj.pure in one shot. `fn` must be a zero-module closure."""
  params = nj.init(fn)({}, *args, seed=seed)
  return nj.pure(fn)(params, *args, seed=seed)


def observe(dyn, tokens, acts, reset, training=True, **kw):
  def fn(tokens, acts, reset):
    carry = dyn.initial(B)
    return dyn.observe(carry, tokens, acts, reset, training=training, **kw)
  return init_and_run(fn, tokens, acts, reset)[1]


# ---------------------------------------------------------------------------
# Canonical latent representation
# ---------------------------------------------------------------------------

class TestEventRepr:

  def _repr(self, logit, **kw):
    dyn = make_dyn(**kw)
    return np.asarray(init_and_run(lambda l: dyn._event_repr(l), logit)[1])

  def test_matches_dreamer_unimixed_categorical(self):
    logit = jnp.asarray(
        np.random.RandomState(0).normal(0, 3, (B, STOCH, CLASSES)), f32)
    want = embodied.jax.outs.Categorical(logit, UNIMIX).logits
    assert np.allclose(self._repr(logit), np.asarray(want), atol=1e-6)

  def test_is_invariant_to_a_constant_logit_shift(self):
    """softmax(l) == softmax(l + c): a shift must not read as a change."""
    logit = jnp.asarray(
        np.random.RandomState(1).normal(0, 3, (B, STOCH, CLASSES)), f32)
    shift = jnp.asarray(
        np.random.RandomState(2).normal(0, 5, (B, STOCH, 1)), f32)
    assert np.allclose(self._repr(logit), self._repr(logit + shift), atol=1e-5)
    # Sanity: a non-constant perturbation does move it.
    noise = jnp.asarray(
        np.random.RandomState(3).normal(0, 1, logit.shape), f32)
    assert not np.allclose(self._repr(logit), self._repr(logit + noise))

  def test_is_finite_at_saturated_logits(self):
    logit = jnp.full((B, STOCH, CLASSES), -1e4, f32).at[..., 0].set(1e4)
    out = self._repr(logit)
    assert np.isfinite(out).all()
    assert out.min() >= np.log(UNIMIX / CLASSES) - 1e-5


# ---------------------------------------------------------------------------
# Shapes, values, and the two masks
# ---------------------------------------------------------------------------

class TestShapesAndMasking:

  def test_scalar_features_are_bt(self):
    _, _, feat = observe(make_dyn(), *make_batch())
    for key in ('haw_prob', 'haw_event', 'haw_lam', 'haw_ctx',
                'haw_delta_mag', 'haw_valid'):
      assert feat[key].shape == (B, T), (key, feat[key].shape)
      assert feat[key].dtype == f32, (key, feat[key].dtype)

  def test_hard_events_are_binary(self):
    _, _, feat = observe(make_dyn(), *make_batch(), training=False,
                         sample_event=True)
    ev = np.asarray(feat['haw_event'])
    assert np.isin(ev, (0.0, 1.0)).all(), np.unique(ev)

  def test_probabilities_are_in_the_open_interval(self):
    _, _, feat = observe(make_dyn(), *make_batch(resets=()))
    prob = np.asarray(feat['haw_prob'])
    assert (prob > 0).all() and (prob < 1).all()
    assert np.isfinite(np.asarray(feat['haw_lam'])).all()

  def test_reset_masks_both_probability_and_event(self):
    """An episode start is not a transition, so it cannot be an event."""
    _, _, feat = observe(make_dyn(), *make_batch(resets=(0, 3)))
    for key in ('haw_prob', 'haw_event', 'haw_lam'):
      col = np.asarray(feat[key])
      assert np.allclose(col[:, [0, 3]], 0.0), key
    assert np.allclose(np.asarray(feat['haw_valid'])[:, [0, 3]], 0.0)
    assert np.allclose(np.asarray(feat['haw_valid'])[:, [1, 2, 4, 5]], 1.0)

  def test_invalid_delta_does_not_invalidate_the_event(self):
    """Step 0 of a fresh carry has no previous r, but still predicts."""
    _, _, feat = observe(make_dyn(), *make_batch(resets=()))
    mag = np.asarray(feat['haw_delta_mag'])
    # log1p(sqrt(1e-8)) is the floor `_delta` returns for a zeroed delta.
    assert (mag[:, 0] < 1e-3).all(), mag[:, 0]
    assert (mag[:, 1:] > 1e-2).all(), mag
    assert (np.asarray(feat['haw_prob'])[:, 0] > 0).all()
    assert np.allclose(np.asarray(feat['haw_valid'])[:, 0], 1.0)

  def test_eval_mode_thresholds_deterministically(self):
    dyn = make_dyn(haw_eval_threshold=0.5)
    _, _, feat = observe(
        dyn, *make_batch(), training=False, sample_event=False)
    prob = np.asarray(feat['haw_prob'])
    valid = np.asarray(feat['haw_valid'])
    assert np.allclose(
        np.asarray(feat['haw_event']), (prob >= 0.5).astype(np.float32) * valid)

  def test_event_quantities_stay_float32_under_bfloat16(self):
    old = nn.COMPUTE_DTYPE
    nn.COMPUTE_DTYPE = jnp.bfloat16
    try:
      carry, _, feat = observe(make_dyn(), *make_batch())
    finally:
      nn.COMPUTE_DTYPE = old
    for key in ('haw_prob', 'haw_lam', 'haw_ctx', 'haw_delta_mag'):
      assert feat[key].dtype == jnp.float32, (key, feat[key].dtype)
    assert carry['haw_prev_repr'].dtype == jnp.float32
    assert carry['haw_state'].dtype == jnp.float32


# ---------------------------------------------------------------------------
# Replay carry contract
# ---------------------------------------------------------------------------

def make_entries(steps=T):
  return {
      'deter': jnp.ones((B, steps, DETER), f32),
      'stoch': jnp.ones((B, steps, STOCH, CLASSES), f32),
      'haw_state': jnp.full((B, steps), 0.7, f32),
      'haw_lam': jnp.full((B, steps), 0.3, f32),
      'haw_type': jnp.zeros((B, steps, TYPES), f32).at[..., 1].set(1.0)}


class TestCarryContract:

  def test_entry_space_excludes_the_previous_representation(self):
    keys = set(make_dyn().entry_space.keys())
    assert keys == {
        'deter', 'stoch', 'haw_state', 'haw_lam', 'haw_type'}, keys

  def test_truncate_zeroes_the_delta_and_keeps_the_hawkes_state(self):
    dyn = make_dyn()
    carry = dyn.initial(B)
    out = dyn.truncate(make_entries(2), carry)
    assert set(out.keys()) == set(carry.keys())
    assert not np.asarray(out['haw_prev_valid']).any()
    assert np.allclose(np.asarray(out['haw_prev_repr']), 0.0)
    assert np.allclose(np.asarray(out['haw_state']), 0.7)
    assert np.allclose(np.asarray(out['haw_lam']), 0.3)
    assert np.allclose(np.asarray(out['haw_type'])[:, 1], 1.0)
    for key in carry:
      assert out[key].shape == carry[key].shape, key
      assert out[key].dtype == carry[key].dtype, key

  def test_starts_rebuilds_the_representation_from_the_prior(self):
    dyn = make_dyn()
    entries = make_entries()

    def fn(entries):
      carry = dyn.initial(B)
      out = dyn.starts(entries, carry, 2)
      want = dyn._event_repr(dyn._prior(nn.cast(out['deter'])))
      return out, want

    _, (out, want) = init_and_run(fn, entries)
    assert set(out.keys()) == {
        'deter', 'stoch', 'haw_state', 'haw_lam', 'haw_type',
        'haw_prev_repr', 'haw_prev_valid'}
    assert out['deter'].shape == (B * 2, DETER)
    assert out['haw_lam'].shape == (B * 2,)
    assert np.asarray(out['haw_prev_valid']).all()
    assert np.allclose(np.asarray(out['haw_prev_repr']), np.asarray(want))

  def test_haw_lam_survives_entry_to_starts_to_imagine(self):
    """lam_{t-1} feeds omega, so a change in it must move the first h."""
    dyn = make_dyn()
    imgacts = {'action': jnp.zeros((B * 2, 3, ACT), f32)}

    def fn(entries, imgacts):
      carry = dyn.initial(B)
      starts = dyn.starts(entries, carry, 2)
      _, feat, _ = dyn.imagine(starts, imgacts, 3, training=False)
      return starts['haw_lam'], feat['deter'][:, 0]

    entries = make_entries()
    bumped = dict(entries, haw_lam=entries['haw_lam'] + 2.0)
    params = nj.init(fn)({}, entries, imgacts, seed=0)
    _, (lam_a, det_a) = nj.pure(fn)(params, entries, imgacts, seed=0)
    _, (lam_b, det_b) = nj.pure(fn)(params, bumped, imgacts, seed=0)
    assert np.allclose(np.asarray(lam_a), 0.3)
    assert np.allclose(np.asarray(lam_b), 2.3)
    assert not np.allclose(np.asarray(det_a), np.asarray(det_b), atol=1e-6)


# ---------------------------------------------------------------------------
# Causality: y_t reaches h_{t+1}, never h_t
# ---------------------------------------------------------------------------

class TestCausality:

  def _run(self, tokens):
    """Observe with g_eta woken up.

    `ctxout` is zero-initialized by design, so at init pi_t is exactly rho and
    reads none of its inputs. Give it weights, or these tests compare rho
    against rho.
    """
    dyn = make_dyn()
    _, acts, reset = make_batch(resets=(0,))

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      return dyn.observe(
          carry, tokens, acts, reset, training=False, sample_event=False)

    params = dict(nj.init(fn)({}, tokens, acts, reset, seed=0))
    key, = [k for k in params if 'ctxout' in k and k.endswith('kernel')]
    params[key] = jnp.asarray(
        np.random.RandomState(0).normal(0, 0.5, params[key].shape),
        params[key].dtype)
    return nj.pure(fn)(params, tokens, acts, reset, seed=0)[1][2]

  def test_current_observation_does_not_change_current_state(self):
    tokens, _, _ = make_batch()
    a = self._run(tokens)
    b = self._run(tokens.at[:, 2].add(5.0))
    # h_t is built before the posterior is read, so everything up to and
    # including the perturbed step's deter is untouched.
    assert np.allclose(
        np.asarray(a['deter'])[:, :3], np.asarray(b['deter'])[:, :3],
        atol=1e-5)

  def test_current_observation_changes_the_current_event_probability(self):
    """The event model reads the posterior, so x_t must move pi_t."""
    tokens, _, _ = make_batch()
    a = np.asarray(self._run(tokens)['haw_prob'])[:, 2]
    b = np.asarray(self._run(tokens.at[:, 2].add(5.0))['haw_prob'])[:, 2]
    assert not np.allclose(a, b, atol=1e-6), (a, b)

  def test_imagination_logits_come_from_the_prior(self):
    dyn = make_dyn()
    imgacts = {'action': jnp.zeros((B * 2, 3, ACT), f32)}

    def fn(entries, imgacts):
      carry = dyn.initial(B)
      starts = dyn.starts(entries, carry, 2)
      _, feat, _ = dyn.imagine(starts, imgacts, 3, training=False)
      return feat['logit'], dyn._prior(feat['deter'])

    _, (logit, prior) = init_and_run(fn, make_entries(), imgacts)
    logit, prior = np.asarray(logit), np.asarray(prior)
    # No bit-exact match available: f32 matmuls default to TF32 on GPU, and the
    # scanned per-step call tiles differently from this re-batched one. Require
    # the gap to be negligible against the logit scale instead.
    gap = np.abs(logit - prior).max()
    assert gap < 0.05 * logit.std(), (gap, logit.std())


# ---------------------------------------------------------------------------
# Gradient routing
# ---------------------------------------------------------------------------

def _grad_norms(lossfn, keyfilter, resets=(0,), seed=0):
  """Gradient norm over params whose path matches `keyfilter`."""
  dyn = make_dyn()
  tokens, acts, reset = make_batch(resets=resets)

  def full(tokens, acts, reset):
    carry = dyn.initial(B)
    return dyn.loss(carry, tokens, acts, reset, training=True)

  params = nj.init(full)({}, tokens, acts, reset, seed=seed)

  def gradfn(tokens, acts, reset):
    def inner(tokens, acts, reset):
      carry = dyn.initial(B)
      _, _, losses, feat, _ = dyn.loss(
          carry, tokens, acts, reset, training=True)
      return lossfn(losses, feat)
    return nj.grad(inner, [dyn])(tokens, acts, reset)

  _, (_, _, grads) = nj.pure(gradfn)(params, tokens, acts, reset, seed=seed)
  sel = [v for k, v in grads.items() if keyfilter(k)]
  assert sel, (sorted(grads), 'no params matched')
  return float(jnp.sqrt(sum((v.astype(f32) ** 2).sum() for v in sel)))


_is_evt = lambda k: any(
    s in k for s in ('haw_base', 'haw_alpha_raw', 'haw_beta_raw', '/ctx'))
_is_rssm = lambda k: any(
    s in k for s in ('/dynin', '/dynhid', '/dyngru', '/obs', '/prior'))
_is_type = lambda k: '/typeout' in k
# Per-channel weights, so the reduction is not blind to the type the way a
# plain sum over a straight-through one-hot is.
_weighted = lambda x: (f32(x) * (1.0 + jnp.arange(x.shape[-1], dtype=f32))).sum()


class TestGradientRouting:

  def test_rate_loss_trains_the_event_model(self):
    n = _grad_norms(lambda los, feat: los['event_rate'].mean(), _is_evt)
    assert n > 0.0, n

  def test_rate_loss_does_not_reshape_the_rssm(self):
    """g_eta's latent inputs are detached, so the budget cannot be met by
    moving the world model."""
    n = _grad_norms(lambda los, feat: los['event_rate'].mean(), _is_rssm)
    assert n == 0.0, n

  def test_typed_event_trains_the_event_model(self):
    """The observed route the reward and continuation heads use. The
    binary half of it: sum_k of y c is exactly y.
    """
    n = _grad_norms(lambda los, feat: _weighted(feat['haw_type']), _is_evt)
    assert n > 0.0, n

  def test_delayed_world_model_path_trains_the_event_model(self):
    """y_t -> M_t -> omega -> h_{t+1}, excluding the head route."""
    n = _grad_norms(
        lambda los, feat: feat['deter'][:, 2:].astype(f32).sum(), _is_evt)
    assert n > 0.0, n

  def test_delayed_path_is_not_vanishing(self):
    """Guard against a dead omega: compare against the direct head route."""
    direct = _grad_norms(
        lambda los, feat: _weighted(feat['haw_type']), _is_evt)
    delayed = _grad_norms(
        lambda los, feat: feat['deter'][:, 2:].astype(f32).sum(), _is_evt)
    assert delayed > 1e-4 * direct, (delayed, direct)

  def test_cluster_losses_train_the_type_readout(self):
    for key in ('event_conf', 'event_use'):
      n = _grad_norms(lambda los, feat: los[key].mean(), _is_type)
      assert n > 0.0, (key, n)

  def test_cluster_losses_touch_nothing_but_the_type_readout(self):
    """Both u_t and the event weights are detached in these losses, so they
    cannot reshape the detector to manufacture easy clusters. Asserted over
    every parameter outside `typeout`, not a hand-listed subset.
    """
    for key in ('event_conf', 'event_use'):
      n = _grad_norms(
          lambda los, feat: los[key].mean(), lambda k: not _is_type(k))
      assert n == 0.0, (key, n)

  def test_typed_event_trains_the_type_readout(self):
    """Downstream utility, unlike the cluster losses, does reach typeout.

    Reduced with per-channel weights, as any real consumer does: `_headfeat`
    concatenates y c into the head input and `_hawkes_embed` into omega, so
    each type meets a different column of the next kernel.
    """
    n = _grad_norms(lambda los, feat: _weighted(feat['haw_type']), _is_type)
    assert n > 0.0, n

  def test_summing_the_typed_event_over_types_is_gradient_free(self):
    """sum_k of a straight-through one-hot is identically 1 -- sg(hard) plus
    probs minus sg(probs). A consumer that only sees the total mass gets no
    type gradient, which is why the test above weights the channels.
    """
    n = _grad_norms(lambda los, feat: feat['haw_type'].sum(), _is_type)
    assert n == 0.0, n

  def test_delta_is_detached(self):
    """dr_t is stop-gradded, so nothing flows back into the logit heads
    through the event model alone."""
    n = _grad_norms(
        lambda los, feat: feat['haw_prob'].sum(), lambda k: 'logit' in k)
    assert n == 0.0, n


# ---------------------------------------------------------------------------
# Imagination
# ---------------------------------------------------------------------------

class TestImagination:

  def _observe_then_imagine(self, dyn, length=4, nlast=2):
    tokens, acts, reset = make_batch()
    imgacts = {'action': jnp.zeros((B * nlast, length, ACT), f32)}

    def fn(tokens, acts, reset, imgacts):
      carry = dyn.initial(B)
      carry, entries, feat = dyn.observe(
          carry, tokens, acts, reset, training=True)
      starts = dyn.starts(entries, carry, nlast)
      imgcarry, imgfeat, _ = dyn.imagine(
          starts, imgacts, length, training=True)
      return feat, imgcarry, imgfeat

    return init_and_run(fn, tokens, acts, reset, imgacts)[1]

  def test_observed_and_imagined_feature_trees_match(self):
    feat, _, imgfeat = self._observe_then_imagine(make_dyn())
    assert set(feat.keys()) == set(imgfeat.keys()), (
        set(feat) ^ set(imgfeat))
    for key in feat:
      assert feat[key].dtype == imgfeat[key].dtype, key
      assert feat[key].shape[2:] == imgfeat[key].shape[2:], key

  def test_imagined_carry_has_every_field(self):
    _, imgcarry, _ = self._observe_then_imagine(make_dyn())
    assert set(imgcarry.keys()) == {
        'deter', 'stoch', 'haw_state', 'haw_lam', 'haw_type',
        'haw_prev_repr', 'haw_prev_valid'}, sorted(imgcarry)

  def test_imagined_events_are_binary(self):
    _, _, imgfeat = self._observe_then_imagine(make_dyn())
    ev = np.asarray(imgfeat['haw_event'])
    assert np.isin(ev, (0.0, 1.0)).all(), np.unique(ev)
    assert np.allclose(np.asarray(imgfeat['haw_valid']), 1.0)

  def test_imagination_accepts_a_full_observe_carry(self):
    """report() hands imagine() the observation carry, not starts()."""
    dyn = make_dyn()
    tokens, acts, reset = make_batch()
    imgacts = {'action': jnp.zeros((B, 3, ACT), f32)}

    def fn(tokens, acts, reset, imgacts):
      carry = dyn.initial(B)
      carry, _, _ = dyn.observe(carry, tokens, acts, reset, training=False)
      return dyn.imagine(carry, imgacts, 3, training=False)[1]

    _, imgfeat = init_and_run(fn, tokens, acts, reset, imgacts)
    assert np.isfinite(np.asarray(imgfeat['haw_prob'])).all()

  def test_long_imagination_stays_finite(self):
    _, _, imgfeat = self._observe_then_imagine(make_dyn(), length=64, nlast=1)
    for key in ('deter', 'haw_prob', 'haw_lam', 'haw_delta_mag'):
      assert np.isfinite(np.asarray(imgfeat[key], np.float32)).all(), key


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def run_loss(dyn, resets=(0,)):
  tokens, acts, reset = make_batch(resets=resets)

  def fn(tokens, acts, reset):
    carry = dyn.initial(B)
    return dyn.loss(carry, tokens, acts, reset, training=True)

  _, (_, _, losses, feat, mets) = init_and_run(fn, tokens, acts, reset)
  return losses, feat, mets


class TestLosses:

  def test_loss_keys_and_shapes(self):
    losses, _, mets = run_loss(make_dyn())
    assert set(losses.keys()) == {
        'dyn', 'rep', 'event_rate', 'event_conf', 'event_use'}, sorted(losses)
    for key, value in losses.items():
      assert value.shape == (B, T), (key, value.shape)
    assert np.isfinite(
        np.asarray([float(v) for v in mets.values()], np.float32)).all()

  def test_no_hawkes_kl_is_returned(self):
    losses, _, _ = run_loss(make_dyn())
    assert 'haw' not in losses

  def test_replay_boundary_steps_still_count_in_the_rate(self):
    """haw_prev_valid must gate the delta only, never the loss mask."""
    _, feat, mets = run_loss(make_dyn(), resets=())
    assert np.allclose(np.asarray(feat['haw_valid']), 1.0)
    assert float(mets['haw_valid_frac']) == 1.0

  def test_reset_steps_are_excluded_from_the_rate(self):
    _, feat, mets = run_loss(make_dyn(), resets=(0, 3))
    valid = np.asarray(feat['haw_valid'])
    prob = np.asarray(feat['haw_prob'])
    want = (valid * prob).sum() / max(valid.sum(), 1.0)
    assert np.allclose(float(mets['event_rate_obs']), want, atol=1e-6)
    assert np.isclose(float(mets['haw_valid_frac']), valid.mean())

  def test_all_reset_batch_keeps_the_losses_finite(self):
    """No event mass anywhere: the usage term must fall back to uniform
    rather than divide zero by zero."""
    losses, feat, mets = run_loss(make_dyn(), resets=tuple(range(T)))
    assert np.allclose(np.asarray(feat['haw_valid']), 0.0)
    for key, value in losses.items():
      assert np.isfinite(np.asarray(value, np.float32)).all(), key
    assert np.allclose(np.asarray(losses['event_use']), 0.0)
    assert np.allclose(np.asarray(losses['event_conf']), 0.0)
    assert np.isfinite(
        np.asarray([float(v) for v in mets.values()], np.float32)).all()
    assert np.isclose(
        float(mets['event_type_effective_count']), TYPES, atol=1e-4)

  def test_type_temperature_must_be_positive(self):
    with pytest.raises(AssertionError):
      make_dyn(haw_type_temp=0.0)

  def test_rate_loss_is_two_sided(self):
    """A collapsed event model must be pushed up, not further down."""
    rho, clip = 0.1, 1e-3
    kl = lambda r: (
        rho * (np.log(rho) - np.log(r)) +
        (1 - rho) * (np.log1p(-rho) - np.log1p(-r)))
    assert kl(0.02) > 0 and kl(0.30) > 0
    assert abs(kl(rho)) < 1e-9
    assert np.isfinite(kl(clip)) and kl(clip) < 1e3

  def test_initial_rate_sits_at_the_target(self):
    """b is initialized so 1 - exp(-softplus(b)) == rho and g_eta starts at
    zero output, so the first step (M = 0) lands exactly on rho. Later steps
    only rise, since the excitation is positive."""
    _, feat, mets = run_loss(make_dyn(haw_target_rate=0.1), resets=())
    assert np.allclose(np.asarray(feat['haw_ctx']), 0.0, atol=1e-6)
    prob = np.asarray(feat['haw_prob'])
    assert np.allclose(prob[:, 0], 0.1, atol=1e-4), prob[:, 0]
    assert float(mets['event_rate_obs']) >= 0.1 - 1e-6


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

def onehot(k, batch=B):
  return jnp.zeros((batch, TYPES), f32).at[:, k].set(1.0)


class TestOmega:
  """omega must stay sensitive to M and lam once the type channels join it."""

  def _omega(self):
    dyn = make_dyn()

    def fn(state, lam, typ):
      return dyn._hawkes_embed(state, lam, typ).astype(f32)

    args = (jnp.full((B,), 0.2, f32), jnp.full((B,), 0.05, f32), onehot(1))
    params = nj.init(fn)({}, *args, seed=0)
    return lambda *a: nj.pure(fn)(params, *a, seed=0)[1]

  def _rel(self, a, b):
    return float(jnp.abs(a - b).mean() / (jnp.abs(a).mean() + 1e-8))

  def test_output_moves_with_the_hawkes_state(self):
    """Held at a fixed nonzero type, so this is the type-domination guard:
    the one-hot arrives at 1.0 while M sits two orders of magnitude below it,
    and haw_type_input_scale is what keeps M readable."""
    omega, lam, typ = self._omega(), jnp.full((B,), 0.05, f32), onehot(1)
    a = omega(jnp.zeros((B,), f32), lam, typ)
    b = omega(jnp.full((B,), 0.1, f32), lam, typ)  # one alpha-sized jump
    assert self._rel(a, b) > 1e-3, self._rel(a, b)

  def test_output_moves_with_the_intensity(self):
    omega, state, typ = self._omega(), jnp.full((B,), 0.2, f32), onehot(1)
    a = omega(state, jnp.zeros((B,), f32), typ)
    b = omega(state, jnp.full((B,), 0.05, f32), typ)
    assert self._rel(a, b) > 1e-4, self._rel(a, b)

  def test_output_moves_with_the_type(self):
    omega = self._omega()
    state, lam = jnp.full((B,), 0.2, f32), jnp.full((B,), 0.05, f32)
    a = omega(state, lam, onehot(0))
    b = omega(state, lam, onehot(2))
    assert self._rel(a, b) > 1e-3, self._rel(a, b)

  def test_gradient_is_nonzero_for_every_input(self):
    omega = self._omega()
    scalar = lambda s, l, c: omega(s, l, c).sum()
    grads = jax.grad(scalar, argnums=(0, 1, 2))(
        jnp.full((B,), 0.2, f32), jnp.full((B,), 0.05, f32), onehot(1))
    for name, g in zip(('state', 'lam', 'type'), grads):
      assert np.isfinite(np.asarray(g)).all(), name
      assert float(jnp.abs(g).sum()) > 1e-8, (name, float(jnp.abs(g).sum()))


class TestTypedEvent:

  def test_typed_event_is_zero_or_one_hot(self):
    _, _, feat = observe(make_dyn(), *make_batch(), training=False,
                         sample_event=True)
    typed = np.asarray(feat['haw_type'])
    event = np.asarray(feat['haw_event'])
    assert np.isin(typed, (0.0, 1.0)).all(), np.unique(typed)
    # Exactly one type active iff an event fired, and y is its sum.
    assert np.allclose(typed.sum(-1), event)

  def test_head_mix_is_a_distribution_and_one_hot_when_observing(self):
    _, _, feat = observe(make_dyn(), *make_batch(), training=False,
                         sample_event=True)
    w = np.asarray(feat['haw_head_mix'])
    assert w.shape == (B, T, TYPES + 1)
    assert np.allclose(w.sum(-1), 1.0, atol=1e-5)
    assert np.isin(w, (0.0, 1.0)).all(), np.unique(w)

  def test_head_mix_is_the_joint_when_imagining(self):
    dyn = make_dyn()
    imgacts = {'action': jnp.zeros((B * 2, 3, ACT), f32)}

    def fn(entries, imgacts):
      carry = dyn.initial(B)
      starts = dyn.starts(entries, carry, 2)
      _, feat, _ = dyn.imagine(starts, imgacts, 3, training=True)
      return feat

    _, feat = init_and_run(fn, make_entries(), imgacts)
    w = np.asarray(feat['haw_head_mix'])
    prob = np.asarray(feat['haw_prob'])
    a = np.asarray(feat['haw_type_prob'])
    assert np.allclose(w.sum(-1), 1.0, atol=1e-5)
    assert np.allclose(w[..., 0], 1.0 - prob, atol=1e-5)
    assert np.allclose(w[..., 1:], prob[..., None] * a, atol=1e-5)

  def test_type_probabilities_are_a_distribution(self):
    _, _, feat = observe(make_dyn(), *make_batch())
    for key in ('haw_type_prob', 'haw_type_prob_sg'):
      a = np.asarray(feat[key])
      assert a.shape == (B, T, TYPES), (key, a.shape)
      assert np.allclose(a.sum(-1), 1.0, atol=1e-5), key
      assert (a > 0).all(), key

  def test_type_head_is_not_zero_initialized(self):
    """At exactly uniform logits the assignment entropy sits at a
    zero-gradient saddle, so `typeout` must start with real weights."""
    _, _, feat = observe(make_dyn(), *make_batch())
    a = np.asarray(feat['haw_type_prob'])
    assert a.std() > 1e-6, a.std()

  def test_reset_masks_the_typed_event(self):
    _, _, feat = observe(make_dyn(), *make_batch(resets=(0, 3)))
    typed = np.asarray(feat['haw_type'])
    assert np.allclose(typed[:, [0, 3]], 0.0)

  def test_previous_type_cannot_cross_an_episode_reset(self):
    dyn = make_dyn()
    tokens, acts, _ = make_batch()

    def fn(typ, tokens, acts, reset):
      carry = dict(dyn.initial(B))
      carry['haw_type'] = typ
      carry['haw_state'] = jnp.full((B,), 0.5, f32)
      carry['haw_lam'] = jnp.full((B,), 0.5, f32)
      return dyn.observe(
          carry, tokens, acts, reset, training=False, sample_event=False)

    _, _, blocked = make_batch(resets=(0,))
    _, _, open_ = make_batch(resets=())
    params = nj.init(fn)({}, onehot(0), tokens, acts, blocked, seed=0)
    step0 = lambda typ, reset: np.asarray(
        nj.pure(fn)(params, typ, tokens, acts, reset, seed=0)[1][2]
        ['deter'])[:, 0]
    # With a reset the incoming type is masked out of omega ...
    assert np.allclose(
        step0(onehot(0), blocked), step0(onehot(2), blocked), atol=1e-6)
    # ... and without one it demonstrably reaches h_t, so the test has teeth.
    assert not np.allclose(
        step0(onehot(0), open_), step0(onehot(2), open_), atol=1e-6)
