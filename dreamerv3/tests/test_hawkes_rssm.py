"""Invariant tests for the binary-event Hawkes RSSM.

Run with:  python -m pytest dreamerv3/tests/test_hawkes_rssm.py -v

COMPUTE_DTYPE is forced to float32 here so the numeric assertions mean
something; production runs use bfloat16 for activations only.
"""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

import embodied.jax.nets as nn

nn.COMPUTE_DTYPE = jnp.float32

from dreamerv3.hawkes_rssm import HawkesRSSM  # noqa: E402

B, T, OBS, ACT = 3, 6, 12, 4
f32 = jnp.float32


def make_dyn(**kw):
  base = dict(
      obs_dim=OBS, deter=32, hidden=16, stoch=4, classes=4, blocks=4,
      act='silu', norm='rms', haw_hidden=16, haw_embed=8,
      haw_context_hidden=8, haw_target_rate=0.1)
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


# ---------------------------------------------------------------------------
# Shapes, values, reset and validity handling
# ---------------------------------------------------------------------------

class TestShapesAndMasking:

  def _feat(self, training=True, resets=(0,)):
    dyn = make_dyn()
    tokens, acts, reset = make_batch(resets=resets)

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      return dyn.observe(carry, tokens, acts, reset, training=training)

    _, (_, _, feat) = init_and_run(fn, tokens, acts, reset)
    return feat, np.asarray(reset)

  def test_scalar_features_are_bt(self):
    feat, _ = self._feat()
    for key in ('haw_prob', 'haw_event', 'haw_prior_prob', 'haw_lam',
                'haw_valid', 'haw_head_event'):
      assert feat[key].shape == (B, T), (key, feat[key].shape)
      assert feat[key].dtype == f32, (key, feat[key].dtype)

  def test_hard_events_are_binary(self):
    feat, _ = self._feat()
    ev = np.asarray(feat['haw_event'])
    assert np.all((ev == 0.0) | (ev == 1.0)), np.unique(ev)

  def test_probabilities_are_open_interval(self):
    feat, _ = self._feat()
    for key in ('haw_prob', 'haw_prior_prob'):
      v = np.asarray(feat[key])
      assert np.isfinite(v).all()
      assert (v >= 0).all() and (v <= 1).all(), (key, v.min(), v.max())
    prior = np.asarray(feat['haw_prior_prob'])
    assert (prior > 0).all() and (prior < 1).all()
    assert np.isfinite(np.asarray(feat['haw_lam'])).all()

  def test_first_episode_step_has_no_event(self):
    feat, reset = self._feat(resets=(0,))
    assert np.allclose(np.asarray(feat['haw_valid'])[reset], 0.0)
    assert np.allclose(np.asarray(feat['haw_prob'])[reset], 0.0)
    assert np.allclose(np.asarray(feat['haw_event'])[reset], 0.0)

  def test_mid_batch_reset_is_invalid_and_next_step_recovers(self):
    feat, _ = self._feat(resets=(0, 3))
    valid = np.asarray(feat['haw_valid'])
    assert np.allclose(valid[:, 0], 0.0)
    assert np.allclose(valid[:, 3], 0.0)
    assert np.allclose(valid[:, 4], 1.0)
    assert np.allclose(valid[:, 1], 1.0)

  def test_eval_mode_thresholds_deterministically(self):
    dyn = make_dyn(haw_eval_threshold=0.5)
    tokens, acts, reset = make_batch()

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      return dyn.observe(
          carry, tokens, acts, reset, training=False, sample_event=False)

    _, (_, _, feat) = init_and_run(fn, tokens, acts, reset)
    prob = np.asarray(feat['haw_prob'])
    ev = np.asarray(feat['haw_event'])
    valid = np.asarray(feat['haw_valid']).astype(bool)
    expect = (prob >= 0.5).astype(np.float32) * valid
    assert np.allclose(ev, expect)


# ---------------------------------------------------------------------------
# Replay carry contract
# ---------------------------------------------------------------------------

class TestCarryContract:

  def test_entry_space_excludes_previous_tokens(self):
    dyn = make_dyn()
    keys = set(dyn.entry_space.keys())
    assert keys == {'deter', 'stoch', 'haw_state', 'haw_prev'}, keys

  def test_truncate_marks_boundary_invalid(self):
    dyn = make_dyn()
    carry = jax.tree.map(lambda x: x, dyn.initial(B))
    entries = {
        'deter': jnp.ones((B, 2, 32), f32),
        'stoch': jnp.ones((B, 2, 4, 4), f32),
        'haw_state': jnp.full((B, 2), 0.7, f32),
        'haw_prev': jnp.ones((B, 2), f32)}
    out = dyn.truncate(entries, carry)
    assert set(out.keys()) == set(carry.keys())
    assert not np.asarray(out['haw_prev_valid']).any()
    assert np.allclose(np.asarray(out['haw_prev_obs']), 0.0)
    assert np.allclose(np.asarray(out['haw_state']), 0.7)
    assert np.allclose(np.asarray(out['haw_prev']), 1.0)
    for key in carry:
      assert out[key].shape == carry[key].shape, key
      assert out[key].dtype == carry[key].dtype, key

  def test_starts_drops_observation_only_fields(self):
    dyn = make_dyn()
    carry = dyn.initial(B)
    entries = {
        'deter': jnp.ones((B, T, 32), f32),
        'stoch': jnp.ones((B, T, 4, 4), f32),
        'haw_state': jnp.zeros((B, T), f32),
        'haw_prev': jnp.zeros((B, T), f32)}
    out = dyn.starts(entries, carry, 2)
    assert set(out.keys()) == {'deter', 'stoch', 'haw_state', 'haw_prev'}
    assert out['deter'].shape == (B * 2, 32)
    assert out['haw_state'].shape == (B * 2,)


# ---------------------------------------------------------------------------
# Causality: x_t must not reach h_t, only h_{t+1}
# ---------------------------------------------------------------------------

class TestCausality:

  def _run(self, tokens):
    dyn = make_dyn()
    _, acts, reset = make_batch(resets=(0,))

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      return dyn.observe(
          carry, tokens, acts, reset, training=False, sample_event=False)

    params = nj.init(fn)({}, tokens, acts, reset, seed=0)
    _, (_, _, feat) = nj.pure(fn)(params, tokens, acts, reset, seed=0)
    return feat

  def test_current_observation_does_not_change_current_state(self):
    tokens, _, _ = make_batch()
    perturbed = tokens.at[:, 2].add(5.0)
    a = self._run(tokens)
    b = self._run(perturbed)
    # Everything strictly before the perturbed step is untouched, and at the
    # perturbed step itself the deterministic state and the Hawkes quantities
    # must still be identical: x_t only reaches h_{t+1}.
    for key in ('deter', 'haw_lam', 'haw_prior_prob'):
      assert np.allclose(
          np.asarray(a[key])[:, :3], np.asarray(b[key])[:, :3],
          atol=1e-5), key

  def test_current_observation_can_change_current_probability(self):
    tokens, _, _ = make_batch()
    perturbed = tokens.at[:, 2].add(5.0)
    a = np.asarray(self._run(tokens)['haw_prob'])[:, 2]
    b = np.asarray(self._run(perturbed)['haw_prob'])[:, 2]
    assert not np.allclose(a, b, atol=1e-6), (a, b)

  def test_event_changes_next_deterministic_state(self):
    """d h_{t+1} / d pi_t must be nonzero through the Hawkes state."""
    dyn = make_dyn()
    tokens, acts, reset = make_batch(resets=(0,))

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      _, _, feat = dyn.observe(
          carry, tokens, acts, reset, training=True)
      return feat

    params = nj.init(fn)({}, tokens, acts, reset, seed=0)

    def gradfn(tokens, acts, reset):
      def inner(tokens, acts, reset):
        feat = fn(tokens, acts, reset)
        return feat['deter'][:, 2:].astype(f32).sum()
      return nj.grad(inner, [dyn])(tokens, acts, reset)

    _, (_, _, grads) = nj.pure(gradfn)(params, tokens, acts, reset, seed=0)
    det = [v for k, v in grads.items() if '/det' in k]
    assert det, sorted(grads)
    norm = float(jnp.sqrt(sum((v.astype(f32) ** 2).sum() for v in det)))
    assert norm > 0.0, norm


# ---------------------------------------------------------------------------
# Gradient routing
# ---------------------------------------------------------------------------

def _grad_norms(lossfn, keyfilter, resets=(0,), seed=0):
  """Return the gradient norm over params whose path matches `keyfilter`."""
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


_is_det = lambda k: '/det' in k
_is_haw = lambda k: any(
    s in k for s in ('haw_base', 'haw_alpha_raw', 'haw_beta_raw', '/ctx'))


class TestGradientRouting:

  def test_haw_loss_has_no_gradient_into_detector(self):
    n = _grad_norms(lambda los, feat: los['haw'].sum(), _is_det)
    assert n == 0.0, n

  def test_haw_loss_trains_hawkes_parameters(self):
    n = _grad_norms(lambda los, feat: los['haw'].sum(), _is_haw)
    assert n > 0.0, n

  def test_rate_loss_trains_detector(self):
    n = _grad_norms(lambda los, feat: los['event_rate'].mean(), _is_det)
    assert n > 0.0, n

  def test_head_event_trains_detector(self):
    """The route the reward and continuation heads use."""
    n = _grad_norms(lambda los, feat: feat['haw_head_event'].sum(), _is_det)
    assert n > 0.0, n

  def test_delayed_world_model_path_trains_detector(self):
    """y_t -> M_{t+1} -> lambda -> omega -> h_{t+1}, excluding the head route."""
    n = _grad_norms(
        lambda los, feat: feat['deter'][:, 2:].astype(f32).sum(), _is_det)
    assert n > 0.0, n

  def test_delayed_path_is_not_vanishing(self):
    """Guard against a dead omega: compare against the direct head route."""
    direct = _grad_norms(
        lambda los, feat: feat['haw_head_event'].sum(), _is_det)
    delayed = _grad_norms(
        lambda los, feat: feat['deter'][:, 2:].astype(f32).sum(), _is_det)
    assert delayed > 1e-4 * direct, (delayed, direct)


# ---------------------------------------------------------------------------
# Live vs. detached Hawkes recurrence
# ---------------------------------------------------------------------------

class TestFittingInvariant:

  @pytest.mark.parametrize('resets', [(0,), (0, 3), (0, 1, 4)])
  def test_lambda_fit_matches_lambda_model(self, resets):
    dyn = make_dyn()
    tokens, acts, reset = make_batch(resets=resets)

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      return dyn.loss(carry, tokens, acts, reset, training=True)

    params = nj.init(fn)({}, tokens, acts, reset, seed=0)
    _, (_, _, _, _, mets) = nj.pure(fn)(params, tokens, acts, reset, seed=0)
    assert float(mets['haw_lam_fit_err']) < 1e-4, float(
        mets['haw_lam_fit_err'])

  def test_invariant_holds_from_a_nonzero_incoming_carry(self):
    dyn = make_dyn()
    tokens, acts, reset = make_batch(resets=())

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      carry['haw_state'] = jnp.full((B,), 0.4, f32)
      carry['haw_prev'] = jnp.array([1.0, 0.0, 1.0], f32)
      carry['haw_prev_valid'] = jnp.ones((B,), bool)
      carry['deter'] = jnp.full((B, 32), 0.1, nn.COMPUTE_DTYPE)
      return dyn.loss(carry, tokens, acts, reset, training=True)

    params = nj.init(fn)({}, tokens, acts, reset, seed=0)
    _, (_, _, _, _, mets) = nj.pure(fn)(params, tokens, acts, reset, seed=0)
    assert float(mets['haw_lam_fit_err']) < 1e-4, float(
        mets['haw_lam_fit_err'])


# ---------------------------------------------------------------------------
# Hawkes embedding conditioning
# ---------------------------------------------------------------------------

class TestEmbedding:

  def _omega(self):
    dyn = make_dyn()

    def fn(state, lam):
      return dyn._hawkes_embed(state, lam).astype(f32)

    state = jnp.full((B,), 0.2, f32)
    lam = jnp.full((B,), 0.05, f32)
    params = nj.init(fn)({}, state, lam, seed=0)
    return lambda s, l: nj.pure(fn)(params, s, l, seed=0)[1]

  def test_omega_output_moves_with_the_hawkes_state(self):
    """An RMS norm on the first (2-D) projection would flatten this."""
    omega = self._omega()
    lam = jnp.full((B,), 0.05, f32)
    a = omega(jnp.zeros((B,), f32), lam)
    b = omega(jnp.full((B,), 0.1, f32), lam)  # one alpha-sized event jump
    rel = float(jnp.abs(a - b).mean() / (jnp.abs(a).mean() + 1e-8))
    assert rel > 1e-3, rel

  def test_omega_gradient_is_nonzero(self):
    omega = self._omega()
    scalar = lambda s, l: omega(s, l).sum()
    gs, gl = jax.grad(scalar, argnums=(0, 1))(
        jnp.full((B,), 0.2, f32), jnp.full((B,), 0.05, f32))
    assert np.isfinite(np.asarray(gs)).all()
    assert float(jnp.abs(gs).sum()) > 1e-4, float(jnp.abs(gs).sum())
    assert float(jnp.abs(gl).sum()) > 1e-8, float(jnp.abs(gl).sum())


# ---------------------------------------------------------------------------
# Imagination / feature-tree compatibility
# ---------------------------------------------------------------------------

class TestImagination:

  def test_observed_and_imagined_trees_match(self):
    dyn = make_dyn()
    tokens, acts, reset = make_batch()

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      carry, entries, feat = dyn.observe(
          carry, tokens, acts, reset, training=True)
      starts = dyn.starts(entries, carry, 2)
      imgacts = {'action': jnp.zeros((B * 2, 4, ACT), f32)}
      _, imgfeat, _ = dyn.imagine(starts, imgacts, 4, training=True)
      return feat, imgfeat

    params = nj.init(fn)({}, tokens, acts, reset, seed=0)
    _, (feat, imgfeat) = nj.pure(fn)(params, tokens, acts, reset, seed=0)
    assert set(feat.keys()) == set(imgfeat.keys()), (
        set(feat) ^ set(imgfeat))
    for key in feat:
      assert feat[key].dtype == imgfeat[key].dtype, key
      assert feat[key].shape[2:] == imgfeat[key].shape[2:], key

  def test_imagination_accepts_a_full_observe_carry(self):
    """report() hands imagine() the observation carry, not starts()."""
    dyn = make_dyn()
    tokens, acts, reset = make_batch()

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      carry, _, _ = dyn.observe(carry, tokens, acts, reset, training=False)
      imgacts = {'action': jnp.zeros((B, 3, ACT), f32)}
      _, imgfeat, _ = dyn.imagine(carry, imgacts, 3, training=False)
      return imgfeat

    params = nj.init(fn)({}, tokens, acts, reset, seed=0)
    _, imgfeat = nj.pure(fn)(params, tokens, acts, reset, seed=0)
    assert imgfeat['haw_lam'].shape == (B, 3)

  def test_long_imagination_stays_finite(self):
    dyn = make_dyn(haw_init_alpha=0.5, haw_init_beta=0.1)
    tokens, acts, reset = make_batch()

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      carry, entries, _ = dyn.observe(
          carry, tokens, acts, reset, training=True)
      starts = dyn.starts(entries, carry, 1)
      imgacts = {'action': jnp.zeros((B, 64, ACT), f32)}
      _, imgfeat, _ = dyn.imagine(starts, imgacts, 64, training=True)
      return imgfeat

    params = nj.init(fn)({}, tokens, acts, reset, seed=0)
    _, imgfeat = nj.pure(fn)(params, tokens, acts, reset, seed=0)
    for key in ('haw_lam', 'haw_prior_prob', 'deter'):
      assert np.isfinite(np.asarray(imgfeat[key], np.float32)).all(), key


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestLosses:

  def test_loss_keys_and_shapes(self):
    dyn = make_dyn()
    tokens, acts, reset = make_batch()

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      return dyn.loss(carry, tokens, acts, reset, training=True)

    params = nj.init(fn)({}, tokens, acts, reset, seed=0)
    _, (_, _, losses, _, mets) = nj.pure(fn)(
        params, tokens, acts, reset, seed=0)
    assert set(losses.keys()) == {'dyn', 'rep', 'haw', 'event_rate'}
    for key, value in losses.items():
      assert value.shape == (B, T), (key, value.shape)
      assert np.isfinite(np.asarray(value, np.float32)).all(), key
    assert np.isfinite(
        np.asarray([float(v) for v in mets.values()], np.float32)).all()

  def test_losses_are_zero_on_invalid_transitions(self):
    dyn = make_dyn()
    tokens, acts, reset = make_batch(resets=(0, 3))

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      return dyn.loss(carry, tokens, acts, reset, training=True)

    params = nj.init(fn)({}, tokens, acts, reset, seed=0)
    _, (_, _, losses, feat, _) = nj.pure(fn)(
        params, tokens, acts, reset, seed=0)
    invalid = np.asarray(feat['haw_valid']) == 0
    assert np.allclose(np.asarray(losses['haw'])[invalid], 0.0)

  def test_rate_loss_is_two_sided(self):
    """A collapsed detector must be pushed up, not further down."""
    rho, clip = 0.1, 1e-3
    kl = lambda r: (
        rho * (np.log(rho) - np.log(r)) +
        (1 - rho) * (np.log1p(-rho) - np.log1p(-r)))
    assert kl(0.02) > 0 and kl(0.30) > 0
    assert abs(kl(rho)) < 1e-9
    # Finite and not explosive at the clip floor.
    assert np.isfinite(kl(clip)) and kl(clip) < 1e3
