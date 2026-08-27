"""Invariant tests for the binary-event Hawkes RSSM.

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

import embodied.jax.nets as nn

nn.COMPUTE_DTYPE = jnp.float32

from dreamerv3.hawkes_rssm import HawkesRSSM  # noqa: E402

B, T, OBS, ACT = 3, 6, 12, 4
DETER, STOCH, CLASSES, UNIMIX, RHO = 32, 4, 4, 0.01, 0.1
f32 = jnp.float32


def make_dyn(**kw):
  base = dict(
      obs_dim=OBS, deter=DETER, hidden=16, stoch=STOCH, classes=CLASSES,
      blocks=4, act='silu', norm='rms', unimix=UNIMIX, haw_hidden=16,
      haw_embed=8, haw_context_hidden=8, haw_target_rate=RHO)
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


def run_loss(dyn, resets=(0,), training=True):
  tokens, acts, reset = make_batch(resets=resets)

  def fn(tokens, acts, reset):
    carry = dyn.initial(B)
    return dyn.loss(carry, tokens, acts, reset, training=training)

  _, (_, _, losses, feat, mets) = init_and_run(fn, tokens, acts, reset)
  return losses, feat, mets


def wake_ctx(params, seed=0):
  """Give g_eta's output layer real weights.

  `ctxout` is zero-initialized by design, so at init the Hawkes context reads
  none of its inputs and p_t is exactly rho. Tests about what the context
  depends on have to wake it first or they compare rho against rho.
  """
  params = dict(params)
  key, = [k for k in params if 'ctxout' in k and k.endswith('kernel')]
  params[key] = jnp.asarray(
      np.random.RandomState(seed).normal(0, 0.5, params[key].shape),
      params[key].dtype)
  return params


def make_entries(steps=T):
  return {
      'deter': jnp.ones((B, steps, DETER), f32),
      'stoch': jnp.ones((B, steps, STOCH, CLASSES), f32),
      'haw_state': jnp.full((B, steps), 0.7, f32),
      'haw_lam': jnp.full((B, steps), 0.3, f32)}


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
    noise = jnp.asarray(
        np.random.RandomState(3).normal(0, 1, logit.shape), f32)
    assert not np.allclose(self._repr(logit), self._repr(logit + noise))

  def test_is_finite_at_saturated_logits(self):
    logit = jnp.full((B, STOCH, CLASSES), -1e4, f32).at[..., 0].set(1e4)
    out = self._repr(logit)
    assert np.isfinite(out).all()
    assert out.min() >= np.log(UNIMIX / CLASSES) - 1e-5


# ---------------------------------------------------------------------------
# Shapes, initialization, and the two masks
# ---------------------------------------------------------------------------

class TestShapesAndMasking:

  def test_scalar_features_are_bt(self):
    _, _, feat = observe(make_dyn(), *make_batch())
    for key in ('haw_prob', 'haw_event', 'haw_prior_prob', 'haw_lam',
                'haw_ctx', 'haw_delta_mag', 'haw_valid'):
      assert feat[key].shape == (B, T), (key, feat[key].shape)
      assert feat[key].dtype == f32, (key, feat[key].dtype)

  def test_hard_events_are_binary(self):
    _, _, feat = observe(make_dyn(), *make_batch())
    ev = np.asarray(feat['haw_event'])
    assert np.isin(ev, (0.0, 1.0)).all(), np.unique(ev)

  def test_detector_starts_near_rho_with_real_variation(self):
    """outscale=0.1 with binit=logit(rho): centred on the budget, but frames
    must already score differently or nothing breaks the symmetry."""
    _, _, feat = observe(make_dyn(), *make_batch(resets=()))
    q = np.asarray(feat['haw_prob'])[:, 1:]  # step 0 has no previous token
    assert abs(q.mean() - RHO) < 0.03, q.mean()
    assert q.std() > 1e-4, q.std()
    assert (q > 0).all() and (q < 1).all()

  def test_hawkes_prior_starts_exactly_at_rho(self):
    """b is set so 1 - exp(-softplus(b)) == rho, and ctxout starts at zero."""
    _, _, feat = observe(make_dyn(), *make_batch(resets=()))
    assert np.allclose(np.asarray(feat['haw_ctx']), 0.0, atol=1e-6)
    assert np.allclose(
        np.asarray(feat['haw_prior_prob'])[:, 0], RHO, atol=1e-4)

  def test_reset_and_replay_boundaries_produce_no_event(self):
    _, _, feat = observe(make_dyn(), *make_batch(resets=(0, 3)))
    valid = np.asarray(feat['haw_valid'])
    assert np.allclose(valid[:, [0, 3]], 0.0)      # reset, and no previous obs
    assert np.allclose(valid[:, [1, 2, 4, 5]], 1.0)
    for key in ('haw_prob', 'haw_event'):
      assert np.allclose(np.asarray(feat[key])[:, [0, 3]], 0.0), key

  def test_invalid_step_leaks_no_latent_delta(self):
    _, _, feat = observe(make_dyn(), *make_batch(resets=(0, 3)))
    mag = np.asarray(feat['haw_delta_mag'])
    # log1p(sqrt(1e-8)) is the floor `_delta` returns for a zeroed delta.
    assert (mag[:, [0, 3]] < 1e-3).all(), mag[:, [0, 3]]
    assert (mag[:, [1, 2, 4, 5]] > 1e-2).all(), mag

  def test_restored_hawkes_state_survives_an_invalid_detector_step(self):
    """An unavailable encoder delta invalidates the detector, not the memory."""
    dyn = make_dyn()
    out = dyn.truncate(make_entries(2), dyn.initial(B))
    assert not np.asarray(out['haw_prev_valid']).any()
    assert np.allclose(np.asarray(out['haw_prev_obs']), 0.0)
    assert np.allclose(np.asarray(out['haw_prev_repr']), 0.0)
    assert np.allclose(np.asarray(out['haw_state']), 0.7)
    assert np.allclose(np.asarray(out['haw_lam']), 0.3)

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
    for key in ('haw_prob', 'haw_prior_prob', 'haw_lam', 'haw_ctx'):
      assert feat[key].dtype == jnp.float32, (key, feat[key].dtype)
    assert carry['haw_prev_repr'].dtype == jnp.float32
    assert carry['haw_state'].dtype == jnp.float32


# ---------------------------------------------------------------------------
# Replay carry contract
# ---------------------------------------------------------------------------

class TestCarryContract:

  def test_entry_space_excludes_the_transient_fields(self):
    keys = set(make_dyn().entry_space.keys())
    assert keys == {'deter', 'stoch', 'haw_state', 'haw_lam'}, keys

  def test_truncate_preserves_the_carry_tree(self):
    dyn = make_dyn()
    carry = dyn.initial(B)
    out = dyn.truncate(make_entries(2), carry)
    assert set(out.keys()) == set(carry.keys())
    for key in carry:
      assert out[key].shape == carry[key].shape, key
      assert out[key].dtype == carry[key].dtype, key

  def test_starts_rebuilds_the_representation_from_the_prior(self):
    dyn = make_dyn()

    def fn(entries):
      carry = dyn.initial(B)
      out = dyn.starts(entries, carry, 2)
      want = dyn._event_repr(dyn._prior(nn.cast(out['deter'])))
      return out, want

    _, (out, want) = init_and_run(fn, make_entries())
    assert set(out.keys()) == {
        'deter', 'stoch', 'haw_state', 'haw_lam',
        'haw_prev_repr', 'haw_prev_valid'}
    assert out['deter'].shape == (B * 2, DETER)
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
# Routing: what the current observation may and may not touch
# ---------------------------------------------------------------------------

class TestRouting:

  def _run(self, tokens):
    dyn = make_dyn()
    _, acts, reset = make_batch(resets=(0,))

    def fn(tokens, acts, reset):
      carry = dyn.initial(B)
      return dyn.observe(
          carry, tokens, acts, reset, training=False, sample_event=False)

    params = wake_ctx(nj.init(fn)({}, tokens, acts, reset, seed=0))
    return nj.pure(fn)(params, tokens, acts, reset, seed=0)[1][2]

  def test_current_observation_does_not_change_current_state(self):
    """h_t is built before x_t is read.

    This also settles the deployment probe: p^probe_t is a function of
    (Mbar_t, h_t, h_{t-1}) alone, so deter being untouched through step t
    means p^probe_t is untouched too.
    """
    tokens, _, _ = make_batch()
    a = self._run(tokens)
    b = self._run(tokens.at[:, 2].add(5.0))
    assert np.allclose(
        np.asarray(a['deter'])[:, :3], np.asarray(b['deter'])[:, :3],
        atol=1e-5)

  def test_current_observation_changes_the_detector(self):
    tokens, _, _ = make_batch()
    a = np.asarray(self._run(tokens)['haw_prob'])[:, 2]
    b = np.asarray(self._run(tokens.at[:, 2].add(5.0))['haw_prob'])[:, 2]
    assert not np.allclose(a, b, atol=1e-6), (a, b)

  def test_current_observation_may_move_the_teacher_forced_hawkes(self):
    """Deliberate: g_eta reads the posterior delta while observing."""
    tokens, _, _ = make_batch()
    a = np.asarray(self._run(tokens)['haw_prior_prob'])[:, 2]
    b = np.asarray(
        self._run(tokens.at[:, 2].add(5.0))['haw_prior_prob'])[:, 2]
    assert not np.allclose(a, b, atol=1e-8), (a, b)

  def test_imagined_hawkes_ignores_the_sampled_latent(self):
    """No z_t in the Hawkes context: at the first imagined step everything
    upstream is deterministic, so only the categorical draw differs."""
    dyn = make_dyn()
    imgacts = {'action': jnp.zeros((B * 2, 2, ACT), f32)}

    def fn(entries, imgacts):
      carry = dyn.initial(B)
      starts = dyn.starts(entries, carry, 2)
      _, feat, _ = dyn.imagine(starts, imgacts, 2, training=True)
      return feat

    entries = make_entries()
    params = wake_ctx(nj.init(fn)({}, entries, imgacts, seed=0))
    a = nj.pure(fn)(params, entries, imgacts, seed=0)[1]
    b = nj.pure(fn)(params, entries, imgacts, seed=7)[1]
    assert not np.allclose(
        np.asarray(a['stoch'])[:, 0], np.asarray(b['stoch'])[:, 0])
    for key in ('haw_prob', 'haw_lam', 'haw_ctx'):
      assert np.allclose(
          np.asarray(a[key])[:, 0], np.asarray(b[key])[:, 0], atol=1e-6), key


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


_is_det = lambda k: '/det' in k
_is_haw = lambda k: any(
    s in k for s in ('haw_base', 'haw_alpha_raw', 'haw_beta_raw', '/ctx'))
_is_rssm = lambda k: any(
    s in k for s in ('/dynin', '/dynhid', '/dyngru', '/obs', '/prior', '/hemb'))


class TestGradientRouting:

  def test_hawkes_kl_trains_the_hawkes_parameters(self):
    n = _grad_norms(lambda los, feat: los['haw'].sum(), _is_haw)
    assert n > 0.0, n

  def test_hawkes_kl_trains_alpha_and_beta_through_the_recurrence(self):
    """The detached fitting scan is what carries this; sg(Mbar) would not."""
    for name in ('haw_alpha_raw', 'haw_beta_raw'):
      n = _grad_norms(lambda los, feat: los['haw'].sum(), lambda k: name in k)
      assert n > 0.0, (name, n)

  def test_hawkes_kl_does_not_reach_the_detector(self):
    n = _grad_norms(lambda los, feat: los['haw'].sum(), _is_det)
    assert n == 0.0, n

  def test_hawkes_kl_does_not_reshape_the_rssm(self):
    n = _grad_norms(lambda los, feat: los['haw'].sum(), _is_rssm)
    assert n == 0.0, n

  def test_rate_loss_trains_the_detector(self):
    n = _grad_norms(lambda los, feat: los['event_rate'].mean(), _is_det)
    assert n > 0.0, n

  def test_rate_loss_does_not_train_the_hawkes_prior(self):
    """The budget belongs to q_t; the Hawkes prior only chases it."""
    n = _grad_norms(lambda los, feat: los['event_rate'].mean(), _is_haw)
    assert n == 0.0, n

  def test_head_event_trains_the_detector(self):
    """The route the reward and continuation heads use."""
    n = _grad_norms(lambda los, feat: feat['haw_head_event'].sum(), _is_det)
    assert n > 0.0, n

  def test_delayed_world_model_path_trains_the_detector(self):
    """y_t -> M_t -> omega -> h_{t+1}, excluding the head route."""
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

  def test_hawkes_context_is_detached_from_the_rssm(self):
    n = _grad_norms(
        lambda los, feat: feat['haw_ctx'].sum(), _is_rssm)
    assert n == 0.0, n


# ---------------------------------------------------------------------------
# Live vs. detached Hawkes recurrence
# ---------------------------------------------------------------------------

class TestFittingInvariant:

  def test_fit_matches_the_live_recurrence(self):
    for resets in ((0,), (0, 3), (0, 1, 5)):
      _, _, mets = run_loss(make_dyn(), resets=resets)
      assert float(mets['haw_lam_fit_err']) < 1e-4, (resets, mets)

  def test_invariant_holds_from_a_nonzero_incoming_carry(self):
    dyn = make_dyn()
    tokens, acts, reset = make_batch(resets=(0, 3))

    def fn(tokens, acts, reset):
      carry = dict(dyn.initial(B))
      carry['haw_state'] = jnp.full((B,), 0.42, f32)
      carry['haw_lam'] = jnp.full((B,), 0.11, f32)
      return dyn.loss(carry, tokens, acts, reset, training=True)

    _, (_, _, _, _, mets) = init_and_run(fn, tokens, acts, reset)
    assert float(mets['haw_lam_fit_err']) < 1e-4, mets['haw_lam_fit_err']


# ---------------------------------------------------------------------------
# Omega
# ---------------------------------------------------------------------------

class TestOmega:

  def _omega(self):
    dyn = make_dyn()

    def fn(state, lam):
      return dyn._hawkes_embed(state, lam).astype(f32)

    args = (jnp.full((B,), 0.2, f32), jnp.full((B,), 0.05, f32))
    params = nj.init(fn)({}, *args, seed=0)
    return lambda *a: nj.pure(fn)(params, *a, seed=0)[1]

  def _rel(self, a, b):
    return float(jnp.abs(a - b).mean() / (jnp.abs(a).mean() + 1e-8))

  def test_output_moves_with_the_hawkes_state(self):
    """An RMS norm on the first (2-D) projection would flatten this."""
    omega, lam = self._omega(), jnp.full((B,), 0.05, f32)
    a = omega(jnp.zeros((B,), f32), lam)
    b = omega(jnp.full((B,), 0.1, f32), lam)  # one alpha-sized event jump
    assert self._rel(a, b) > 1e-3, self._rel(a, b)

  def test_output_moves_with_the_intensity(self):
    omega, state = self._omega(), jnp.full((B,), 0.2, f32)
    a = omega(state, jnp.zeros((B,), f32))
    b = omega(state, jnp.full((B,), 0.05, f32))
    assert self._rel(a, b) > 1e-4, self._rel(a, b)

  def test_gradient_is_nonzero_for_both_inputs(self):
    omega = self._omega()
    scalar = lambda s, l: omega(s, l).sum()
    gs, gl = jax.grad(scalar, argnums=(0, 1))(
        jnp.full((B,), 0.2, f32), jnp.full((B,), 0.05, f32))
    assert np.isfinite(np.asarray(gs)).all()
    assert float(jnp.abs(gs).sum()) > 1e-4, float(jnp.abs(gs).sum())
    assert float(jnp.abs(gl).sum()) > 1e-8, float(jnp.abs(gl).sum())


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
    assert set(feat.keys()) == set(imgfeat.keys()), set(feat) ^ set(imgfeat)
    for key in feat:
      assert feat[key].dtype == imgfeat[key].dtype, key
      assert feat[key].shape[2:] == imgfeat[key].shape[2:], key

  def test_imagined_carry_drops_the_observation_only_field(self):
    _, imgcarry, _ = self._observe_then_imagine(make_dyn())
    assert set(imgcarry.keys()) == {
        'deter', 'stoch', 'haw_state', 'haw_lam',
        'haw_prev_repr', 'haw_prev_valid'}, sorted(imgcarry)

  def test_imagined_events_come_from_the_hawkes_prior(self):
    _, _, imgfeat = self._observe_then_imagine(make_dyn())
    prob = np.asarray(imgfeat['haw_prob'])
    assert np.allclose(prob, np.asarray(imgfeat['haw_prior_prob']))
    assert np.allclose(prob, np.asarray(imgfeat['haw_head_event']))
    ev = np.asarray(imgfeat['haw_event'])
    assert np.isin(ev, (0.0, 1.0)).all(), np.unique(ev)

  def test_observed_head_event_is_the_realized_detector_event(self):
    feat, _, _ = self._observe_then_imagine(make_dyn())
    assert np.allclose(
        np.asarray(feat['haw_head_event']), np.asarray(feat['haw_event']))

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
# Losses and probes
# ---------------------------------------------------------------------------

_PROBES = (
    'haw_gap_teacher', 'haw_gap_deploy',
    'haw_gap_memory', 'haw_gap_memory_rel')


class TestLosses:

  def test_loss_keys_and_shapes(self):
    losses, _, mets = run_loss(make_dyn())
    assert set(losses.keys()) == {'dyn', 'rep', 'haw', 'event_rate'}, sorted(
        losses)
    for key, value in losses.items():
      assert value.shape == (B, T), (key, value.shape)
    assert np.isfinite(
        np.asarray([float(v) for v in mets.values()], np.float32)).all()

  def test_hawkes_kl_is_zero_on_invalid_steps(self):
    losses, feat, _ = run_loss(make_dyn(), resets=(0, 3))
    invalid = np.asarray(feat['haw_valid']) == 0
    assert np.allclose(np.asarray(losses['haw'])[invalid], 0.0)

  def test_rate_is_measured_on_the_detector(self):
    _, feat, mets = run_loss(make_dyn(), resets=(0, 3))
    valid = np.asarray(feat['haw_valid'])
    q = np.asarray(feat['haw_prob'])
    want = (valid * q).sum() / max(valid.sum(), 1.0)
    assert np.allclose(float(mets['event_rate_obs']), want, atol=1e-6)

  def test_rate_loss_is_two_sided(self):
    """A collapsed detector must be pushed up, not further down."""
    rho, clip = 0.1, 1e-3
    kl = lambda r: (
        rho * (np.log(rho) - np.log(r)) +
        (1 - rho) * (np.log1p(-rho) - np.log1p(-r)))
    assert kl(0.02) > 0 and kl(0.30) > 0
    assert abs(kl(rho)) < 1e-9
    assert np.isfinite(kl(clip)) and kl(clip) < 1e3

  def test_probes_are_report_only(self):
    _, _, train_mets = run_loss(make_dyn(), training=True)
    _, _, report_mets = run_loss(make_dyn(), training=False)
    assert not any(k.startswith('haw_gap') for k in train_mets), sorted(
        train_mets)
    assert set(_PROBES) <= set(report_mets), sorted(report_mets)

  def test_probes_are_finite_and_nonnegative(self):
    _, _, mets = run_loss(make_dyn(), resets=(0, 3), training=False)
    for key in _PROBES + ('haw_probe_rate', 'haw_delta_mag_prior'):
      value = float(mets[key])
      assert np.isfinite(value), (key, value)
      assert value >= 0.0, (key, value)

  def test_memory_probe_is_zero_when_the_memory_never_varies(self):
    """alpha = 0 leaves Mbar constant at zero, so ablating its variation
    changes nothing. Any nonzero reading later means memory is in use."""
    _, _, mets = run_loss(
        make_dyn(haw_init_alpha=1e-8), resets=(0,), training=False)
    assert float(mets['haw_gap_memory']) < 1e-6, mets['haw_gap_memory']
