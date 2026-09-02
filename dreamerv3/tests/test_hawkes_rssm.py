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
TYPES = 4
f32 = jnp.float32


def make_dyn(**kw):
  base = dict(
      obs_dim=OBS, deter=DETER, hidden=16, stoch=STOCH, classes=CLASSES,
      blocks=4, act='silu', norm='rms', unimix=UNIMIX, haw_hidden=16,
      haw_embed=8, haw_context_hidden=8, haw_target_rate=RHO, haw_types=TYPES)
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

  def test_detector_starts_near_the_configured_rho(self):
    """binit=logit(rho) has to survive the wider input: q must still be
    centred on the budget after sg(h_t) joined xi_t."""
    for rho in (0.03, 0.1, 0.3):
      dyn = make_dyn(haw_target_rate=rho)
      _, _, feat = observe(dyn, *make_batch(resets=()))
      q = np.asarray(feat['haw_prob'])[:, 1:]
      assert abs(q.mean() - rho) < max(0.02, 0.5 * rho), (rho, q.mean())
      assert q.std() > 1e-4, (rho, q.std())

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

  def test_explicit_nonsampling_mode_thresholds_deterministically(self):
    """Only the diagnostic path; normal evaluation samples."""
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
    n = _grad_norms(lambda los, feat: (1.0 - feat['haw_head_mix'][..., 0]).sum(), _is_det)
    assert n > 0.0, n

  def test_delayed_world_model_path_trains_the_detector(self):
    """y_t -> M_t -> omega -> h_{t+1}, excluding the head route."""
    n = _grad_norms(
        lambda los, feat: feat['deter'][:, 2:].astype(f32).sum(), _is_det)
    assert n > 0.0, n

  def test_delayed_path_is_not_vanishing(self):
    """Guard against a dead omega: compare against the direct head route."""
    direct = _grad_norms(
        lambda los, feat: (1.0 - feat['haw_head_mix'][..., 0]).sum(), _is_det)
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

  def test_stripped_repfeat_matches_the_imagined_tree(self):
    """The invariant that matters is at Agent.loss()'s concat: raw observed
    features carry diagnostic-only fields, `loss()` drops them, and what comes
    back must line up with imagination key for key."""
    dyn = make_dyn()
    tokens, acts, reset = make_batch()
    imgacts = {'action': jnp.zeros((B * 2, 3, ACT), f32)}

    def fn(tokens, acts, reset, imgacts):
      carry = dyn.initial(B)
      _, entries, _, repfeat, _ = dyn.loss(
          carry, tokens, acts, reset, training=True)
      livecarry, rawfeat = dyn.observe(
          dyn.initial(B), tokens, acts, reset, training=True)[0::2]
      starts = dyn.starts(entries, livecarry, 2)
      _, imgfeat, _ = dyn.imagine(starts, imgacts, 3, training=True)
      return repfeat, rawfeat, imgfeat

    _, (repfeat, rawfeat, imgfeat) = init_and_run(
        fn, tokens, acts, reset, imgacts)
    assert set(repfeat.keys()) == set(imgfeat.keys()), (
        set(repfeat) ^ set(imgfeat))
    for key in repfeat:
      assert repfeat[key].dtype == imgfeat[key].dtype, key
      assert repfeat[key].shape[2:] == imgfeat[key].shape[2:], key
    # The diagnostic fields exist upstream and only upstream.
    for key in ('haw_type_prob', 'haw_obs_delta_mag'):
      assert key in rawfeat, key
      assert key not in repfeat, key
      assert key not in imgfeat, key

  def test_imagined_carry_drops_the_observation_only_field(self):
    _, imgcarry, _ = self._observe_then_imagine(make_dyn())
    assert set(imgcarry.keys()) == {
        'deter', 'stoch', 'haw_state', 'haw_lam',
        'haw_prev_repr', 'haw_prev_valid'}, sorted(imgcarry)

  def test_imagined_events_come_from_the_hawkes_prior(self):
    _, _, imgfeat = self._observe_then_imagine(make_dyn())
    prob = np.asarray(imgfeat['haw_prob'])
    assert np.allclose(prob, np.asarray(imgfeat['haw_prior_prob']))
    mix = np.asarray(imgfeat['haw_head_mix'])
    assert np.allclose(mix[..., 0], 1.0 - prob, atol=1e-5)
    assert np.allclose(
        mix[..., 1:],
        prob[..., None] * np.asarray(imgfeat['haw_type_prior_prob']),
        atol=1e-5)
    ev = np.asarray(imgfeat['haw_event'])
    assert np.isin(ev, (0.0, 1.0)).all(), np.unique(ev)

  def test_observed_head_mix_is_one_hot_on_the_realized_outcome(self):
    feat, _, _ = self._observe_then_imagine(make_dyn())
    mix = np.asarray(feat['haw_head_mix'])
    assert mix.shape[-1] == TYPES + 1
    assert np.allclose(mix.sum(-1), 1.0, atol=1e-5)
    assert np.isin(mix, (0.0, 1.0)).all(), np.unique(mix)
    # Channel 0 is 1 - y, so the rest carry exactly the realized event.
    assert np.allclose(1.0 - mix[..., 0], np.asarray(feat['haw_event']))

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

  def test_imagination_ignores_the_incoming_representation(self):
    """report() hands over an observe carry holding the *posterior* r, so
    imagine() must rebuild the prior one or the first imagined delta is
    posterior-to-prior."""
    dyn = make_dyn()
    imgacts = {'action': jnp.zeros((B, 3, ACT), f32)}

    def fn(repr_in, imgacts):
      carry = dict(dyn.initial(B))
      carry['haw_prev_repr'] = repr_in
      return dyn.imagine(carry, imgacts, 3, training=False)[1]

    zeros = jnp.zeros((B, STOCH, CLASSES), f32)
    junk = jnp.full((B, STOCH, CLASSES), -3.0, f32)
    params = wake_ctx(nj.init(fn)({}, zeros, imgacts, seed=0))
    a = nj.pure(fn)(params, zeros, imgacts, seed=0)[1]
    b = nj.pure(fn)(params, junk, imgacts, seed=0)[1]
    for key in ('haw_prob', 'haw_ctx', 'haw_delta_mag'):
      assert np.allclose(
          np.asarray(a[key]), np.asarray(b[key]), atol=1e-6), key

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
    assert set(losses.keys()) == {
        'dyn', 'rep', 'haw', 'event_rate', 'event_conf', 'event_use',
        'event_type_prior'}, sorted(losses)
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

  def test_prob_std_time_ignores_invalid_steps(self):
    """Forced zeros at resets and replay boundaries are not variation."""
    _, feat, mets = run_loss(make_dyn(), resets=(0, 3))
    valid = np.asarray(feat['haw_valid'])
    q = np.asarray(feat['haw_prob'])
    n = np.maximum(valid.sum(1), 1.0)
    mu = (valid * q).sum(1) / n
    var = (valid * (q - mu[:, None]) ** 2).sum(1) / n
    want = float(np.sqrt(var + 1e-12).mean())
    assert np.allclose(float(mets['event_prob_std_time_obs']), want, atol=1e-6)
    # The unweighted version counts the forced zeros and reads much larger.
    assert float(mets['event_prob_std_time_obs']) < float(q.std(1).mean())

  def test_report_loss_samples_events(self):
    """Thresholded events would be all zero near rho, emptying the Hawkes
    memory and making the memory probe read a false collapse."""
    # High target rate so this cannot fail on an unlucky draw: at rho=0.1
    # over 15 valid steps, P(no event) would be ~20%.
    _, feat, _ = run_loss(
        make_dyn(haw_target_rate=0.8), resets=(), training=False)
    assert np.asarray(feat['haw_event']).sum() > 0.0


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

_is_type = lambda k: '/type0' in k or '/typeout' in k
_is_tprior = lambda k: '/tprior' in k
# Per-channel weights. sum_k of a straight-through one-hot is identically 1
# (sg(hard) - sg(prob) + prob), so a plain .sum() over the type axis is
# gradient-free and would make the routing tests below vacuous. Every real
# consumer meets each channel with a different weight: `_headfeat` concatenates
# y*c into the head input, so each type hits a different column of the next
# kernel.
_weighted = lambda x: (f32(x) * (1.0 + jnp.arange(x.shape[-1], dtype=f32))).sum()


class TestEventTypes:

  def test_type_probabilities_are_distributions(self):
    _, _, feat = observe(make_dyn(), *make_batch())
    for key in ('haw_type_prob', 'haw_type_prior_prob'):
      c = np.asarray(feat[key])
      assert c.shape == (B, T, TYPES), (key, c.shape)
      assert c.dtype == f32, key
      assert np.isfinite(c).all(), key
      assert np.allclose(c.sum(-1), 1.0, atol=1e-5), key
      assert (c > 0).all(), key

  def test_invalid_steps_get_a_uniform_posterior(self):
    _, _, feat = observe(make_dyn(), *make_batch(resets=(0, 3)))
    c = np.asarray(feat['haw_type_prob'])
    assert np.allclose(c[:, [0, 3]], 1.0 / TYPES, atol=1e-6)

  def test_type_readout_is_not_zero_initialized(self):
    """At exactly uniform logits the assignment entropy sits at a
    zero-gradient saddle, so `typeout` must start with real weights."""
    _, _, feat = observe(make_dyn(), *make_batch(resets=()))
    assert np.asarray(feat['haw_type_prob'])[:, 1:].std() > 1e-6

  def test_detector_input_layout(self):
    """The type posterior reads this verbatim, so the layout is a contract:
    [a_{t-1}, sg(h_t), symlog(sg(dx_t)), log1p(RMS(sg(dx_t)))]."""
    dyn = make_dyn()
    action = jnp.asarray(
        np.random.RandomState(0).normal(0, 1, (B, ACT)), f32)
    deter = jnp.asarray(
        np.random.RandomState(2).normal(0, 1, (B, DETER)), f32)
    delta = jnp.asarray(
        np.random.RandomState(1).normal(0, 1, (B, OBS)), f32)
    _, (inp, mag) = init_and_run(
        lambda a, h, d: dyn._detector_input(a, h, d), action, deter, delta)
    assert inp.shape == (B, ACT + DETER + OBS + 1), inp.shape
    assert np.allclose(np.asarray(inp)[:, :ACT], np.asarray(action))
    assert np.allclose(
        np.asarray(inp)[:, ACT:ACT + DETER], np.asarray(deter))
    assert np.allclose(
        np.asarray(inp)[:, ACT + DETER:-1],
        np.asarray(nn.symlog(delta)), atol=1e-6)
    assert np.allclose(np.asarray(inp)[:, -1], np.asarray(mag))

  def test_detector_reads_the_recurrent_history(self):
    """q_t must move when h_t moves: the whole point of feeding sg(h_t) is
    that "the gripper moved" and "the gripper closed on the tool" are the
    same pixels under a different task history."""
    dyn = make_dyn()
    action = jnp.zeros((B, ACT), f32)
    delta = jnp.asarray(
        np.random.RandomState(1).normal(0, 1, (B, OBS)), f32)
    rng = np.random.RandomState(3)
    h1 = jnp.asarray(rng.normal(0, 1, (B, DETER)), f32)
    h2 = jnp.asarray(rng.normal(0, 1, (B, DETER)), f32)

    def fn(a, h, d):
      inp, _ = dyn._detector_input(a, h, d)
      return dyn._detector(inp)

    params = nj.init(fn)({}, action, h1, delta, seed=0)
    q1 = nj.pure(fn)(params, action, h1, delta, seed=0)[1]
    q2 = nj.pure(fn)(params, action, h2, delta, seed=0)[1]
    assert np.abs(np.asarray(q1) - np.asarray(q2)).max() > 1e-4, (q1, q2)

  def test_detector_does_not_reshape_the_rssm(self):
    """sg(h_t) on the detector input: the rate budget must not be able to
    make its own target easier by moving the recurrence."""
    n = _grad_norms(lambda los, feat: los['event_rate'].mean(), _is_rssm)
    assert n == 0.0, n

  def test_detector_input_gradient_reaches_only_the_detector(self):
    n = _grad_norms(lambda los, feat: los['event_rate'].mean(), _is_det)
    assert n > 0.0, n

  def test_types_never_enter_the_carry_or_replay(self):
    dyn = make_dyn()
    carry, _, _ = observe(dyn, *make_batch())
    assert not any('type' in k for k in carry), sorted(carry)
    assert not any('type' in k for k in dyn.entry_space), sorted(
        dyn.entry_space)

  def test_cluster_losses_train_only_the_classifier(self):
    for key in ('event_conf', 'event_use'):
      assert _grad_norms(lambda los, feat: los[key].mean(), _is_type) > 0.0, key
      assert _grad_norms(
          lambda los, feat: los[key].mean(), lambda k: not _is_type(k)) == 0.0, key

  def test_type_prior_kl_trains_only_the_type_prior(self):
    n = _grad_norms(
        lambda los, feat: los['event_type_prior'].mean(), _is_tprior)
    assert n > 0.0, n
    assert _grad_norms(
        lambda los, feat: los['event_type_prior'].mean(),
        lambda k: not _is_tprior(k)) == 0.0

  def test_typed_head_channel_trains_the_classifier(self):
    """L_rew/con -> ct_st -> C_phi."""
    n = _grad_norms(
        lambda los, feat: _weighted(feat['haw_head_mix'][..., 1:]), _is_type)
    assert n > 0.0, n

  def test_typed_head_channel_does_not_reach_the_detector(self):
    """sg(y) on the typed channel is what blocks the second route."""
    n = _grad_norms(
        lambda los, feat: _weighted(feat['haw_head_mix'][..., 1:]), _is_det)
    assert n == 0.0, n

  def test_summing_the_typed_channel_over_types_is_gradient_free(self):
    """sum_k (sg(y) c_k) is exactly sg(y), so a consumer that only sees the
    total typed mass trains nothing. This is why the two tests above weight
    the channels, and why any future consumer must too."""
    n = _grad_norms(
        lambda los, feat: feat['haw_head_mix'][..., 1:].sum(), _is_type)
    assert n == 0.0, n

  def test_binary_head_channel_still_trains_the_detector(self):
    n = _grad_norms(
        lambda los, feat: (1.0 - feat['haw_head_mix'][..., 0]).sum(), _is_det)
    assert n > 0.0, n

  def test_zero_event_mass_keeps_the_type_losses_finite(self):
    losses, _, mets = run_loss(make_dyn(), resets=tuple(range(T)))
    for key in ('event_conf', 'event_use', 'event_type_prior'):
      assert np.isfinite(np.asarray(losses[key])).all(), key
    assert np.allclose(np.asarray(losses['event_use']), 0.0)
    assert np.isclose(
        float(mets['event_type_effective_count']), TYPES, atol=1e-4)
    assert np.isfinite(
        np.asarray([float(v) for v in mets.values()], np.float32)).all()

  def test_future_head_feature_inherits_the_head_routing(self):
    """`Agent._futurefeat` reads the same two channels out of
    `haw_head_mix` that `_headfeat` does, so the auxiliary future-reward
    head inherits the routing verbatim: y_t to the detector, sg(y_t) c^ST
    to the classifier, and nothing to the RSSM through the detached z_t."""
    from dreamerv3.agent import Agent
    consume = lambda los, feat: _weighted(Agent._futurefeat(None, feat))
    assert _grad_norms(consume, _is_det) > 0.0
    assert _grad_norms(consume, _is_type) > 0.0
    assert _grad_norms(consume, _is_rssm) == 0.0

  def test_future_head_typed_channel_cannot_reach_the_detector(self):
    from dreamerv3.agent import Agent
    typed = lambda los, feat: _weighted(
        Agent._futurefeat(None, feat)[..., -TYPES:])
    assert _grad_norms(typed, _is_type) > 0.0
    assert _grad_norms(typed, _is_det) == 0.0

  def test_per_type_metrics_are_present_and_finite(self):
    _, _, mets = run_loss(make_dyn(), resets=(0, 3))
    keys = ['event_type_entropy_sample', 'event_type_entropy_usage',
            'event_type_effective_count', 'event_type_max_occupancy',
            'event_type_min_occupancy', 'event_type_prior_kl',
            'event_type_prob_spread', 'event_type_prob_within_ratio']
    keys += [f'event_type_usage_obs/{k}' for k in range(TYPES)]
    keys += [f'event_type_prob_mean/{k}' for k in range(TYPES)]
    keys += [f'event_type_detector_mag_mean/{k}' for k in range(TYPES)]
    for key in keys:
      assert key in mets, key
      assert np.isfinite(float(mets[key])), (key, mets[key])
