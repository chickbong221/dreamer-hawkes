"""Wiring tests for the auxiliary future-reward head and the head inputs.

Run with:  python -m pytest dreamerv3/tests/test_future_head.py -v

Two layers. `future_return_target` is a pure function and is tested against a
brute-force reference. Everything else needs a real Dreamer model, so one is
built directly -- `embodied.jax.Agent.__new__` is bypassed because it sets up
devices, meshes and shardings that a wiring test has no use for. size1m on a
5-D vector observation keeps the whole file to a few seconds of tracing.

COMPUTE_DTYPE is forced to float32 so the numeric assertions mean something.
"""

import pathlib

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import ruamel.yaml as yaml

import embodied.jax.nets as nn

nn.COMPUTE_DTYPE = jnp.float32

from dreamerv3 import agent as agentmod                       # noqa: E402
from dreamerv3.agent import Agent, future_return_target       # noqa: E402

B, T, OBS, ACT = 2, 20, 5, 3
f32 = jnp.float32

CONFIGS = yaml.YAML(typ='safe').load(
    (pathlib.Path(agentmod.__file__).parent / 'configs.yaml').read_text())


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def make_model(hawkes=True):
  """A tiny Dreamer model, without the device setup in Agent.__new__."""
  config = elements.Config(CONFIGS['defaults'])
  config = config.update(CONFIGS['size1m'])
  if hawkes:
    config = config.update(CONFIGS['hawkes'])
  # Same regex form the size presets use.
  config = config.update({
      r'.*\.units': 32,
      r'.*\.bins': 15,
      r'.*\.haw_hidden': 16,
      r'.*\.haw_context_hidden': 16,
      r'.*\.haw_embed': 8,
  })
  obs_space = {
      'vector': elements.Space(np.float32, (OBS,)),
      'is_first': elements.Space(bool),
      'is_last': elements.Space(bool),
      'is_terminal': elements.Space(bool),
      'reward': elements.Space(np.float32)}
  act_space = {'action': elements.Space(np.float32, (ACT,), -1, 1)}
  agent_config = elements.Config(
      **config.agent,
      logdir='', seed=0, jax=config.jax,
      batch_size=B, batch_length=T, replay_context=0,
      report_length=T, replica=0, replicas=1)
  model = object.__new__(Agent)
  Agent.__init__(model, obs_space, act_space, agent_config)
  return model


def make_data(seed=0, resets=(0,)):
  rng = np.random.RandomState(seed)
  first = np.zeros((B, T), bool)
  for t in resets:
    first[:, t] = True
  obs = {
      'vector': jnp.asarray(rng.normal(0, 1, (B, T, OBS)), f32),
      'reward': jnp.asarray(rng.normal(0, 1, (B, T)), f32),
      'is_first': jnp.asarray(first),
      'is_last': jnp.zeros((B, T), bool),
      'is_terminal': jnp.zeros((B, T), bool)}
  prevact = {'action': jnp.asarray(rng.uniform(-1, 1, (B, T, ACT)), f32)}
  return obs, prevact


_CACHE = {}


def state():
  """Model, initialized params and one batch, built once for the whole file."""
  if 'state' not in _CACHE:
    model = make_model()
    obs, prevact = make_data()

    def full(obs, prevact):
      carry = tuple(model.init_train(B)[:3])
      return model.loss(carry, obs, prevact, training=True)

    params = nj.init(full)({}, obs, prevact, seed=0)
    _CACHE['state'] = (model, params, (obs, prevact))
  return _CACHE['state']


def run_loss():
  model, params, (obs, prevact) = state()

  def full(obs, prevact):
    carry = tuple(model.init_train(B)[:3])
    return model.loss(carry, obs, prevact, training=True)

  _, (_, (_, _, outs, metrics)) = nj.pure(full)(
      params, obs, prevact, seed=0)
  return model, params, outs, metrics


def grad_norm(key, keyfilter):
  """Norm of d(losses[key]) / d(params matching keyfilter)."""
  model, params, (obs, prevact) = state()

  def gradfn(obs, prevact):
    def inner(obs, prevact):
      carry = tuple(model.init_train(B)[:3])
      _, (_, _, outs, _) = model.loss(carry, obs, prevact, training=True)
      return outs['losses'][key].mean()
    return nj.grad(inner, model.modules)(obs, prevact)

  _, (_, _, grads) = nj.pure(gradfn)(params, obs, prevact, seed=0)
  sel = [v for k, v in grads.items() if keyfilter(k)]
  assert sel, (key, sorted(grads)[:20], 'no params matched')
  return float(jnp.sqrt(sum((v.astype(f32) ** 2).sum() for v in sel)))


_is_det = lambda k: k.startswith('dyn/') and '/det' in k
_is_type = lambda k: k.startswith('dyn/') and (
    '/type0' in k or '/typeout' in k)
_is_rssm = lambda k: k.startswith('dyn/') and any(
    s in k for s in ('/dynin', '/dynhid', '/dyngru', '/obs', '/prior'))
_is_enc = lambda k: k.startswith('enc/')
_is_future = lambda k: k.startswith('future_rew/')


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------

def brute_force(rew, first, H, disc):
  rew, first = np.asarray(rew), np.asarray(first)
  b, t_ = rew.shape
  tgt = np.zeros((b, t_), np.float32)
  msk = np.zeros((b, t_), np.float32)
  wsum = sum(disc ** (k - 1) for k in range(1, H + 1))
  for i in range(b):
    for t in range(t_):
      if t + H >= t_:
        continue
      msk[i, t] = float(
          all(not first[i, t + j] for j in range(1, H + 1)))
      tgt[i, t] = sum(
          disc ** (k - 1) * rew[i, t + k] for k in range(1, H + 1)) / wsum
  return tgt, msk


class TestFutureTarget:

  def test_matches_a_brute_force_reference(self):
    rng = np.random.RandomState(0)
    for H in (1, 2, 5):
      for length in (8, 13):
        rew = jnp.asarray(rng.normal(0, 1, (3, length)), f32)
        first = jnp.asarray(rng.rand(3, length) < 0.2)
        tgt, msk = future_return_target(rew, first, H, 0.997)
        ref_t, ref_m = brute_force(rew, first, H, 0.997)
        assert np.allclose(np.asarray(tgt), ref_t, atol=1e-5), (H, length)
        assert np.allclose(np.asarray(msk), ref_m), (H, length)

  def test_horizon_one_is_the_next_reward(self):
    """The sum starts at r_{t+1}, never r_t."""
    rew = jnp.asarray(np.arange(12, dtype=np.float32).reshape(2, 6))
    first = jnp.zeros((2, 6), bool)
    tgt, msk = future_return_target(rew, first, 1, 0.9)
    assert np.allclose(np.asarray(tgt)[:, :5], np.asarray(rew)[:, 1:])
    assert np.allclose(np.asarray(msk), [[1, 1, 1, 1, 1, 0]] * 2)

  def test_target_uses_exactly_h_rewards(self):
    """A unit impulse at time j contributes to exactly the H frames that
    can see it, which is what "no shortening at the tail" means."""
    H, length = 4, 12
    for j in range(1, length):
      rew = jnp.asarray(np.eye(length, dtype=np.float32)[None, j])
      tgt, msk = future_return_target(
          rew, jnp.zeros((1, length), bool), H, 1.0)
      hit = np.nonzero(np.asarray(tgt)[0] * np.asarray(msk)[0])[0]
      want = [t for t in range(length - H) if j - H <= t <= j - 1]
      assert list(hit) == want, (j, hit, want)

  def test_batch_tail_frames_are_excluded(self):
    H, length = 5, 14
    _, msk = future_return_target(
        jnp.ones((2, length), f32), jnp.zeros((2, length), bool), H, 0.99)
    msk = np.asarray(msk)
    assert msk[:, :length - H].all()
    assert not msk[:, length - H:].any()
    assert msk.sum() == 2 * (length - H)

  def test_the_horizon_never_crosses_a_reset(self):
    H, length = 4, 12
    first = np.zeros((2, length), bool)
    first[:, 7] = True
    _, msk = future_return_target(
        jnp.ones((2, length), f32), jnp.asarray(first), H, 0.99)
    msk = np.asarray(msk)[0]
    # t + j == 7 for some j in 1..H, i.e. t in {3, 4, 5, 6}.
    assert not msk[[3, 4, 5, 6]].any(), msk
    assert msk[[0, 1, 2]].all(), msk

  def test_a_horizon_longer_than_the_sequence_is_all_invalid(self):
    tgt, msk = future_return_target(
        jnp.ones((2, 6), f32), jnp.zeros((2, 6), bool), 6, 0.99)
    assert np.asarray(msk).sum() == 0.0
    assert np.isfinite(np.asarray(tgt)).all()
    assert np.allclose(np.asarray(tgt), 0.0)

  def test_discounting_is_normalized(self):
    """Constant rewards give exactly that constant back, for any discount."""
    for disc in (1.0, 0.9, 0.997):
      tgt, msk = future_return_target(
          jnp.full((2, 12), 3.0, f32), jnp.zeros((2, 12), bool), 5, disc)
      sel = np.asarray(tgt)[np.asarray(msk) > 0]
      assert np.allclose(sel, 3.0, atol=1e-5), (disc, sel)


# ---------------------------------------------------------------------------
# The head input
# ---------------------------------------------------------------------------

class TestFutureFeat:

  def make_feat(self, seed=0):
    rng = np.random.RandomState(seed)
    stoch = jnp.asarray(rng.normal(0, 1, (B, T, 4, 3)), f32)
    event = jnp.asarray((rng.rand(B, T) < 0.5).astype(np.float32))
    onehot = jax.nn.one_hot(rng.randint(0, 4, (B, T)), 4, dtype=f32)
    mix = jnp.concatenate(
        [(1.0 - event)[..., None], event[..., None] * onehot], -1)
    return dict(stoch=stoch, haw_head_mix=mix), event, onehot

  def test_layout_is_z_then_event_then_typed_event(self):
    feat, event, onehot = self.make_feat()
    out = np.asarray(Agent._futurefeat(None, feat))
    assert out.shape == (B, T, 4 * 3 + 1 + 4), out.shape
    assert np.allclose(out[..., :12], np.asarray(feat['stoch']).reshape(
        B, T, 12))
    assert np.allclose(out[..., 12], np.asarray(event))
    assert np.allclose(
        out[..., 13:], np.asarray(event)[..., None] * np.asarray(onehot))

  def test_there_is_no_deterministic_state_in_the_input(self):
    """The head predicts a long-horizon return from the *event*, not from
    the recurrent state that already summarizes the task."""
    feat, _, _ = self.make_feat()
    assert 'deter' not in feat
    width = Agent._futurefeat(None, feat).shape[-1]
    assert width == 4 * 3 + 1 + 4, width

  def test_the_latent_is_detached(self):
    feat, _, _ = self.make_feat()
    grad = jax.grad(
        lambda s: Agent._futurefeat(None, dict(feat, stoch=s)).sum())(
            feat['stoch'])
    assert np.allclose(np.asarray(grad), 0.0)


# ---------------------------------------------------------------------------
# Gradient routing through the full model
# ---------------------------------------------------------------------------

class TestFutureLossRouting:

  def test_it_trains_its_own_head(self):
    assert grad_norm('event_future_rew', _is_future) > 0.0

  def test_it_trains_the_detector_through_the_binary_event(self):
    assert grad_norm('event_future_rew', _is_det) > 0.0

  def test_it_trains_the_classifier_through_the_typed_event(self):
    assert grad_norm('event_future_rew', _is_type) > 0.0

  def test_it_cannot_reach_the_rssm(self):
    """sg(z_t) and no h_t: the auxiliary target must not be able to reshape
    the world model to make itself easier to predict."""
    assert grad_norm('event_future_rew', _is_rssm) == 0.0

  def test_it_cannot_reach_the_encoder(self):
    assert grad_norm('event_future_rew', _is_enc) == 0.0


class TestContinuationRouting:

  def test_continuation_never_reaches_the_classifier(self):
    """ManiSkill never signals task termination, so a near-constant target
    must not be allowed to name event types. Types enter neither the Hawkes
    memory nor the recurrence, so removing them from this head's input
    removes them from its gradient entirely -- unlike the binary event,
    which every [h, z] consumer still reaches through the shared
    y_t -> M_t -> h_{t+1} recurrence. That delayed route is the world
    model's, not this head's, and it is asserted in test_hawkes_rssm.py."""
    assert grad_norm('con', _is_type) == 0.0

  def test_continuation_still_trains_the_world_model(self):
    assert grad_norm('con', _is_rssm) > 0.0

  def test_the_reward_head_still_reaches_both(self):
    assert grad_norm('rew', _is_det) > 0.0
    assert grad_norm('rew', _is_type) > 0.0


# ---------------------------------------------------------------------------
# What each head is allowed to see
# ---------------------------------------------------------------------------

class Spy:
  """Records the input shape of every call and delegates unchanged."""

  def __init__(self, inner):
    self.inner = inner
    self.shapes = []

  def __call__(self, x, *args, **kwargs):
    self.shapes.append(tuple(x.shape))
    return self.inner(x, *args, **kwargs)


class TestHeadInputs:

  def spy(self):
    if 'spy' in _CACHE:
      return _CACHE['spy']
    model = make_model()
    spies = {k: Spy(getattr(model, k))
             for k in ('rew', 'con', 'pol', 'val', 'future_rew')}
    for k, v in spies.items():
      setattr(model, k, v)
    obs, prevact = make_data()

    def full(obs, prevact):
      carry = tuple(model.init_train(B)[:3])
      return model.loss(carry, obs, prevact, training=True)

    nj.init(full)({}, obs, prevact, seed=0)
    base = model.dyn.deter + model.dyn.stoch * model.dyn.classes
    _CACHE['spy'] = (model, spies, base)
    return _CACHE['spy']

  def test_continuation_sees_only_the_base_feature(self):
    model, spies, base = self.spy()
    assert spies['con'].shapes, 'the continuation head was never called'
    assert all(s[-1] == base for s in spies['con'].shapes), (
        spies['con'].shapes, base)

  def test_actor_and_critic_see_only_the_base_feature(self):
    model, spies, base = self.spy()
    for key in ('pol', 'val'):
      assert spies[key].shapes, key
      assert all(s[-1] == base for s in spies[key].shapes), (
          key, spies[key].shapes, base)

  def test_the_reward_head_still_sees_the_event_and_its_type(self):
    model, spies, base = self.spy()
    K = model.dyn.haw_types
    assert all(s[-1] == base + 1 + K for s in spies['rew'].shapes), (
        spies['rew'].shapes, base, K)

  def test_the_future_head_is_never_evaluated_in_imagination(self):
    """It is an auxiliary training target only: not added to the imagined
    reward, not a value target, not an intrinsic reward."""
    model, spies, _ = self.spy()
    assert spies['future_rew'].shapes
    assert all(s[:2] == (B, T) for s in spies['future_rew'].shapes), (
        spies['future_rew'].shapes)


# ---------------------------------------------------------------------------
# Losses, scales and metrics
# ---------------------------------------------------------------------------

class TestLossesAndMetrics:

  def test_the_scale_exists_only_under_hawkes(self):
    assert make_model(hawkes=True).scales['event_future_rew'] == 0.1
    plain = make_model(hawkes=False)
    assert plain.future_rew is None
    assert 'event_future_rew' not in plain.scales

  def test_clustering_is_off_by_default_but_still_computed(self):
    model = make_model()
    assert model.scales['event_conf'] == 0.0
    assert model.scales['event_use'] == 0.0
    _, _, outs, _ = run_loss()
    for key in ('event_conf', 'event_use'):
      assert key in outs['losses'], key

  def test_the_loss_has_the_batch_shape_and_is_finite(self):
    _, _, outs, _ = run_loss()
    loss = np.asarray(outs['losses']['event_future_rew'])
    assert loss.shape == (B, T), loss.shape
    assert np.isfinite(loss).all()
    # Broadcast of one normalized scalar, so every entry is identical.
    assert np.allclose(loss, loss.flat[0])

  def test_the_new_metrics_are_present_and_finite(self):
    _, _, _, metrics = run_loss()
    keys = ['event_future_loss', 'event_future_target_mean',
            'event_future_pred_mean', 'event_future_full_horizon_frac',
            'event_future_sens_binary', 'event_future_sens_type']
    for key in keys:
      assert key in metrics, key
      assert np.isfinite(float(metrics[key])), (key, metrics[key])

  def test_the_continuation_type_spread_metric_is_gone(self):
    """The continuation head no longer consumes types, so reporting a
    spread across them would be reporting a constant zero."""
    _, _, _, metrics = run_loss()
    assert 'event_type_cont_spread' not in metrics
    assert 'event_type_reward_spread' in metrics

  def test_the_full_horizon_fraction_matches_the_target(self):
    model, _, _, metrics = run_loss()
    _, _, (obs, _) = state()
    _, umask = future_return_target(
        obs['reward'], obs['is_first'],
        model.config.event_future_horizon, 1 - 1 / model.config.horizon)
    assert np.isclose(
        float(metrics['event_future_full_horizon_frac']),
        float(np.asarray(umask).mean()), atol=1e-6)
    assert 0.0 < float(metrics['event_future_full_horizon_frac']) < 1.0

  def test_the_loss_is_normalized_by_the_valid_count(self):
    """Not by B * T: the value must not shrink when a batch happens to hold
    fewer full-horizon frames."""
    model, params, outs, metrics = run_loss()
    _, _, (obs, _) = state()
    target, umask = future_return_target(
        obs['reward'], obs['is_first'],
        model.config.event_future_horizon, 1 - 1 / model.config.horizon)

    def fn(repfeat, target):
      return model.future_rew(model._futurefeat(repfeat), 2).loss(target)

    _, ell = nj.pure(fn)(params, outs['repfeat'], target, seed=0)
    ell, umask = np.asarray(ell, np.float64), np.asarray(umask, np.float64)
    want = (umask * ell).sum() / umask.sum()
    assert np.isclose(float(metrics['event_future_loss']), want, rtol=1e-4)
    # And it is genuinely different from the unnormalized mean, so the test
    # above cannot pass by accident.
    assert not np.isclose(want, ell.mean(), rtol=1e-3)

  def test_an_empty_horizon_gives_an_exact_zero(self):
    """Every frame reset: no full-horizon window survives anywhere."""
    model, params, _ = state()
    obs, prevact = make_data(resets=tuple(range(T)))

    def full(obs, prevact):
      carry = tuple(model.init_train(B)[:3])
      return model.loss(carry, obs, prevact, training=True)

    _, (_, (_, _, outs, metrics)) = nj.pure(full)(
        params, obs, prevact, seed=0)
    loss = np.asarray(outs['losses']['event_future_rew'])
    assert np.isfinite(loss).all()
    assert (loss == 0.0).all(), loss
    assert float(metrics['event_future_full_horizon_frac']) == 0.0
