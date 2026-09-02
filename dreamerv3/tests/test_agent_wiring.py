"""Agent-level wiring tests that do not need a full Agent."""

from dreamerv3.agent import _dyn_policy_kw


class TestPolicyKwargs:

  def test_eval_samples_events_like_training(self):
    """Regression guard: evaluation must not fall back to the 0.5 threshold,
    or no event ever fires at a sparse rho and the event panels stay empty."""
    train = _dyn_policy_kw(True, 'train')
    eval_ = _dyn_policy_kw(True, 'eval')
    assert train == eval_, (train, eval_)
    assert eval_['sample_event'] is True

  def test_non_hawkes_gets_no_event_flag(self):
    assert 'sample_event' not in _dyn_policy_kw(False, 'eval')
