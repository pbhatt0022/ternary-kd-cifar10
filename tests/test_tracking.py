"""Reversal-rate checks driven by the hand-built state sequence of pre-registration §9.

    weight 1: +1 +1 +1 +1   -> 0 reversals
    weight 2:  0 +1  0 +1   -> 2 reversals (epochs 3 and 4)
    weight 3: -1  0 +1 +1   -> 0 reversals (monotone)
    weight 4: -1 +1  0  0   -> 1 reversal (epoch 3)
    weight 5:  0 +1 +1  0   -> 1 reversal (epoch 4; compares to the change at epoch 2)

Epochs 1 and 2 must report None.
"""

import torch

from src.tracking import TernaryTracker

# One filter of 5 weights. Setting a weight to +/-1 and the rest to 0 makes the
# ternarization exactly reproduce the intended state: Delta = 0.75/5 * sum|w| = 0.15*count,
# and count <= 5 keeps Delta <= 0.75 < 1, so every nonzero weight survives the threshold.
SEQUENCE = [
    [+1, 0, -1, -1, 0],
    [+1, +1, 0, +1, +1],
    [+1, 0, +1, 0, +1],
    [+1, +1, +1, 0, 0],
]
EXPECTED_PER_EPOCH = [None, None, 2 / 5, 2 / 5]
EXPECTED_TOTAL_REVERSALS = 4


class _FakeModel:
    """Stands in for a model with one ternarized layer named 'layer'."""

    def __init__(self):
        self.weight = None

    def set_state(self, states):
        self.weight = torch.tensor(states, dtype=torch.float32).reshape(1, 5, 1, 1)

    def get_submodule(self, name):
        assert name == "layer"
        return self


def test_reversal_sequence_matches_prereg_table():
    model = _FakeModel()
    tracker = TernaryTracker(["layer"])
    observed = []

    for states in SEQUENCE:
        model.set_state(states)
        per_layer, aggregate = tracker.update(model)
        rate = per_layer["layer"]["reversal_rate"]
        assert rate == aggregate  # single layer: aggregate is the same number
        observed.append(rate)

    assert observed == EXPECTED_PER_EPOCH
    assert sum(r * 5 for r in observed if r is not None) == EXPECTED_TOTAL_REVERSALS


def test_states_round_trip_through_ternarize():
    """Guards the construction above: the tracker must see exactly the intended states."""
    model = _FakeModel()
    tracker = TernaryTracker(["layer"])
    for states in SEQUENCE:
        model.set_state(states)
        tracker.update(model)
        assert tracker.prev_state["layer"].flatten().tolist() == states


def test_epochs_one_and_two_are_none_not_zero():
    model = _FakeModel()
    tracker = TernaryTracker(["layer"])
    for states in SEQUENCE[:2]:
        model.set_state(states)
        per_layer, aggregate = tracker.update(model)
        assert per_layer["layer"]["reversal_rate"] is None
        assert aggregate is None


def test_zero_fraction():
    model = _FakeModel()
    tracker = TernaryTracker(["layer"])
    model.set_state([+1, 0, 0, 0, -1])  # 3 of 5 zero
    per_layer, _ = tracker.update(model)
    assert per_layer["layer"]["zero_fraction"] == 3 / 5
    assert per_layer["layer"]["n_weights"] == 5


def test_long_gap_compares_to_last_actual_change():
    """A weight stable for many epochs compares against its last change, not epoch t-1."""
    model = _FakeModel()
    tracker = TernaryTracker(["layer"])
    # one weight: 0 -> +1, then stable for 4 epochs, then +1 -> 0. That final move is a
    # reversal against the epoch-2 change even though nothing happened in between.
    states = [[0, 0, 0, 0, 0], [+1, 0, 0, 0, 0], [+1, 0, 0, 0, 0],
              [+1, 0, 0, 0, 0], [+1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    rates = []
    for s in states:
        model.set_state(s)
        per_layer, _ = tracker.update(model)
        rates.append(per_layer["layer"]["reversal_rate"])
    assert rates[2] == 0.0 and rates[3] == 0.0 and rates[4] == 0.0
    assert rates[5] == 1 / 5


def test_state_dict_round_trip():
    """Resume must continue the series, not restart it (amendment A3)."""
    model = _FakeModel()
    a = TernaryTracker(["layer"])
    for states in SEQUENCE[:3]:
        model.set_state(states)
        a.update(model)

    b = TernaryTracker(["layer"])
    b.load_state_dict(a.state_dict())
    assert b.epoch == a.epoch

    model.set_state(SEQUENCE[3])
    resumed, _ = b.update(model)
    model.set_state(SEQUENCE[3])
    direct, _ = a.update(model)
    assert resumed["layer"]["reversal_rate"] == direct["layer"]["reversal_rate"] == 2 / 5
