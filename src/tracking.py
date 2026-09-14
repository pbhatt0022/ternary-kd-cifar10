"""Per-layer ternary tracking metrics. Pre-registration §7. Ternary arms only.

Computed once per epoch, never per step. The reversal rate is the subtle one: a reversal
compares against a weight's most recent prior change, however many epochs ago that was,
not against the previous epoch's direction.
"""

import torch

from src.quantizer import ternarize

REVERSAL_DEFINED_FROM_EPOCH = 3  # prereg §7: needs two prior changes to compare


class TernaryTracker:
    """Holds, per ternarized layer, two persistent buffers shaped like the weight:

    prev_state      -- ternary assignment at the end of the previous epoch, in {-1,0,+1}
    last_change_dir -- direction of that weight's most recent state change, ever;
                       0 means "has never changed"

    Both go into resume.pt (amendment A3), so a resumed run continues the same series
    rather than restarting it and under-reporting reversals.
    """

    def __init__(self, layer_names):
        self.names = list(layer_names)
        self.epoch = 0
        self.prev_state = {}
        self.last_change_dir = {}

    @torch.no_grad()
    def update(self, model):
        """Call at the end of each epoch. Returns (per_layer dict, aggregate_or_None)."""
        self.epoch += 1
        per_layer = {}
        total_reversals = 0
        total_weights = 0

        for name in self.names:
            module = model.get_submodule(name)
            w_t, _ = ternarize(module.weight.detach())
            # alpha > 0, so sign(w_t) recovers the assignment exactly.
            cur = torch.sign(w_t).to(torch.int8)
            n = cur.numel()

            if name in self.prev_state:
                prev = self.prev_state[name]
                lcd = self.last_change_dir[name]
                changed = cur != prev
                direction = torch.sign(cur - prev)  # in {-2..2} -> sign in {-1,0,+1}
                reversal = changed & (lcd != 0) & (direction != lcd)
                n_reversals = int(reversal.sum())
                self.last_change_dir[name] = torch.where(changed, direction, lcd)
            else:
                n_reversals = 0
                self.last_change_dir[name] = torch.zeros_like(cur)
            self.prev_state[name] = cur

            # Degenerate count comes from the last forward pass of the epoch (handoff
            # §4.6), not from the recomputation above: the weights moved since then.
            degenerate = getattr(module, "_last_degenerate", None)

            per_layer[name] = {
                "zero_fraction": int((cur == 0).sum()) / n,
                "reversal_rate": self._rate(n_reversals, n),
                "degenerate_filter_count": (
                    None if degenerate is None else int(degenerate.sum())
                ),
                "n_weights": n,
            }
            total_reversals += n_reversals
            total_weights += n

        return per_layer, self._rate(total_reversals, total_weights)

    def _rate(self, n_reversals, n_weights):
        """None -- emitted as JSON null -- for epochs 1 and 2, where no reversal is
        detectable. A 0.0 there would be indistinguishable from a genuinely stable epoch.
        Denominator is ALL weights in the layer, not only those that ever changed."""
        if self.epoch < REVERSAL_DEFINED_FROM_EPOCH:
            return None
        return n_reversals / n_weights

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "prev_state": {k: v.clone() for k, v in self.prev_state.items()},
            "last_change_dir": {k: v.clone() for k, v in self.last_change_dir.items()},
        }

    def load_state_dict(self, sd):
        # Both directions copy. Handing out or adopting live references makes the restored
        # tracker alias the original, so an update to one silently rewrites the other's
        # prev_state and its next reversal count comes out zero.
        self.epoch = sd["epoch"]
        self.prev_state = {k: v.clone() for k, v in sd["prev_state"].items()}
        self.last_change_dir = {k: v.clone() for k, v in sd["last_change_dir"].items()}
