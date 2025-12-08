from dataclasses import dataclass, is_dataclass, field
from bigraph_schema.schema import Node, Integer, Float
from bigraph_schema.methods import apply


@dataclass(kw_only=True)
class CountConcentrationVolume(Node):
    counts: Integer = field(default_factory=Integer)
    concentration: Float = field(default_factory=Float)
    volume: Float = field(default_factory=Float)


def apply_conc_counts_volume(schema, current, update, path):
    """
    Type: {
        'volume': float,          # container size
        'counts': float,          # total amount
        'concentration': float,   # counts / volume
    }

    Semantics:
      - Updates are treated as *deltas*:
          update = {
              'volume': ΔV (optional),
              'counts': ΔN (optional),
              'concentration': ΔC (optional),
          }
      - Counts are the canonical amount.
      - Concentration is derived: concentration = counts / volume.
      - If volume changes, we keep counts (amount) fixed and recompute concentration.
      - If concentration changes, we interpret ΔC as an additional amount: ΔN_conc = ΔC * V_new.
    """
    if current is None:
        current = {'volume': 0.0, 'counts': 0.0, 'concentration': 0.0}

    if not isinstance(update, dict):
        raise ValueError(
            f"Update to conc_counts_volume at {path} must be a dict, got {type(update)}"
        )

    # Extract current state
    volume = float(current.get('volume', 0.0))
    counts = float(current.get('counts', 0.0))

    # Extract deltas (default to 0)
    dV = float(update.get('volume', 0.0)) if 'volume' in update else 0.0
    dN = float(update.get('counts', 0.0)) if 'counts' in update else 0.0
    dC = float(update.get('concentration', 0.0)) if 'concentration' in update else 0.0

    # 1. Update volume first
    V_new = volume + dV
    if V_new <= 0:
        raise ValueError(
            f"Volume would become non-positive at {path}: {V_new}"
        )

    # 2. Interpret all changes as changes in amount (counts are canonical)
    amount = counts
    d_amount_from_counts = dN
    d_amount_from_conc = dC * V_new  # concentration * volume = counts (in arbitrary units)

    amount_new = amount + d_amount_from_counts + d_amount_from_conc

    # Enforce non-negativity on amount
    if amount_new < 0:
        amount_new = 0.0

    counts_new = amount_new
    concentration_new = counts_new / V_new if V_new > 0 else 0.0

    return {
        'volume': V_new,
        'counts': counts_new,
        'concentration': concentration_new,
    }

@apply.dispatch
def apply(schema: CountConcentrationVolume, current, update, path):
    return apply_conc_counts_volume(schema, current, update, path)


