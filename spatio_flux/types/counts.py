from dataclasses import dataclass, is_dataclass, field
from bigraph_schema.schema import Node, Integer, Float
from bigraph_schema.methods import apply


@dataclass(kw_only=True)
class CountConcentrationVolume(Node):
    count: Float = field(default_factory=Float)
    concentration: Float = field(default_factory=Float)
    volume: Float = field(default_factory=Float)


def apply_count_concentration_volume(schema, current, update, path):
    """
    Type: {
        'volume': float,          # container size
        'count': float,          # total amount
        'concentration': float,   # count / volume
    }

    Semantics:
      - Updates are treated as *deltas*:
          update = {
              'volume': ΔV (optional),
              'count': ΔN (optional),
              'concentration': ΔC (optional),
          }
      - Count are the canonical amount.
      - Concentration is derived: concentration = count / volume.
      - If volume changes, we keep count (amount) fixed and recompute concentration.
      - If concentration changes, we interpret ΔC as an additional amount: ΔN_conc = ΔC * V_new.
    """
    if current is None:
        current = {'volume': 0.0, 'count': 0.0, 'concentration': 0.0}

    if not isinstance(update, dict):
        raise ValueError(
            f"Update to conc_count_volume at {path} must be a dict, got {type(update)}"
        )

    # Extract current state
    volume = float(current.get('volume', 0.0))
    count = float(current.get('count', 0.0))

    # Extract deltas (default to 0)
    dV = float(update.get('volume', 0.0)) if 'volume' in update else 0.0
    dN = float(update.get('count', 0.0)) if 'count' in update else 0.0
    dC = float(update.get('concentration', 0.0)) if 'concentration' in update else 0.0

    # 1. Update volume first
    V_new = volume + dV
    if V_new <= 0:
        raise ValueError(
            f"Volume would become non-positive at {path}: {V_new}"
        )

    # 2. Interpret all changes as changes in amount (count are canonical)
    amount = count
    d_amount_from_count = dN
    d_amount_from_conc = dC * V_new  # concentration * volume = count (in arbitrary units)

    amount_new = amount + d_amount_from_count + d_amount_from_conc

    # Enforce non-negativity on amount
    if amount_new < 0:
        amount_new = 0.0

    count_new = amount_new
    concentration_new = count_new / V_new if V_new > 0 else 0.0

    return {
        'volume': V_new,
        'count': count_new,
        'concentration': concentration_new,
    }


@apply.dispatch
def apply(schema: CountConcentrationVolume, current, update, path):
    return apply_count_concentration_volume(schema, current, update, path), []

