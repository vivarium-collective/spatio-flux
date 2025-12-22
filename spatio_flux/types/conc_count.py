"""

TODO -- this needs to be called before running processes so that the new count/conc/volume type is known
"""
import numpy as np
from dataclasses import dataclass, is_dataclass, field
from bigraph_schema.schema import Node, Integer, Float, Map, Array
from bigraph_schema.methods import apply, deserialize
from spatio_flux.types.positive import Count, Concentration, Volume


@dataclass(kw_only=True)
class ConcentrationCount(Node):
    concentration: Concentration = field(default_factory=Concentration)
    count: Count = field(default_factory=Count)
    

@dataclass(kw_only=True)
class ConcentrationCountVolume(Node):
    concentration: Concentration = field(default_factory=Concentration)
    count: Count = field(default_factory=Count)
    volume: Volume = field(default_factory=Volume)


@dataclass(kw_only=True)
class SubstratesVolume(Node):
    substrates: Map = field(default_factory=lambda: Map(_value=ConcentrationCount()))
    volume: Volume = field(default_factory=Volume)


@dataclass(kw_only=True)
class ConcentrationsCountsField(Node):
    concentrations: Array = field(default_factory=lambda: Array(_data=np.dtype('float')))
    counts: Array = field(default_factory=lambda: Array(_data=np.dtype('float')))
    

@dataclass(kw_only=True)
class SubstratesFields(Node):
    substrates: Map = field(default_factory=lambda: Map(_value=ConcentrationsCountsField()))
    volume: Volume = field(default_factory=Volume)
    

def apply_concentration_count_volume(schema, current, update, path):
    # TODO -- this needs to be initialized properly so that if conc is given, count and volume are set accordingly
    import ipdb; ipdb.set_trace()

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
        'concentration': concentration_new,
        'count': count_new,
        'volume': V_new,
    }


@apply.dispatch
def apply(schema: ConcentrationCountVolume, current, update, path):
    return apply_concentration_count_volume(schema, current, update, path), []


@deserialize.dispatch
def deserialize(core, schema: ConcentrationCountVolume, encode, path=()):
    import ipdb; ipdb.set_trace()

    final = {}
    if 'volume' not in encode:
        final['volume'] = 1.0
    else:
        final['volume'] = encode['volume']

    if 'concentration' in encode:
        final['concentration'] = encode['concentration']
        final['count'] = final['concentration'] * final['volume']
    elif 'count' in encode:
        final['count'] = encode['count']
        final['concentration'] = final['count'] / final['volume']
    else:
        final['concentration'] = 0.0
        final['count'] = 0.0

    return schema, final, []
