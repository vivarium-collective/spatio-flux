
import numpy as np
import math


class ConcCountsGrid:
    """
    Grid of conc_counts over space for a single molecule.

    - volume: 2D array of floats
    - counts: 2D array (float or int, your choice)
    - concentration is derived: counts / volume

    Supports:
        grid[i, j]                 -> ConcCountsCell
        grid['concentration']      -> 2D array of concentrations
        grid.concentration         -> 2D array of concentrations

    And vectorized updates via the concentration property setter.
    """

    def __init__(self, volume, counts):
        volume = np.asarray(volume, dtype=float)
        counts = np.asarray(counts, dtype=float)  # can later cast to int if desired

        if volume.shape != counts.shape:
            raise ValueError("volume and counts must have the same shape")

        self.volume = volume
        self.counts = counts

    @classmethod
    def zeros(cls, shape, volume=1.0):
        """Convenience constructor: uniform volume, zero counts."""
        vol = np.full(shape, volume, dtype=float)
        counts = np.zeros(shape, dtype=float)
        return cls(vol, counts)

    # --- unified access: grid[i, j] vs grid['concentration'] ---

    def __getitem__(self, key):
        """
        If key is a string: return the field array (volume/counts/concentration).
        Otherwise: treat as an index and return a ConcCountsCell.
        """
        if isinstance(key, str):
            if key == 'volume':
                return self.volume
            elif key == 'counts':
                return self.counts
            elif key == 'concentration':
                return self.concentration
            else:
                raise KeyError(f"Unknown key for ConcCountsGrid: {key!r}")
        else:
            # index (i, j, slice, etc.) -> cell proxy
            return ConcCountsCell(self, key)

    @property
    def shape(self):
        return self.volume.shape

    # --- concentration as a derived, vectorized field ---

    @property
    def concentration(self):
        """Full 2D concentration array (derived)."""
        return self.counts / self.volume

    @concentration.setter
    def concentration(self, C_new):
        """
        Set absolute concentration field, vectorized:
            counts = C_new * volume
        """
        C_new = np.asarray(C_new, dtype=float)
        if C_new.shape != self.volume.shape:
            raise ValueError(
                f"Concentration shape {C_new.shape} does not match volume shape {self.volume.shape}"
            )
        self.counts[...] = C_new * self.volume

    # --- optional helpers for vectorized updates (deltas) ---

    def add_counts(self, dN):
        """
        Add a delta counts field (same shape) to the grid.
        """
        dN = np.asarray(dN, dtype=float)
        if dN.shape != self.counts.shape:
            raise ValueError("dN shape must match counts shape")
        self.counts += dN

    def add_concentration(self, dC):
        """
        Add a delta concentration field (same shape):
            ΔN = ΔC * V  (elementwise).
        """
        dC = np.asarray(dC, dtype=float)
        if dC.shape != self.volume.shape:
            raise ValueError("dC shape must match volume shape")
        self.counts += dC * self.volume



class ConcCountsCell:
    """
    Proxy for a single (i, j) cell in a ConcCountsGrid.

    Supports:
        cell.concentration
        cell.counts
        cell.volume

        cell['concentration']
        cell['counts']
        cell['volume']

    Assignments:
        cell.counts = new_counts
        cell.concentration = new_concentration   # N = C * V
        cell.volume = new_volume                 # keep counts; C adjusts
    """

    def __init__(self, grid: ConcCountsGrid, idx):
        self._grid = grid
        self._idx = idx  # e.g. (i, j)

    # --- attribute access ---

    @property
    def volume(self):
        return self._grid.volume[self._idx]

    @volume.setter
    def volume(self, value):
        self._grid.volume[self._idx] = float(value)

    @property
    def counts(self):
        return self._grid.counts[self._idx]

    @counts.setter
    def counts(self, value):
        self._grid.counts[self._idx] = float(value)

    @property
    def concentration(self):
        return self._grid.counts[self._idx] / self._grid.volume[self._idx]

    @concentration.setter
    def concentration(self, value):
        V = self._grid.volume[self._idx]
        self._grid.counts[self._idx] = float(value) * V

    # --- dict-like access with same semantics ---

    def __getitem__(self, key):
        if key == 'volume':
            return self.volume
        elif key == 'counts':
            return self.counts
        elif key == 'concentration':
            return self.concentration
        else:
            raise KeyError(f"Unknown key for ConcCountsCell: {key!r}")

    def __setitem__(self, key, value):
        if key == 'volume':
            self.volume = value
        elif key == 'counts':
            self.counts = value
        elif key == 'concentration':
            self.concentration = value
        else:
            raise KeyError(f"Unknown key for ConcCountsCell: {key!r}")


if __name__ == "__main__":
    # Example usage / quick sanity checks

    # -------------------------------
    # 1. Create a substrate_array with two species
    # -------------------------------
    shape = (4, 4)
    substrate_array = {
        "glc": ConcCountsGrid.zeros(shape, volume=1.0),
        "o2":  ConcCountsGrid.zeros(shape, volume=2.0),
    }

    glc = substrate_array["glc"]
    o2 = substrate_array["o2"]

    print("Initial GLc volume field:\n", glc.volume)
    print("Initial GLc counts field:\n", glc.counts)
    print("Initial GLc concentration field:\n", glc.concentration)
    print()

    # -------------------------------
    # 2. Per-site operations
    # -------------------------------
    i, j = 1, 2

    # Access local concentration (attribute)
    c_local_attr = glc[i, j].concentration
    # Access local concentration (dict-style)
    c_local_dict = glc[i, j]["concentration"]

    print(f"Initial GLc concentration at ({i}, {j}) [attr]: {c_local_attr}")
    print(f"Initial GLc concentration at ({i}, {j}) [dict]: {c_local_dict}")

    # Set local counts directly
    glc[i, j].counts = 10.0
    print(f"After setting counts=10 at ({i}, {j}), concentration:",
          glc[i, j].concentration)

    # Set local concentration instead (N = C * V)
    glc[i, j].concentration = 5.0
    print(f"After setting concentration=5 at ({i}, {j}):")
    print("  counts:", glc[i, j].counts)
    print("  volume:", glc[i, j].volume)
    print("  concentration:", glc[i, j].concentration)
    print()

    # -------------------------------
    # 3. Full-field operations
    # -------------------------------
    # Set a uniform concentration field for GLc
    glc.concentration = np.ones(glc.shape) * 2.0  # C = 2 everywhere, so N = 2 * V
    print("GLc counts after setting uniform concentration=2:\n", glc.counts)
    print("GLc concentration field:\n", glc.concentration)
    print()

    # Add some random counts (vectorized)
    dN = np.random.poisson(1.0, size=glc.shape)
    glc.add_counts(dN)
    print("GLc counts after adding random dN:\n", glc.counts)
    print("GLc concentration after adding dN:\n", glc.concentration)
    print()

    # Add a delta concentration field (vectorized)
    dC = np.full(glc.shape, 0.5)  # add 0.5 concentration everywhere
    glc.add_concentration(dC)
    print("GLc counts after adding dC=0.5 everywhere:\n", glc.counts)
    print("GLc concentration after adding dC:\n", glc.concentration)
    print()

    # -------------------------------
    # 4. Same API works for O2
    # -------------------------------
    # Different volume, same operations
    o2.concentration = np.linspace(0, 1, o2.volume.size).reshape(o2.shape)
    print("O2 volume field:\n", o2.volume)
    print("O2 counts derived from concentration * volume:\n", o2.counts)
    print("O2 concentration field:\n", o2.concentration)
    print()

    # Per-site check for O2
    i2, j2 = 2, 3
    print(f"O2 at ({i2}, {j2}):")
    print("  volume:", o2[i2, j2].volume)
    print("  counts:", o2[i2, j2].counts)
    print("  concentration:", o2[i2, j2].concentration)


