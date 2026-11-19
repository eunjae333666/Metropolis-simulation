"""Square-lattice Ising model with periodic boundaries."""

import math
from operator import index

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _spin_neighbor_product(spins, i, j):
    """Return the spin times the sum of its four periodic neighbors."""
    last = spins.shape[0] - 1
    above = i - 1 if i > 0 else last
    below = i + 1 if i < last else 0
    left = j - 1 if j > 0 else last
    right = j + 1 if j < last else 0
    return spins[i, j] * (
        spins[above, j] + spins[below, j] + spins[i, left] + spins[i, right]
    )


class IsingModel:
    """L × L, zero-field Ising model with L >= 2 and spins in {-1, +1}."""

    def __init__(self, L, J=1.0, initialization="random", seed=None):
        L = index(L)
        if L < 2:
            raise ValueError("L must be at least 2")
        self._J = float(J)
        if not math.isfinite(self._J):
            raise ValueError("J must be finite")

        if initialization == "random":
            rng = np.random.default_rng(seed)
            self._spins = rng.integers(0, 2, size=(L, L), dtype=np.int8)
            self._spins *= 2
            self._spins -= 1
        elif initialization in ("up", "down"):
            spin = 1 if initialization == "up" else -1
            self._spins = np.full((L, L), spin, dtype=np.int8)
        else:
            raise ValueError("initialization must be 'random', 'up', or 'down'")

        # Keep the bond sum as an integer to avoid accumulating roundoff in E.
        self._bond_sum = self._calculate_bond_sum()
        self._magnetization = self.calculate_magnetization()

    @property
    def spins(self):
        """Read-only lattice view for measurements and plotting."""
        view = self._spins.view()
        view.flags.writeable = False
        return view

    @property
    def L(self):
        return self._spins.shape[0]

    @property
    def N(self):
        return self._spins.size

    @property
    def J(self):
        return self._J

    @property
    def energy(self):
        return -self.J * self._bond_sum

    @property
    def magnetization(self):
        return self._magnetization

    @property
    def energy_per_spin(self):
        return self.energy / self.N

    @property
    def magnetization_per_spin(self):
        return self.magnetization / self.N

    @property
    def abs_magnetization_per_spin(self):
        return abs(self.magnetization_per_spin)

    def _calculate_bond_sum(self):
        # Count the bond to the right and below each site once, including wraps.
        spins = self._spins
        right = np.sum(spins * np.roll(spins, -1, axis=1), dtype=np.int64)
        below = np.sum(spins * np.roll(spins, -1, axis=0), dtype=np.int64)
        return int(right + below)

    def calculate_energy(self):
        """Recompute the total energy from the full lattice."""
        return -self.J * self._calculate_bond_sum()

    def calculate_magnetization(self):
        """Recompute the total magnetization from the full lattice."""
        return int(self._spins.sum(dtype=np.int64))

    def delta_energy(self, i, j):
        """Return the energy change of flipping a site in O(1)."""
        product = _spin_neighbor_product(self._spins, i % self.L, j % self.L)
        return (2 * product) * self.J

    def flip_spin(self, i, j):
        """Flip a site and update the cached totals in O(1)."""
        i, j = i % self.L, j % self.L
        old_spin = int(self._spins[i, j])
        self._bond_sum -= 2 * _spin_neighbor_product(self._spins, i, j)
        self._magnetization -= 2 * old_spin
        self._spins[i, j] = -old_spin
