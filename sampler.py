"""Single-spin Metropolis sampling with a compiled trial loop."""

import math
from operator import index
from typing import Protocol

import numpy as np
from numba import njit

from model import IsingModel, _spin_neighbor_product


class Sampler(Protocol):
    """Interface the runner needs from a sampler."""

    @property
    def model(self) -> IsingModel: ...

    @property
    def acceptance_rate(self) -> float: ...

    def reset_statistics(self) -> None: ...

    def sweeps(self, count: int) -> int: ...


@njit(cache=True)
def _metropolis_trials(spins, bond_sum, magnetization, probabilities, rng, attempts):
    """Run independent site proposals and return updated totals and acceptances."""
    L = spins.shape[0]
    accepted = 0
    for _ in range(attempts):
        i = rng.integers(0, L)
        j = rng.integers(0, L)
        product = _spin_neighbor_product(spins, i, j)
        # The spin-neighbor product takes values -4, -2, 0, 2, 4.
        probability = probabilities[(product + 4) // 2]
        if probability == 1.0 or rng.random() < probability:
            old_spin = spins[i, j]
            spins[i, j] = -old_spin
            bond_sum -= 2 * product
            magnetization -= 2 * old_spin
            accepted += 1
    return bond_sum, magnetization, accepted


class MetropolisSampler:
    """Sample a model at fixed temperature in units with k_B = 1."""

    def __init__(self, model: IsingModel, T: float, seed=None):
        self._model = model
        self._T = float(T)
        if not math.isfinite(self._T) or self._T < 0:
            raise ValueError("T must be finite and nonnegative")
        self._rng = np.random.default_rng(seed)
        self._probabilities = np.ones(5)
        for entry, product in enumerate(range(-4, 5, 2)):
            delta_energy = (2 * product) * model.J
            if delta_energy > 0:
                self._probabilities[entry] = (
                    math.exp(-delta_energy / self._T) if self._T > 0 else 0.0
                )
        self.reset_statistics()

    @property
    def model(self):
        return self._model

    @property
    def T(self):
        return self._T

    @property
    def acceptance_rate(self):
        """Fraction of attempted flips accepted since the last reset."""
        return self.accepted_flips / self.attempted_flips if self.attempted_flips else 0.0

    def reset_statistics(self):
        """Reset acceptance counters while retaining the lattice and RNG."""
        self.attempted_flips = 0
        self.accepted_flips = 0

    def _attempt(self, attempts):
        model = self.model
        model._bond_sum, model._magnetization, accepted = _metropolis_trials(
            model._spins,
            model._bond_sum,
            model._magnetization,
            self._probabilities,
            self._rng,
            attempts,
        )
        self.attempted_flips += attempts
        self.accepted_flips += accepted
        return accepted

    def step(self):
        """Attempt one flip and report whether it was accepted."""
        return bool(self._attempt(1))

    def sweep(self):
        """Attempt L² flips and return the number accepted."""
        return self.sweeps(1)

    def sweeps(self, count):
        """Run a batch of sweeps in one JIT call and return its acceptances."""
        count = index(count)
        if count < 0:
            raise ValueError("sweep count must be nonnegative")
        if count == 0:
            return 0
        return self._attempt(count * self.model.N)
