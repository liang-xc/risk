import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from cds import CDS


@dataclass
class CDXCDO:
    """
    A class to represent a CDX CDO
    The major goal of this class is to expected tranches losses for a one-period
    CDO with copula

    Attributes:
        cds_port: a basket of cds for default probabilities calculations
        rho: copula correlation
    """

    cds_port: list[CDS]
    rho: float = 0.5  # value from Hull p.262

    def __post_init__(self) -> None:
        """
        initiate post_init attributes

        Attributes:
            num_sec: number of securities
            uncond_pd: 1 period unconditional default probabilities from cds_port
            _seed: seed for random number generator
            wm: generated random numbers for market factor
                from standard normal distribution
        """
        self.num_sec: int = len(self.cds_port)
        self.uncond_pd: np.ndarray = np.array([c.survival_prob(1) for c in self.cds_port])
        self._seed = 1222
        rng = np.random.default_rng(self._seed)
        self.wm = rng.normal(size=1_000)

    def prob(self, uncon_prob: float, f: float) -> float:
        """
        helper function to calculate conditonal probability

        Args:
            uncon_prob (float): unconditional probability
            f (float): condition Wm = f

        Returns:
            float: conditonal probability
        """
        return stats.norm.pdf(
            (stats.norm.ppf(uncon_prob) - np.sqrt(self.rho) * f) / np.sqrt(1 - self.rho)
        )

    def conditional_pd(self) -> float:
        """
        use copula to get conditional PD

        Returns:
            float: conditional default probabilities
                assumed to be equal under Vasecik
        """
        result = np.empty((self.num_sec, len(self.wm)))
        for i, pi in enumerate(self.uncond_pd):
            for j, f in enumerate(self.wm):
                result[i, j] = self.prob(pi, f)
        return np.mean(result)

    def loss(self, num_of_default: int) -> np.ndarray:
        """
        use Vasecik to calculate loss function

        Args:
            num_of_default (int): number of loan defaults

        Returns:
            np.ndarray: loss
        """
        assert (
            num_of_default <= self.num_sec
        ), "number of default exceeds total number of debt"

        result = np.zeros(self.num_sec)
        p = self.conditional_pd()
        for i in range(num_of_default):
            integral = np.mean(
                [
                    self.prob(p, f) ** i * (1 - self.prob(p, f)) ** (self.num_sec - i)
                    for f in self.wm
                ]
            )
            result[i] = math.comb(self.num_sec, i) * integral

        return np.sum(result)
