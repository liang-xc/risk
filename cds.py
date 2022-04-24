from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np
from scipy import optimize


@dataclass
class CDS:
    """
    A class to represent a Credit Default Swap

    Attributes:
        id
        period
        spread
        recovery_rate
        risk_free_rate
    """

    id: float
    period: np.ndarray
    spread: np.ndarray
    recovery_rate: float = 0.4
    risk_free_rate: float = 0.02

    def __hash__(self) -> int:
        return hash(self.id)

    @cached_property
    def discount_factor(self) -> np.ndarray:
        """
        calculate discount factors for period
        rf: risk-free rate
        return discount factors in chronological order
        """

        return np.array([np.exp(-self.risk_free_rate * t) for t in self.period])

    @cache
    def survival_prob(self, idx: int) -> float:
        """
        calculates the survival probabilities at idx
        based on back-of-the-envelop formula and recursive algorithm

        """

        if idx == 0:
            p1 = self.spread[0] / (1 - self.recovery_rate)
            return 1 - p1
        elif idx == -1:
            return 1  # used for default leg calculation
        else:

            def prob_calc(prob: float) -> float:
                # premium leg
                premium_pv = 0
                for i in range(idx):  # every payment before last period
                    premium_pv += self.spread[idx] * (
                        self.survival_prob(i) * self.discount_factor[i]
                    )
                premium_pv += self.spread[idx] * (
                    prob * self.discount_factor[idx]
                )  # last period

                # default leg
                default_pv = 0
                for i in range(idx):  # every payment before last period
                    default_pv += (1 - self.recovery_rate) * (
                        (self.survival_prob(i - 1) - self.survival_prob(i))
                        * self.discount_factor[i]
                    )
                default_pv += (1 - self.recovery_rate) * (
                    (self.survival_prob(idx - 1) - prob) * self.discount_factor[i]
                )  # last period

                return premium_pv - default_pv

            return optimize.brentq(prob_calc, 0, 1)

    def survival_prob_list(self) -> list[float]:
        """
        calculates all survival probability within maturity
        return: list of probability in chronological order
        """
        return [self.survival_prob(i) for i in range(len(self.period))]
