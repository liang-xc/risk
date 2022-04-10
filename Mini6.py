from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import optimize, stats


@dataclass
class KMV:
    """
    a class to represent a KMV model

    Attributes:
    STD: short term debt
    LTD: long term debt
    Et: equity value
    sigma_E: equity volatility
    tao: time to maturity
    r: risk-free rate
    """

    STD: float
    LTD: float
    Et: float
    sigma_E: float
    tao: float = 1.0
    r: float = 0.02

    @cached_property
    def K(self) -> float:
        return self.STD + self.LTD / 2

    def d2(self, a_para: np.ndarray) -> float:
        return (
            np.log(a_para[0])
            - np.log(self.K)
            + (self.r - (a_para[1] ** 2) / 2) * self.tao
        ) / (a_para[1] * np.sqrt(self.tao))

    def d1(self, a_para: np.ndarray) -> float:
        return self.d2(a_para) + a_para[1] * np.sqrt(self.tao)

    def E_eq(self, a_para: np.ndarray) -> float:
        """equation for Et"""
        return self.Et - (
            a_para[0] * stats.norm.cdf(self.d1(a_para))
            - np.exp(-self.r * self.tao) * self.K * stats.norm.cdf(self.d2(a_para))
        )

    def sigma_eq(self, a_para: np.ndarray) -> float:
        """equation for sigma"""
        return (
            self.sigma_E
            - a_para[1] * stats.norm.cdf(self.d1(a_para)) * a_para[0] / self.Et
        )

    @cached_property
    def KMV_solver(self) -> np.ndarray:
        """solve for A and sigma"""

        def eqs(a_para):
            return [self.E_eq(a_para), self.sigma_eq(a_para)]

        sol = optimize.root(eqs, np.array([1_000_000_000, 1_000_000]), method="broyden1")
        return sol.x

    def distance_to_default(self) -> float:
        return self.d2(self.KMV_solver)

    def probability_of_default(self) -> float:
        return 1 - stats.norm.cdf(self.d2(self.KMV_solver))

    def expected_recovery(self) -> float:
        return self.KMV_solver[0] * (1 - stats.norm.cdf(self.d1(self.KMV_solver)))


def main():
    # INTC data
    intc_std = 4_771_000_000
    intc_ltd = 33_805_000_000
    intc_e = 192_250_000_000

    inteldf = pd.read_csv("./HW1_data/INTC.csv")
    intc_return = inteldf["Close"].pct_change()
    intc_e_sigma = np.std(intc_return)

    intc_kmv = KMV(intc_std, intc_ltd, intc_e, intc_e_sigma)

    print("At and sigma_t is: ")
    print(intc_kmv.KMV_solver)

    print(f"DD: {intc_kmv.distance_to_default()}")
    print(f"PD: {intc_kmv.probability_of_default()}")
    print(f"expected recovery: {intc_kmv.expected_recovery()}")


if __name__ == "__main__":
    main()
