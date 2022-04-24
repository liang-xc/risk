from dataclasses import dataclass
from functools import cached_property

import numpy as np
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
        """
        use a combination of STD and LTD as the one term debt for Merton's model

        Returns:
            float: K
        """
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
        """
        Equation for Et, used for root finding

        Args:
            a_para (np.ndarray): A, sigma


        Returns:
            float: difference
        """
        return self.Et - (
            a_para[0] * stats.norm.cdf(self.d1(a_para))
            - np.exp(-self.r * self.tao) * self.K * stats.norm.cdf(self.d2(a_para))
        )

    def sigma_eq(self, a_para: np.ndarray) -> float:
        """
        Equation for sigma_E, used for root finding

        Args:
            a_para (np.ndarray): A, sigma

        Returns:
            float: difference
        """
        return (
            self.sigma_E
            - a_para[1] * stats.norm.cdf(self.d1(a_para)) * a_para[0] / self.Et
        )

    @cached_property
    def KMV_solver(self) -> np.ndarray:
        """
        solve for A and sigma

        Returns:
            np.ndarray: A, sigma
        """

        def eqs(a_para):
            return [self.E_eq(a_para), self.sigma_eq(a_para)]

        sol = optimize.root(eqs, np.array([1_000_000_000, 1_000_000]), method="df-sane")
        return sol.x

    def distance_to_default(self) -> float:
        """
        distance to default (DD)

        Returns:
            float: DD
        """
        return self.d2(self.KMV_solver)

    def probability_of_default(self) -> float:
        """
        probability of default (PD)

        Returns:
            float: PD
        """
        return 1 - stats.norm.cdf(self.d2(self.KMV_solver))

    def expected_recovery(self) -> float:
        """
        expected recovery rate

        Returns:
            float:
        """
        return self.KMV_solver[0] * (1 - stats.norm.cdf(self.d1(self.KMV_solver)))


@dataclass
class Geske:
    """
    A class to represent a 2 period 2 debt Geske Model

    Attributes:
        t: time interval for the model
        k1: first debt (STD)
        t1: time to maturity of k1
        k2: second debt (LTD)
        t2: time to maturity of k2
        Et: equity value
        sigma_E: equity volatility
        r: risk-free rate
    """

    t: float
    k1: float
    t1: float
    k2: float
    t2: float
    Et: float
    sigma_E: float
    r: float = 0.02

    def __post_init__(self) -> None:
        """
        initiate post_init attributes

        Attributes:
            rho: correlation coefficient of two debts
        """
        self.rho: float = np.sqrt((self.t1 - self.t) / (self.t2 - self.t))

    def A1_bar_eq(self, a) -> float:
        """
        equation used to solve for A1_bar
        """
        x1 = (
            np.log(a)
            - np.log(self.k2)
            + (self.r + self.sigma_E**2 / 2) * (self.t2 - self.t1)
        ) / (self.sigma_E * np.sqrt(self.t2 - self.t1))

        x2 = (
            np.log(a)
            - np.log(self.k2)
            + (self.r - self.sigma_E**2 / 2) * (self.t2 - self.t1)
        ) / (self.sigma_E * np.sqrt(self.t2 - self.t1))

        return (
            self.k1
            - a * stats.norm.cdf(x1)
            - np.exp(-self.r * (self.t2 - self.t1)) * self.k2 * stats.norm.cdf(x2)
        )

    @cached_property
    def A1_bar(self) -> float:
        """
        solve for A1_bar
        """
        sol = optimize.brentq(self.A1_bar_eq, 0, 1e20)
        return sol

    def Geske_eqs(self, a_para: np.ndarray) -> np.ndarray:
        """
        Equations for E_t and sigma_E
        used for root finding

        Args:
            a_para (np.ndarray): A, sigma

        Returns:
            np.ndarray: differences of two equaitons
        """
        h11 = (
            np.log(a_para[0])
            - np.log(self.A1_bar)
            + (self.r + a_para[1] ** 2 / 2) * (self.t1 - self.t)
        ) / (a_para[1] * np.sqrt(self.t1 - self.t))

        h12 = (
            np.log(a_para[0])
            - np.log(self.A1_bar)
            + (self.r - a_para[1] ** 2 / 2) * (self.t1 - self.t)
        ) / (a_para[1] * np.sqrt(self.t1 - self.t))

        h21 = (
            np.log(a_para[0])
            - np.log(self.k2)
            + (self.r + a_para[1] ** 2 / 2) * (self.t2 - self.t)
        ) / (a_para[1] * np.sqrt(self.t2 - self.t))

        h22 = (
            np.log(a_para[0])
            - np.log(self.k2)
            + (self.r - a_para[1] ** 2 / 2) * (self.t2 - self.t)
        ) / (a_para[1] * np.sqrt(self.t2 - self.t))

        Et_eq = (
            a_para[0]
            * stats.multivariate_normal.cdf([h11, h21], mean=np.zeros(2), cov=self.rho)
            - np.exp(-self.r * (self.t2 - self.t))
            * self.k2
            * stats.multivariate_normal.cdf([h12, h22], mean=np.zeros(2), cov=self.rho)
            - np.exp(-self.r * (self.t1 - self.t)) * self.k1 * stats.norm.cdf(h12)
            - self.Et
        )

        sigma_eq = (
            a_para[0]
            / self.Et
            * a_para[1]
            * stats.multivariate_normal.cdf([h11, h21], mean=np.zeros(2), cov=self.rho)
            - self.sigma_E
        )

        return np.array([Et_eq, sigma_eq])

    @cached_property
    def Geske_solver(self) -> np.ndarray:
        """
        solve for A and sigma

        Returns:
            np.ndarray: A, sigma
        """
        converge_flag = False
        sol = optimize.root(
            self.Geske_eqs, np.array([1_000_000_000, 1]), method="df-sane"
        )
        converge_flag = sol.success

        if not converge_flag:
            sol = optimize.root(
                self.Geske_eqs, np.array([10_000_000, 0.05]), method="df-sane"
            )
            converge_flag = sol.success

        if not converge_flag:
            sol = optimize.root(
                self.Geske_eqs, np.array([100_000_000_000, 1000]), method="df-sane"
            )
            converge_flag = sol.success

        return sol.x

    def probability_of_default(self) -> np.ndarray:
        """
        probability of default for two periods

        Returns:
            np.ndarray: 2 periods PD
        """
        a_para = self.Geske_solver

        h12 = (
            np.log(a_para[0])
            - np.log(self.A1_bar)
            + (self.r - a_para[1] ** 2 / 2) * (self.t1 - self.t)
        ) / (a_para[1] * np.sqrt(self.t1 - self.t))

        h22 = (
            np.log(a_para[0])
            - np.log(self.k2)
            + (self.r - a_para[1] ** 2 / 2) * (self.t2 - self.t)
        ) / (a_para[1] * np.sqrt(self.t2 - self.t))

        q1 = stats.norm.cdf(h12)
        p1 = 1 - q1
        q2 = stats.multivariate_normal.cdf([h12, h22], mean=np.zeros(2), cov=self.rho)
        p2 = (q1 - q2) / q1
        return np.array([p1, p2])
