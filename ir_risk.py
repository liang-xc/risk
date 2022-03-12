from dataclasses import dataclass
from typing import Generator

import numpy as np


def discount_factor_to_yield(discount: float, t: float) -> float:
    """
    convert discount factor to yield
    """
    return -np.log(discount) / t


@dataclass
class Bond:
    """A class to represent a bond"""

    coupon: float
    coupon_freq: int
    maturity: float
    par: float = 100.0

    def coupon_time(self) -> np.ndarray:
        """return time from 0 to coupon payment in years"""
        to_maturity = self.maturity
        year_between_coupons = 1 / self.coupon_freq
        result = []

        while to_maturity > 0:
            if to_maturity < year_between_coupons:
                break
            else:
                result.append(to_maturity)
                to_maturity -= year_between_coupons

        return result

    def g_ik(self, y1: "CIR", y2: "CIR", y1_flag=True) -> Generator:
        for t in self.coupon_time():
            y1t = modify_tao(t, y1)
            y2t = modify_tao(t, y2)
            p = y1t.pt() * y2t.pt()
            if y1_flag:
                yield -p * y1.bt()
            else:
                yield -p * y2.bt()


@dataclass
class Vasicek:
    """A class to build up a Vasicek model"""

    tao: float
    alpha: float
    mu: float
    sigma: float
    lamb: float
    rt: float

    def bt(self) -> float:
        return (1 - np.exp(-self.alpha * self.tao)) / self.alpha

    def nln_at(self) -> float:
        return (
            self.mu
            - self.sigma * self.lamb / self.alpha
            - self.sigma**2 / (2 * self.alpha**2)
        ) * (self.tao - self.bt()) + (self.sigma**2 * self.bt() ** 2) / (4 * self.alpha)

    def at(self) -> float:
        return np.exp(-self.nln_at())

    def pt(self) -> float:
        return self.at() * np.exp(-self.rt * self.bt())


@dataclass
class CIR:
    """A class to build up a Cox-Ingersoll-Ross model"""

    tao: float
    alpha: float
    mu: float
    sigma: float
    lamb: float
    rt: float

    def gamma(self) -> float:
        return np.sqrt((self.alpha + self.lamb) ** 2 + 2 * self.sigma**2)

    def bt(self) -> float:
        return (
            2
            * (np.exp(self.gamma() * self.tao) - 1)
            / (
                (self.alpha + self.lamb + self.gamma())
                * (np.exp(self.gamma() * self.tao) - 1)
                + 2 * self.gamma()
            )
        )

    def at(self) -> float:
        return (
            2
            * self.gamma()
            * np.exp((self.alpha + self.lamb + self.gamma()) * self.tao / 2)
            / (
                (self.alpha + self.lamb + self.gamma())
                * (np.exp(self.gamma() * self.tao) - 1)
                + 2 * self.gamma()
            )
        ) ** (2 * self.alpha * self.mu / self.sigma**2)

    def pt(self) -> float:
        return self.at() * np.exp(-self.rt * self.bt())


def modify_tao(tao: float, model: CIR) -> CIR:
    """return a new CIR model with different tao"""
    return CIR(tao, model.alpha, model.mu, model.sigma, model.lamb, model.rt)


def var_dP(b: Bond, y1: CIR, y2: CIR) -> float:
    """
    calculates the variance of dP based on two-factor model
    r = y1 + y2

    bond: a Bond object
    y1: first factor modeled by CIR
    y2: second factor modeled by CIR

    return: variance of dP based on model
    """
    y1 = modify_tao(b.maturity, y1)
    y2 = modify_tao(b.maturity, y2)
    p = y1.pt() * y2.pt()
    v1 = np.sqrt(y1.sigma * y1.rt)
    v2 = np.sqrt(y2.sigma * y2.rt)

    if b.coupon == 0:  # Zero coupon bond
        return p**2 * ((y1.bt() * v1) ** 2 + (y2.bt() * v2) ** 2)
    else:  # coupon bond
        a1 = b.coupon / b.coupon_freq * sum(b.g_ik(y1, y2)) + next(b.g_ik(y1, y2))
        a2 = b.coupon / b.coupon_freq * sum(b.g_ik(y1, y2, False)) + next(
            b.g_ik(y1, y2, False)
        )
        return (a1 * v1) ** 2 + (a2 * v2) ** 2


def cov_dP_dPi(z: Bond, c: Bond, y1: CIR, y2: CIR) -> float:
    """
    calculates the covariance between dP(zero coupon bond) and dPi(coupon bond)
    based on two-factor model
    r = y1 + y2

    z: zero coupon bond
    c: coupon bond
    y1: first factor modeled by CIR
    y2: second factor modeled by CIR

    return: covariance of dP and dPi based on model
    """
    y1 = modify_tao(z.maturity, y1)
    y2 = modify_tao(z.maturity, y2)
    p = y1.pt() * y2.pt()
    v1 = np.sqrt(y1.sigma * y1.rt)
    v2 = np.sqrt(y2.sigma * y2.rt)

    a1 = c.coupon / c.coupon_freq * sum(c.g_ik(y1, y2)) + next(c.g_ik(y1, y2))
    b1 = a1 * -y1.bt() * p
    a2 = c.coupon / c.coupon_freq * sum(c.g_ik(y1, y2, False)) + next(
        c.g_ik(y1, y2, False)
    )
    b2 = a2 * -y2.bt() * p
    return (b1 * v1) ** 2 + (b2 * v2) ** 2


def discount_factor_to_yield_curve(
    discount_factors: dict[float:float],
) -> dict[float:float]:
    """convert a dict of discount factors to a yield curve"""
    return {t: discount_factor_to_yield(d, t) for t, d in discount_factors.items()}


def yield_curve_to_discount_factors(yield_curve: dict[float:float]) -> dict[float:float]:
    """convert a yield curve to a dict of discount factors"""
    return {t: np.exp(-y * t) for t, y in yield_curve.items()}


def yield_to_forward(yield_curve: dict[float:float]) -> dict[float:float]:
    """
    convert yield curve to forward curve
    continuous compounding
    """
    # set first element of  forward curve
    first = next(iter(yield_curve.items()))
    forward_curve = {first[0]: first[1]}

    for t in yield_curve:
        if t in forward_curve.keys():
            pass
        else:
            forward_curve[t] = yield_curve[t] * t - yield_curve[t - 1] * (t - 1)

    return forward_curve


def forward_curve(tao: float, discount_factors: dict[float:float]) -> dict[float:float]:
    """
    convert discount factors to a forward curve starting from time tao
    """
    yield_curve = discount_factor_to_yield_curve(discount_factors)
    yield_maturity = list(yield_curve.keys())
    forward_maturity = [m - tao for m in yield_maturity if m > tao]
    forward_rates = [
        (yield_curve[tao + m] * (tao + m) - yield_curve[tao] * tao) / m
        for m in forward_maturity
    ]
    return dict(zip(forward_maturity, forward_rates))


def swap_rate(tao: float, discount_factors: dict[float:float]) -> float:
    """
    compute swap rate given time t and discount factors (could be transferred from yield curve)
    """
    # construct yield curve from discount factors
    yield_curve = discount_factor_to_yield_curve(discount_factors)

    # calculate forward rates
    forward_curve = yield_to_forward(yield_curve)

    nominator = 0
    denominator = 0
    for t in range(1, tao + 1):
        nominator += discount_factors[t] * forward_curve[t]
        denominator += discount_factors[t]

    return nominator / denominator
