import math

import numpy as np
from scipy.optimize import root


class EuropeanOption:
    rf = 0  # risk-free rate

    def __init__(
        self,
        spot: float,
        strike: float,
        maturity: float,
        vol: float,
        div: float,
        mult: float = 1,
        call_flag: int = 1,
    ):
        self.Spot = spot
        self.Strike = strike

        if maturity < 0:
            raise ValueError("maturity has to be a positive number.")
        else:
            self.Maturity = maturity

        self.Volatility = vol
        self.DividendYield = div
        self.Multiplier = mult

        if call_flag == 1 or call_flag == 0:
            self.CallFlag = call_flag
        else:
            raise ValueError(
                f"{call_flag} is not a valid input as CallFlag, should be either 0 or 1."
            )

        self.__d1 = (
            np.log(spot / strike) + (self.rf - div + vol**2 / 2) * maturity
        ) / (vol * np.sqrt(maturity))
        self.__d2 = self.__d1 - vol * np.sqrt(maturity)

    def __call__(self, st: float) -> float:
        """
        calculate the option premium for a given stock price
        st: given spot price, potentially different from spot
        return: value of option premium, float
        """
        if self.CallFlag == 1:

            return st * np.exp(-self.DividendYield * self.Maturity) * self.N(
                self.__d1
            ) - self.Strike * np.exp(-self.rf * self.Maturity) * self.N(self.__d2)
        else:
            return self.Strike * np.exp(-self.rf * self.Maturity) * self.N(
                -self.__d2
            ) - st * np.exp(-self.DividendYield * self.Maturity) * self.N(-self.__d1)

    def __str__(self):
        if self.CallFlag == 1:
            return f"{self.__maturity_to_str()} {self.Strike}-strike calls"
        else:
            return f"{self.__maturity_to_str()} {self.Strike}-strike puts"

    def __imul__(self, mult: float):
        self.Multiplier = self.Multiplier * mult
        self.Strike /= mult
        self.Spot /= mult
        return self

    def N(self, x) -> float:
        return (1 + math.erf(x / np.sqrt(2))) / 2

    def dN(self, x) -> float:
        return 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)

    @staticmethod
    def set_risk_free_rate(r: float) -> None:
        EuropeanOption.rf = r

    def price(self) -> float:
        """
        return: current price of option using BSM, float
        """
        return self(self.Spot) * self.Multiplier

    def delta(self) -> float:
        """
        return: delta of option
        """
        if self.CallFlag == 1:
            return self.N(self.__d1)
        else:
            return self.N(self.__d1) - 1

    def gamma(self) -> float:
        """
        return: gamma of option
        """
        return self.dN(self.__d1) / (self.Spot * self.Volatility * np.sqrt(self.Maturity))

    def vega(self) -> float:
        """
        return: vega of option
        """
        return self.Spot * self.dN(self.__d1) * np.sqrt(self.Maturity)

    def theta(self) -> float:
        """
        return: theta of option
        """
        if self.CallFlag == 1:
            return -(self.Spot * self.dN(self.__d1) * self.Volatility) / (
                2 * np.sqrt(self.Maturity)
            ) - self.rf * self.Strike * np.exp(-self.rf * self.Maturity) * self.N(
                self.__d2
            )
        else:
            return -(self.Spot * self.dN(self.__d1) * self.Volatility) / (
                2 * np.sqrt(self.Maturity)
            ) + self.rf * self.Strike * np.exp(-self.rf * self.Maturity) * self.N(
                -self.__d2
            )

    def __maturity_to_str(self) -> str:
        """
        return maturity in string format
        """
        if self.Maturity >= 1:
            return f"{self.Maturity}-years"
        else:
            return f"{self.Maturity*12}-months"


def implied_vol(
    s: float, k: float, t: float, div: float, rf, mkt_price: float, call_flag: int
) -> float:
    """
    Calculate the implied volatility of a option given its market price
    """

    def err(vol: float):
        option = EuropeanOption(s, k, t, vol, div, call_flag=call_flag)
        option.set_risk_free_rate(rf)
        return option.price() - mkt_price

    iv = root(err, 0.01).x
    return iv
