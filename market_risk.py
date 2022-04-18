import datetime
from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd
import sklearn
import yfinance
from scipy import optimize, stats


# VaR calculations
def generate_return_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    generate two columns for return and PnL

    Args:
        df (pd.DataFrame): stock data dataframe

    Returns:
        pd.DataFrame: dataframe with two cols added
    """
    df["Return"] = df["Close"] / df["Close"].shift(1) - 1
    df["PnL"] = df["Close"] - df["Close"].shift(1)
    df.dropna(inplace=True)
    return df


def VaR(arr: np.ndarray) -> float:
    """
    95% parametric VaR for a single stock

    Args:
        arr (np.ndarray): return or PnL series

    Returns:
        float: VaR
    """
    return np.abs(arr.mean() - stats.norm.ppf(0.95) * arr.std())


def portfolio_VaR(arr: np.ndarray, weight: np.ndarray) -> float:
    """
    95% parametric VaR for a stock portfolio

    Args:
        arr (np.ndarray): 2D array for stock return or PnL series
        weight (np.ndarray): weight of each component

    Returns:
        float: portfolio VaR
    """
    assert len(arr) == len(weight), "input array and weight array must be of same length."
    cov_matrix = np.cov(arr)
    portfolio_variance = np.dot(np.dot(weight, cov_matrix), weight.T)
    # portfolio_mean = np.dot(weight, np.array(list(map(np.mean, arr))))
    return stats.norm.ppf(0.95) * np.sqrt(portfolio_variance)


def pca_VaR(arr: np.ndarray, weight: np.ndarray, n: int) -> float:
    """
    95% VaR for a stock portfolio with Principal Component Analysis (PCA)

    Args:
        arr (np.ndarray): 2D array for stock return or PnL series
        weight (np.ndarray): weight of each stock
        n (int): number of PCA components

    Returns:
        float: VaR
    """
    pca = sklearn.decomposition.PCA(n_components=n)
    pca_data = pca.fit_transform(arr.T)
    pca_loadings: np.ndarray = pca.components_
    pca_cov = np.cov(pca_loadings)
    cov_matrix = ((pca_loadings.T).dot(pca_cov)).dot(pca_loadings)
    portfolio_variance = np.dot(np.dot(weight, cov_matrix), weight.T)
    return stats.norm.ppf(0.95) * np.sqrt(portfolio_variance)


def component_VaR(arr: np.ndarray, weight: np.ndarray) -> Generator[float, None, None]:
    """
    component VaR for each component of a stock portfolio

    Args:
        arr (np.ndarray): 2D array for stock return or PnL series
        weight (np.ndarray): weight for each stock

    Yields:
        Generator[float]: component VaR
    """

    def percentage_allocation() -> list[float]:
        cov_matrix = np.cov(arr)
        temp = []
        for w, cov in zip(weight, cov_matrix):
            temp.append(np.dot(weight, cov) * w)
        alloc = list(map(lambda x: x / sum(temp), temp))
        return alloc

    alloc = percentage_allocation()
    port_var = portfolio_VaR(arr, weight)
    for a in alloc:
        yield a * port_var


def marginal_VaR(arr: np.ndarray, weight: np.ndarray) -> Generator[float, None, None]:
    """
    marginal VaR for each component of a stock portfolio

    Args:
        arr (np.ndarray): 2D array for stock return or PnL series
        weight (np.ndarray): weight for each stock

    Yields:
        Generator[float]: marginal VaR
    """
    port_std = portfolio_VaR(arr, weight) / stats.norm.ppf(0.95)
    cov_matrix = np.cov(arr)
    for idx, (sigma, w) in enumerate(zip(cov_matrix, weight)):
        w_minor = np.delete(weight, idx, 0)
        cov_minor = np.delete(cov_matrix, idx, 0)
        cov_minor = np.delete(cov_minor, idx, 1)
        yi = port_std - np.sqrt((w_minor.dot(cov_minor)).dot(w_minor.T))
        yield stats.norm.ppf(0.95) * yi


def decay_VaR(arr: np.ndarray, theta: float) -> float:
    """
    decay VaR for a stock

    Args:
        arr (np.ndarray): stock return or PnL series
        theta (float): decay rate

    Returns:
        float: decay VaR
    """

    def sigma(t: int) -> float:
        variance = 0
        r_bar = arr[0:t].mean()
        for i in range(len(arr) - 1):
            variance += (theta**i) * ((arr[t - i - 1] - r_bar) ** 2)

        variance *= (1.0 - theta) / (1.0 - theta ** len(arr))
        return np.sqrt(variance)

    decay_sigma = sigma(len(arr))
    parametric_VaR = abs(arr.mean() - stats.norm.ppf(0.95) * decay_sigma)
    return parametric_VaR


@dataclass
class EVT:

    """
    compute VaR under Extreme Value Theory

    Attributes:
        hist_data: historical data
    """

    hist_data: np.ndarray

    def __post_init__(self) -> None:
        """
        post_init attributes

        Attributes:
            u: 5% VaR as -u
            tail_obs: tail observations
            nu: number of observations in the tail
        """
        self.u: float = stats.norm.ppf(0.95) * self.hist_data.std()
        self.tail_obs: np.ndarray = -self.hist_data[self.hist_data < -self.u]
        self.nu: int = len(self.tail_obs)

    def parameter_estimation(self) -> np.ndarray:
        """
        estimate the two parameters of the generalized Pareto distribution
        using maximum likelihood estimation

        Returns:
            np.ndarray: two parameters
                xi
                beta
        """

        def likelihood(para: np.ndarray) -> float:
            dist = stats.genpareto(c=para[0], scale=para[1])
            return -np.sum([np.log(dist.pdf(t)) for t in self.tail_obs])

        mle = optimize.minimize(likelihood, np.array([100, 100]), method="Nelder-Mead")
        return mle.x

    def u_star(self) -> float:
        """
        same risk as u (5%) under EVT

        Returns:
            float: risk value under EVT
        """
        mle_para = self.parameter_estimation()
        xi, beta = mle_para[0], mle_para[1]
        return self.u + beta / xi * ((len(self.hist_data) / self.nu * 0.05) ** (-xi) - 1)


def fetch_option_data(ticker: str) -> pd.DataFrame:
    """
    fetch option data from Yahoo Finance

    Args:
        ticker (str): ticker of the underlying security

    Returns:
        pd.DataFrame: all option data
    """
    result = pd.DataFrame()
    t = yfinance.Ticker(ticker)
    expirations = t.options

    for e in expirations:
        opt = t.option_chain(e)
        opt_df = pd.concat([opt.calls, opt.puts])
        opt_df["ExpirationDate"] = e
        result = pd.concat([result, opt_df], ignore_index=True)

    # Dummy variable column if the option is a CALL
    def dummy(x: str) -> int:
        if "C" in x:
            return 1
        else:
            return 0

    result["is_call"] = result["contractSymbol"].str[4:].apply(dummy)

    result["ExpirationDate"] = pd.to_datetime(result["ExpirationDate"])

    result["Maturity"] = (
        result["ExpirationDate"] - datetime.datetime.today()
    ) / np.timedelta64(1, "Y")

    result[["bid", "ask", "strike"]] = result[["bid", "ask", "strike"]].apply(
        pd.to_numeric
    )
    result["Mid"] = (result["bid"] + result["ask"]) / 2

    # Drop unnecessary columns
    result.drop(
        columns=[
            "contractSize",
            "currency",
            "change",
            "percentChange",
            "lastTradeDate",
            "lastPrice",
        ],
        inplace=True,
    )

    return result
