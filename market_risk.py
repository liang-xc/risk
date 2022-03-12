from typing import Iterator

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn


# VaR calculations
def generate_return_pnl(df: pd.DataFrame) -> pd.DataFrame:
    df["Return"] = df["Close"] / df["Close"].shift(1) - 1
    df["PnL"] = df["Close"] - df["Close"].shift(1)
    df.dropna(inplace=True)
    return df


def VaR(arr: np.ndarray) -> float:
    """calculate parametric VaR for a single stock"""
    return np.abs(arr.mean() - stats.norm.ppf(0.95) * arr.std())


def portfolio_VaR(arr: np.ndarray, weight: np.ndarray) -> float:
    assert len(arr) == len(weight), "input array and weight array must be of same length."
    cov_matrix = np.cov(arr)
    portfolio_variance = np.dot(np.dot(weight, cov_matrix), weight.T)
    # portfolio_mean = np.dot(weight, np.array(list(map(np.mean, arr))))
    return stats.norm.ppf(0.95) * np.sqrt(portfolio_variance)


def pca_VaR(arr: np.ndarray, weight: np.ndarray, n: int) -> float:
    """
    calculate PCA VaR of portfolio.
    n is the number of Principle Components
    """
    pca = sklearn.decomposition.PCA(n_components=n)
    pca_data = pca.fit_transform(arr.T)
    pca_loadings: np.ndarray = pca.components_
    pca_cov = np.cov(pca_loadings)
    cov_matrix = ((pca_loadings.T).dot(pca_cov)).dot(pca_loadings)
    portfolio_variance = np.dot(np.dot(weight, cov_matrix), weight.T)
    return stats.norm.ppf(0.95) * np.sqrt(portfolio_variance)


def percentage_allocation(arr: np.ndarray, weight: np.ndarray) -> list[float]:
    cov_matrix = np.cov(arr)
    temp = []
    for w, cov in zip(weight, cov_matrix):
        temp.append(np.dot(weight, cov) * w)
    alloc = list(map(lambda x: x / sum(temp), temp))
    return alloc


def component_VaR(arr: np.ndarray, weight: np.ndarray) -> Iterator[float]:
    alloc = percentage_allocation(arr, weight)
    port_var = portfolio_VaR(arr, weight)
    for a in alloc:
        yield a * port_var


def marginal_VaR(arr: np.ndarray, weight: np.ndarray) -> Iterator[float]:
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
    theta: decay rate
    return VaR value
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


# Fixed Income Risk Management
def discount_factor_to_yield(discount: float, t: float) -> float:
    """
    convert discount factor to yield
    """
    return -np.log(discount) / t


def vasicek(
    tao: float, alpha: float, mu: float, sigma: float, lamb: float, rt: float
) -> float:
    """
    calculate interest rates based on Vasicek model
    return the risk-free discount factor
    tao = T - t
    """
    bt = (1 - np.exp(-alpha * tao)) / alpha

    nln_at = (mu - sigma * lamb / alpha - sigma**2 / (2 * alpha**2)) * (tao - bt) + (
        sigma**2 * bt**2
    ) / (4 * alpha)

    at = np.exp(-nln_at)
    pt = at * np.exp(-rt * bt)
    return pt


def cir(
    tao: float, alpha: float, mu: float, sigma: float, lamb: float, rt: float
) -> float:
    """
    calculate interest rates based on CIR model
    return the risk-free discount factor
    tao = T - t
    """
    gamma = np.sqrt((alpha + lamb) ** 2 + 2 * sigma**2)
    bt = (
        2
        * (np.exp(gamma * tao) - 1)
        / ((alpha + lamb + gamma) * (np.exp(gamma * tao) - 1) + 2 * gamma)
    )

    at = (
        2
        * gamma
        * np.exp((alpha + lamb + gamma) * tao / 2)
        / ((alpha + lamb + gamma) * (np.exp(gamma * tao) - 1) + 2 * gamma)
    ) ** (2 * alpha * mu / sigma**2)

    pt = at * np.exp(-rt * bt)
    return pt
