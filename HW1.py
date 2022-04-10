import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA

import market_risk


def pca_var(arr: np.ndarray, weight: np.ndarray, n: int) -> float:
    """
    calculate PCA VaR of portfolio.
    n is the number of Principle Components
    """
    pca = PCA(n_components=n)
    pca_data = pca.fit_transform(arr.T)
    pca_loadings: np.ndarray = pca.components_
    pca_cov = np.cov(pca_loadings)
    cov_matrix = ((pca_loadings.T).dot(pca_cov)).dot(pca_loadings)
    portfolio_variance = np.dot(np.dot(weight, cov_matrix), weight.T)
    return stats.norm.ppf(0.95) * np.sqrt(portfolio_variance)


def main():
    portfolio_component = [
        "AIG",
        "C",
        "CAT",
        "COST",
        "INTC",
        "LMT",
        "MSFT",
        "MUFG",
        "PFE",
        "SONY",
    ]
    df_list = []
    for pc in portfolio_component:
        temp_df = market_risk.generate_return_pnl(pd.read_csv(f"./HW1_data/{pc}.csv"))
        df_list.append(temp_df)

    print(f"portfolio components: {portfolio_component}")
    print(
        "portfolio components are of same weight and assigned $100,000 for each component"
    )
    #  portfolio components are of same weight
    weight = np.zeros(10) + 1
    #  Total value of portfolio to be 1,000,000, so 100,000 for each component
    shares = []
    for df in df_list:
        shares.append(100000 / df["Close"][1])
    shares = np.array(shares)

    port_returns = []
    port_pnls = []
    for df, s in zip(df_list, shares):
        port_returns.append(df["Return"])
        port_pnls.append(df["PnL"] * s)
    port_returns = np.array(port_returns)
    port_pnls = np.array(port_pnls)

    #  historical VaR of portfolio
    returns_sum = np.sum(port_returns, axis=0)
    h_var = np.percentile(returns_sum, 5)
    print(f"Historical VaR of portfolio return is {h_var*100}%.")
    #  with decay=0.95
    decay_returns = returns_sum.copy()
    for idx, r in enumerate(decay_returns):
        decay_times = len(decay_returns) - idx
        r = r * (0.95**decay_times)

    hd_var = np.percentile(decay_returns, 5)
    print(f"Historical VaR of portfolio return with decay=0.95 is {hd_var*100}%.")
    #  parametric VaR of portfolio
    r_para_var = market_risk.portfolio_VaR(port_returns, weight)
    pnl_para_var = market_risk.portfolio_VaR(port_pnls, shares)
    print(f"Parametric VaR of portfolio return is {r_para_var*100}%.")
    print(f"Parametric Var of portfolio PnL is {pnl_para_var}.")
    #  with decay=0.95
    d_var = market_risk.decay_VaR(np.sum(port_returns, axis=0), 0.95)
    print(f"95% VaR for portfolio return with decay=0.95 is {d_var*100}%")

    #  factor model VaR of portfolio
    f_var = pca_var(port_returns, weight, 2)
    print(f"Factor-model VaR of portfolio return is {f_var*100}%")


if __name__ == "__main__":
    main()
