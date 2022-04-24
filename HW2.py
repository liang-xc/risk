import warnings

import numpy as np
import pandas as pd

import credit_risk

warnings.filterwarnings("ignore", category=RuntimeWarning)


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

    debt_info = pd.read_csv(
        "./HW2_data/debt_info.csv", index_col=["Ticker"], thousands=","
    )

    # KMV
    for t in portfolio_component:
        df = pd.read_csv(f"./HW1_data/{t}.csv")
        r = df["Close"].pct_change().dropna()
        e_sigma = np.std(r)
        kmv = credit_risk.KMV(
            debt_info["STD"][t], debt_info["LTD"][t], debt_info["Et"][t], e_sigma
        )
        print(f"{t} default probability according to KMV:")
        print(kmv.probability_of_default())
        print(f"{t} recovery rate according to KMV:")
        print(kmv.expected_recovery())

    # Geske
    for t in portfolio_component:
        df = pd.read_csv(f"./HW1_data/{t}.csv")
        r = df["Close"].pct_change().dropna()
        e_sigma = np.std(r)
        geske = credit_risk.Geske(
            0,
            debt_info["STD"][t],
            1,
            debt_info["LTD"][t],
            10,
            debt_info["Et"][t],
            e_sigma,
        )
        print(f"{t} default probability according to Geske:")
        print(geske.probability_of_default())


if __name__ == "__main__":
    main()
