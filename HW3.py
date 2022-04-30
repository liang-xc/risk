import numpy as np
import pandas as pd

from cdo import CDXCDO
from cds import CDS
from EuropeanOption import EuropeanOption


def main():
    portfolio_component = [
        "AIG",
        "C",
        "CAT",
        "COST",
        "GS",
        "INTC",
        "LMT",
        "MSFT",
        "PFE",
        "SONY",
    ]
    cds_list = []
    for t in portfolio_component:
        try:
            df = pd.read_csv(f"./CDS_data/{t}.csv")
        except FileNotFoundError:
            continue

        maturities = np.array([1, 2])
        mid = (df["Bid"] + df["Ask"]) / 2
        mid = mid.to_numpy()
        mid = mid * 1e-4
        cds_list.append(CDS(t, maturities, mid))

    cdxcdo = CDXCDO(cds_list)
    print(f"conditional PD: \t {cdxcdo.conditional_pd()}")

    print("loss function according to Vasicek:")
    for i in range(len(portfolio_component) + 1):
        print(f"{i}: \t {cdxcdo.loss(i)}")

    senior = (
        EuropeanOption(100, 375, 1, 0.3, 0).price()
        - EuropeanOption(100, 187.5, 1, 0.3, 0).price()
    )
    mezzanine1 = (
        EuropeanOption(100, 187.5, 1, 0.3, 0).price()
        - EuropeanOption(100, 125, 1, 0.3, 0).price()
    )
    mezzanine2 = (
        EuropeanOption(100, 125, 1, 0.3, 0).price()
        - EuropeanOption(100, 87.5, 1, 0.3, 0).price()
    )
    mezzanine3 = (
        EuropeanOption(100, 87.5, 1, 0.3, 0).price()
        - EuropeanOption(100, 37.5, 1, 0.3, 0).price()
    )
    equity = EuropeanOption(100, 87.5, 1, 0.3, 0).price() - 100

    print("expected tranche losses based on BS: ")
    print(f"{senior=}")
    print(f"{mezzanine1=}")
    print(f"{mezzanine2=}")
    print(f"{mezzanine3=}")
    print(f"{equity=}")


if __name__ == "__main__":
    main()
