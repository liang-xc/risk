import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance

import credit_risk
import ir_risk
import market_risk
from cds import CDS
from EuropeanOption import EuropeanOption


def mini1():
    # 1a
    inteldf = market_risk.generate_return_pnl(pd.read_csv("./HW1_data/INTC.csv"))
    n = 1000000 / inteldf["Close"][1]

    re_hist = np.percentile(inteldf["Return"], 5)
    re_para = market_risk.VaR(inteldf["Return"])
    pnl_hist = np.percentile(inteldf["PnL"], 5)
    pnl_para = market_risk.VaR(inteldf["PnL"])
    with open("mini.txt", "w") as fout:
        fout.write("1a. INTC stock VaR calculation: \n")
        fout.write(f"95% historical VaR for INTC Return is {re_hist*100}%\n")
        fout.write(f"95% parametric VaR for INTC Return is {re_para*100}%\n")
        fout.write(f"95% historical VaR for INTC PnL is {pnl_hist*n}\n")
        fout.write(f"95% parametric VaR for INTC PnL is {pnl_para*n}\n")
        fout.write("\n")

    # 1b
    cdf = market_risk.generate_return_pnl(pd.read_csv("./HW1_data/C.csv"))
    w = np.array([1, 1])

    intel_shares = 500000 / inteldf["Close"][1]
    c_shares = 500000 / cdf["Close"][1]
    n = np.array([intel_shares, c_shares])

    portfolio_return = inteldf["Return"] + cdf["Return"]
    portfolio_pnl = inteldf["PnL"] * intel_shares + cdf["PnL"] * c_shares

    hist_re = np.percentile(portfolio_return, 5)
    hist_pnl = np.percentile(portfolio_pnl, 5)

    port_re = market_risk.portfolio_VaR(np.array([inteldf["Return"], cdf["Return"]]), w)
    port_pnl = market_risk.portfolio_VaR(np.array([inteldf["PnL"], cdf["PnL"]]), n)
    with open("mini.txt", "a") as fout:
        fout.write("1b. INTC and C portfolio VaR calculation:\n")
        fout.write(f"95% historical VaR for portfolio return is {hist_re*100}%\n")
        fout.write(f"95% parametric VaR for portfolio return is {port_re*100}%\n")
        fout.write(f"95% historical VaR for portfolio PnL is {hist_pnl}\n")
        fout.write(f"95% parametric VaR for portfolio PnL is {port_pnl}\n")
        fout.write("\n")

    # 1c
    inteldf["Date"] = pd.to_datetime(inteldf["Date"])
    maturity = pd.to_datetime("2022-01-04")
    inteldf["Maturity"] = (maturity - inteldf["Date"]).dt.days / 365

    call = []
    cport_dV = []
    for i in range(1, len(inteldf)):
        option = EuropeanOption(
            inteldf["Close"][i], 50, inteldf["Maturity"][i], inteldf["Close"].std(), 0
        )
        call.append(option.price())
        cport_dV.append(inteldf["PnL"] * (1 + option.delta()))

    delta = EuropeanOption(
        inteldf["Close"][1], 50, inteldf["Maturity"][i], inteldf["Close"].std(), 0
    ).delta()
    cport_std = inteldf["Close"].std() * 1 * (1 + delta)
    cport_pVaR = stats.norm.ppf(0.95) * cport_std

    cport_dV = np.array(cport_dV)
    cport_dV = cport_dV[np.logical_not(np.isnan(cport_dV))]
    cpot_hVaR = np.percentile(cport_dV, 5)

    with open("mini.txt", "a") as fout:
        fout.write("1c. INTC and INTC Call option portfolio VaR calculation:\n")
        fout.write(f"95% historical VaR for portfolio is {cpot_hVaR}\n")
        fout.write(f"95% parametric VaR for portfolio is {cport_pVaR}\n")
        fout.write("\n")


def mini2():
    # 2a
    cdf = market_risk.generate_return_pnl(pd.read_csv("./HW1_data/C.csv"))
    inteldf = market_risk.generate_return_pnl(pd.read_csv("./HW1_data/INTC.csv"))
    msftdf = market_risk.generate_return_pnl(pd.read_csv("./HW1_data/MSFT.csv"))
    port_component = ["C", "INTC", "MSFT"]
    w = np.array([100, 100, 100])
    port_comp_var = market_risk.component_VaR(
        np.array([cdf["PnL"], inteldf["PnL"], msftdf["PnL"]]), w
    )

    with open("mini.txt", "a") as fout:
        fout.write("2a. Component VaR and Marginal VaR calculation:\n")
        for s, var in zip(port_component, port_comp_var):
            fout.write(f"Component VaR of {s} is {var}.\n")

    port_marginal_var = market_risk.marginal_VaR(
        np.array([cdf["PnL"], inteldf["PnL"], msftdf["PnL"]]), w
    )
    with open("mini.txt", "a") as fout:
        for s, var in zip(port_component, port_marginal_var):
            fout.write(f"Marginal VaR of {s} is {var}.\n")
        fout.write("\n")

    # 2b
    re_var_decay = market_risk.decay_VaR(inteldf["Return"], 0.95)
    with open("mini.txt", "a") as fout:
        fout.write("2b. decay VaR calculation:\n")
        fout.write(f"95% VaR for INTC Return with decay=0.95 is {re_var_decay*100}%\n")
        fout.write("\n")


def mini3():
    # Replicate Vasicek
    vasicek_df = pd.DataFrame(index=np.arange(1, 31))
    v_discount = np.array(
        [
            ir_risk.Vasicek(t, 0.2456, 0.0648, 0.0289, -0.2718, 0.06).pt()
            for t in range(1, 31)
        ]
    )
    v_yield = np.array(
        [ir_risk.discount_factor_to_yield(v, t) for v, t in zip(v_discount, range(1, 31))]
    )
    vasicek_df["DiscountFactor"] = v_discount
    vasicek_df["Yield"] = v_yield
    with open("mini.txt", "a") as fout:
        fout.write("3a:\n")
        fout.write("Replicate Vasicek:\n")
        fout.write(vasicek_df.to_string())
        fout.write("\n")

    # Replicate CIR
    cir_df = pd.DataFrame(index=np.arange(1, 31))
    cir_discount = np.array(
        [ir_risk.CIR(t, 0.2456, 0.0648, 0.14998, -0.129, 0.06).pt() for t in range(1, 31)]
    )
    cir_yield = np.array(
        [
            ir_risk.discount_factor_to_yield(c, t)
            for c, t in zip(cir_discount, range(1, 31))
        ]
    )
    cir_df["DiscountFactor"] = cir_discount
    cir_df["Yield"] = cir_yield
    with open("mini.txt", "a") as fout:
        fout.write("Replicate CIR:\n")
        fout.write(cir_df.to_string())
        fout.write("\n")

    # compute factor based VaR of a coupon bond and a zero coupon bond
    zero_bond = ir_risk.Bond(0, 1, 2)
    coupon_bond = ir_risk.Bond(4, 2, 4)
    y1 = ir_risk.CIR(1, 1.8341, 0.051480, 0.154300, -0.125300, 0.01)
    y2 = ir_risk.CIR(1, 0.005212, 0.030830, 0.066890, -0.066500, 0.02)
    variance_covariance_matrix = np.array(
        [
            [
                ir_risk.var_dP(zero_bond, y1, y2),
                ir_risk.cov_dP_dPi(zero_bond, coupon_bond, y1, y2),
            ],
            [
                ir_risk.cov_dP_dPi(zero_bond, coupon_bond, y1, y2),
                ir_risk.var_dP(coupon_bond, y1, y2),
            ],
        ]
    )
    n = np.array([100, 100]).T
    var = stats.norm.ppf(0.95) * np.sqrt((n.dot(variance_covariance_matrix)).dot(n.T))
    with open("mini.txt", "a") as fout:
        fout.write("3b.\n")
        fout.write(
            "Bond portfolio of zero coupon bond with 2 years to maturity and coupon bond with 4 years to maturity 4 coupon and semi annual coupon payment.\n"
        )
        fout.write("Both bond are 100.\n")
        fout.write(f"VaR of the bond portfolio is {var}.\n")
        fout.write("\n")


def mini4():
    v_discount = {
        t: ir_risk.Vasicek(t, 0.2456, 0.0648, 0.0289, -0.2718, 0.06).pt()
        for t in range(1, 31)
    }
    v_yield = ir_risk.discount_factor_to_yield_curve(v_discount)
    v_forward = ir_risk.yield_to_forward(v_yield)

    with open("mini.txt", "a") as fout:
        # 4a. compute swap rate for 2~30 yrs
        fout.write("4a. swap rate for 2~30 yrs\n")
        for i in range(2, 31):
            sr = ir_risk.swap_rate(i, v_discount)
            fout.write(f"{i}:\t {sr: .4f}\n")

        fout.write("\n")

        # 4b. 1 year forward rates 1 year from now till 29 years from now
        fout.write("4b. 1 year forward rates 1 year from now till 29 years from now\n")
        for i, f in v_forward.items():
            if i == 1:
                pass
            else:
                fout.write(f"{i - 1}:\t {f:.4f}\n")

        fout.write("\n")

        # 4c. forward curve of next year
        forward_curve_next = ir_risk.forward_curve(1, v_discount)
        fout.write("4c. forward curve of next year\n")
        for i, f in forward_curve_next.items():
            fout.write(f"{i}:\t {f:.4f}\n")

        sr_1_9 = ir_risk.swap_rate(
            9, ir_risk.yield_curve_to_discount_factors(forward_curve_next)
        )
        fout.write(f"9 yr swap rate next year:\t {sr_1_9:.4f}\n")
        fout.write(
            f"10 yr swap rate this year: \t {ir_risk.swap_rate(10, v_discount):.4f}\n"
        )
        fout.write(
            "9 yr swap rate next year is higher. This means that 10 yr swap will have negative value.\n"
        )
        fout.write("\n")


def mini5():
    # INTC CDS SR
    # Quote on 03/15/2022
    t = np.array([1, 2, 3, 4, 5])
    test_bid = np.array([1.80, 4.32, 10.25, 19.73, 24.52])
    test_ask = np.array([12.66, 17.88, 26.33, 34.84, 44.24])
    test = (test_ask + test_bid) / 2
    test *= 0.0001

    test_cds = CDS(1, t, test)
    surv_prob = test_cds.survival_prob_list()

    with open("mini.txt", "a") as fout:
        fout.write("5.\n")
        fout.write("survival probabilities:\n")
        fout.write(f"{surv_prob}\n")
        fout.write("unconditional default probabilities:\n")
        fout.write(f"{-np.diff(surv_prob, prepend=1)}\n")
        fout.write("\n")


def mini6():
    # INTC data
    intc_std = 4_771_000_000
    intc_ltd = 33_805_000_000
    intc_e = 192_250_000_000

    inteldf = pd.read_csv("./HW1_data/INTC.csv")
    intc_return = inteldf["Close"].pct_change()
    intc_e_sigma = np.std(intc_return)

    intc_kmv = credit_risk.KMV(intc_std, intc_ltd, intc_e, intc_e_sigma)

    with open("mini.txt", "a") as fout:
        fout.write("6.\n")
        fout.write("At and sigma_t is: \n")
        fout.write(f"{intc_kmv.KMV_solver}\n")

        fout.write(f"DD: {intc_kmv.distance_to_default()}\n")
        fout.write(f"PD: {intc_kmv.probability_of_default()}\n")
        fout.write(f"expected recovery: {intc_kmv.expected_recovery()}\n")
        fout.write("\n")


def mini7():
    # intc_option = fetch_option_data("INTC")

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.plot_trisurf(
    #    intc_option["Maturity"],
    #    intc_option["strike"],
    #    intc_option["impliedVolatility"],
    #    linewidth=0.2,
    #    antialiased=True,
    # )

    # fig.savefig("iv surface.png", format="png", dpi=800)

    intc = yfinance.Ticker("INTC")
    intc_df = intc.history(start="2017-1-1", end="2022-1-1", interval="1d")
    intc_df = market_risk.generate_return_pnl(intc_df)
    intc_hist = intc_df["Return"].to_numpy()
    intc_evt = market_risk.EVT(intc_hist)
    params = intc_evt.parameter_estimation()

    with open("mini.txt", "a") as fout:
        fout.write("7.\n")
        fout.write(f"5% VaR is {intc_evt.u:.4f}\n")
        fout.write(f"nu is {intc_evt.nu}\n")
        fout.write(f"xi is {params[0]:.4f} and beta is {params[1]:.4f}\n")
        fout.write(f"u_star is {intc_evt.u_star():.4f}\n")
        fout.write("\n")


def main():
    mini1()
    mini2()
    mini3()
    mini4()
    mini5()
    mini6()
    mini7()


if __name__ == "__main__":
    main()
