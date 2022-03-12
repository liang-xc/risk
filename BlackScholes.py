import math

import numpy as np


def european_option(S, K, d, r, sigma, tao, type='call'):
    """
    Black-Scholes European call options price and its greeks
    S: Current underlying price
    K: Strike price
    d: dividend rate
    r: risk-free rate
    sigma: volatility of underlying
    tao: time to exercise
    type: type of the option, should be either "call" or "put"
    """

    def option():
        print(f'Price of the {type} option is {price()}.')

    def is_call():
        if type == 'call':
            return True
        elif type == 'put':
            return False
        else:
            raise ValueError(
                f'{type} is not a valid type of option, use "call" or "put".')

    _call = is_call()

    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2)
          * tao) / (sigma * np.sqrt(tao))
    d2 = d1 - sigma * np.sqrt(tao)

    def N(x):
        return (1 + math.erf(x / np.sqrt(2))) / 2

    def dN(x):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)

    def price():
        if _call:
            return S * np.exp(-d * tao) * N(d1) - K * np.exp(-r * tao) * N(d2)
        else:
            return K * np.exp(-r * tao) * N(-d2) - S * np.exp(-d * tao) * N(-d1)

    def delta():
        if _call:
            return N(d1)
        else:
            return N(d1) - 1

    def gamma():
        return dN(d1) / (S * sigma * np.sqrt(tao))

    def vega():
        return S * dN(d1) * np.sqrt(tao)

    def theta():
        if _call:
            return -(S * dN(d1) * sigma) / (2 * np.sqrt(tao)) - r * K * np.exp(-r * tao) * N(d2)
        else:
            return -(S * dN(d1) * sigma) / (2 * np.sqrt(tao)) + r * K * np.exp(-r * tao) * N(-d2)

    def rho():
        if _call:
            return K * tao * np.exp(-r * tao) * N(d2)
        else:
            return -K * tao * np.exp(-r * tao) * N(-d2)

    # attach enclosed function to an object as its attributes
    option.price = price
    option.delta = delta
    option.gamma = gamma
    option.vega = vega
    option.theta = theta
    option.rho = rho

    return option


def main():
    call_sample = european_option(100, 100, 0, 0, 0.3, 1, 'call')
    call_sample()
    print(f'delta: {call_sample.delta()}')
    print(f'gamma: {call_sample.gamma()}')
    print(f'vega: {call_sample.vega()}')
    print(f'theta: {call_sample.theta()}')
    print(f'rho: {call_sample.rho()}')

    put_sample = european_option(100, 100, 0, 0, 0.3, 1, 'put')
    put_sample()
    print(f'delta: {put_sample.delta()}')
    print(f'gamma: {put_sample.gamma()}')
    print(f'vega: {put_sample.vega()}')
    print(f'theta: {put_sample.theta()}')
    print(f'rho: {put_sample.rho()}')


if __name__ == '__main__':
    main()
