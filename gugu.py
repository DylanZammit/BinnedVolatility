import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaincc
from scipy.stats import invgamma
from scipy.optimize import fsolve
import pandas as pd
import yfinance as yf

def get_data(ticker='AAPL', start='2010-01-01', end='2021-03-01', use_daily=True):
    a = yf.Ticker(ticker)

    if use_daily:
        df = a.history(start='2010-01-01', end='2021-03-01').Close
    else:
        df = a.history(period='1mo', interval='5m')
        df = df.reset_index().Close

    X = df.apply(np.log)
    return X

class Kernel:

    def __init__(self, X, m):

        n = len(X)
        Y = X.diff()
        h = m/(2*n)
        N = n//m
        r = n-m*N

        self.m = m
        self.n = n
        self.h = h
        self.X = X
        self.Y = Y
        self.r = r

    def s2_kernel(self, t): #boxcar function
        Y = self.Y
        mid = int(t*self.n)
        l, r = self.m//2, self.m//2 #check if this correct
        inds = list(range(max([0, mid-l]), min([mid+self.r, self.n-1])))
        return 1/(2*self.h)*(Y.iloc[inds]**2).sum()

    def s2_kernel_gauss(self, t, Y): #gauss kernel
        return sum(self.gaus_ker((t-i/self.n)/self.h)*y**2 for i, y in enumerate(self.Y.fillna(0)))/h

    def gaus_ker(self, x):
        return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

class Gugu:

    def __init__(self, X, m, alpha=0.1, beta=0.1):
        n = len(X)
        N = n//m
        r = n-m*N
        Y = X.diff()

        self.X = X
        self.Y = Y
        self.m = m
        self.N = N
        self.r = r
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def s2_gugu(self, reps=True):

        Y = self.Y.copy()
        m = self.m
        N = self.N
        r = self.r
        n = self.n
        alpha = self.alpha
        beta = self.beta

        z = [(Y.iloc[m*k:m*(k+1)]**2).sum() for k in range(N-1)]
        z.append((Y.iloc[m*(N-1):]**2).sum())

        alpha_1 = [alpha+m/2]*len(z[:-1]) + [alpha+(m+r)/2]
        beta_1 = [beta+n*zk/2 for zk in z]

        mean = [b/(a-1) for a,b in zip(alpha_1, beta_1)]
        upper_95 = [invgamma.ppf(0.975, a, scale=b) for a,b in zip(alpha_1, beta_1)]
        lower_95 = [invgamma.ppf(0.025, a, scale=b) for a,b in zip(alpha_1, beta_1)]    

        if reps:
            reps = [m]*(N-1)+[r+m]
            mean = np.repeat(mean, reps)
            upper_95 = np.repeat(upper_95, reps)
            lower_95 = np.repeat(lower_95, reps)

        return mean, lower_95, upper_95

def main():

    X = get_data()
    m = 30
    g = Gugu(X, m)
    k = Kernel(X, m)
    
    dom = np.linspace(0, 1, len(X))

    mean_nopa, low_nopa, up_nopa = g.s2_gugu()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title('Apple Stock Returns')
    X.plot(ax=ax[0])

    ax[1].set_title('Apple Volatility Estimate')
    ax[1].plot(X.index, mean_nopa,label='Original Method')
    ax[1].fill_between(X.index, low_nopa, up_nopa, alpha=0.5)
    ax[1].plot(X.index, [k.s2_kernel(t) for t in dom], label='Boxcar Kernel', alpha=0.7)
    # plt.plot(X.index, [k.s2_kernel_gauss(t, Y) for t in dom], label='kernel_gauss')
    plt.legend()

    plt.show()

if __name__=='__main__':
    main()
