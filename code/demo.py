#!/usr/bin/env python3
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

#Initial price of stock
S0 = 100
#Strike price
K = 100
#Interest rate
r = 0.01
#Coefficient of dW in SDE for GBM
sigma = 0.01
#Time step
dt = 0.1
#Time until expiry
T = 90
#Number of simulations
n = 1000

#Brownian motion
dW = np.random.normal(size=(n,int(T/dt)-1))
W = np.hstack([np.zeros((n,1)),np.cumsum(dW,axis=1)])
#Time dimension
times = np.linspace(0,T,int(T/dt))
#GBM
S = S0*np.exp(sigma*W-sigma**2*times/2)
dS = np.diff(S,axis=1)

#Differentiate Black-Scholes formula wrt S
#https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#The_Options_Greeks
d1 = (np.log(S/K)+(r*sigma**2/2)*(T-times))/(sigma*np.sqrt(T-times))
delta = sp.stats.norm.cdf(d1[:,:int(T/dt)-1])

#Replicating portfolio
portfolio = np.cumsum(delta * dS,axis=1)

#Stock price
plt.plot(times,S.T);
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.savefig('../output/stock_price.png')
plt.close()
#Replicating portfolio value
plt.plot(times[:-1],portfolio.T)
plt.xlabel('Time')
plt.ylabel('Value of Replicating Portfolio')
plt.savefig('../output/replicating_portfolio.png')
plt.close()
#Payoff diagram
plt.scatter(S[:,-1],portfolio[:,-1])
plt.xlabel('Terminal Stock Price')
plt.ylabel('Terminal Value of Replicating Portfolio')
plt.savefig('../output/payoff_diagram.png')
plt.close()
#plt.scatter(S[:,:-1].reshape(-1),delta.reshape(-1));plt.show()
