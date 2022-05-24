#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:10:55 2019

@author: ZexinLiu
"""

import numpy as np
from matplotlib import pyplot as plt

freq = 10
# f = lambda x: np.sin(freq*np.pi*x)

def f(x): # f is not smooth
    conds = [x < -0.5, (x > -0.5) & (x < 0.5), x > 0.5]
    funcs = [lambda x: 0, lambda x: 1, lambda x: 0]
    return np.piecewise(x, conds, funcs)


Fs = 10000
def fourierSeries(period, N):
    """Calculate the Fourier series coefficients up to the Nth harmonic"""
    result = []
    T = len(period)
    t = np.arange(T)
    for n in range(N+1):
        an = 2/T*(period * np.cos(2*np.pi*n*t/T)).sum()
        bn = 2/T*(period * np.sin(2*np.pi*n*t/T)).sum()
        result.append((an, bn))
    return np.array(result)

t_period = np.arange(-1, 1, 1/Fs)
ab = fourierSeries(f(t_period), 10)
print (ab.shape)

# def u(x, anbn):
#     result = 0
#     for n, (a, b) in enumerate(anbn):
#         if n == 0:
#             result = a/2
#         else:
#             result = result + (a*np.cos(2*np.pi*n*x/2) + b * np.sin(2*np.pi*n*x/2)) * (n*np.pi)**(-2*s)
#     return result
#    
# 
# def reconstruct(P, anbn): # verified
#     result = 0
#     t = np.arange(P)
#     for n, (a, b) in enumerate(anbn):
#         if n == 0:
#             a = a/2
#         result = result + a*np.cos(2*np.pi*n*t/P) + b * np.sin(2*np.pi*n*t/P)
#     return result
# 
# F = fourierSeries(f(t_period), 100)
# plt.figure()
# plt.plot(t_period, f(t_period), label='Original', lw=5)
# plt.plot(t_period, reconstruct(len(t_period), ab[:20,:]), label='Reconstructed with 20 Harmonics', lw=1)
# plt.plot(t_period, reconstruct(len(t_period), ab[:100,:]), label='Reconstructed with 100 Harmonics', lw=1)
# plt.legend()
# plt.show()
