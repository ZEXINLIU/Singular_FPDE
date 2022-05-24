#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 18:34:51 2019

@author: ZexinLiu
"""

import numpy as np
from families import JacobiPolynomials


def basis_coeff_lin(N):
    J = JacobiPolynomials()
    J.__init__(alpha=0., beta=0.)
    P = J.eval([-1,1], np.arange(2))
    Q = np.zeros(2,)
    cd = np.zeros((2, np.max(N)-1))
    for i in range (np.max(N)-1):
        Q[0] = J.eval(-1,np.arange(i+2,i+3))
        Q[1] = J.eval(1,np.arange(i+2,i+3))
        cd[:,i] = np.linalg.solve(P,-Q)
    return cd

def basis_coeff_nonlin(N):
    J = JacobiPolynomials()
    J.__init__(alpha=0., beta=0.)
    Q = np.zeros(2,)
    fe = np.zeros((2, np.max(N)-1))
    for i in range (np.max(N)-1):
        P = J.eval([-1,1], np.arange(i,i+2))
        Q[0] = J.eval(-1,np.arange(i+2,i+3))
        Q[1] = J.eval(1,np.arange(i+2,i+3))
        fe[:,i] = np.linalg.solve(P,-Q)
    return fe

if __name__ == '__main__':
    N = np.linspace(1,6,6).astype('int')
    print (basis_coeff_lin(N))
    print (basis_coeff_nonlin(N))