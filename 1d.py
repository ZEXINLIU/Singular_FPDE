#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:31:36 2019

@author: zexin
"""

import numpy as np
from matplotlib import pyplot as plt
from families import JacobiPolynomials
import scipy

"""
1d case

(-delta)^s u = f, x in (-1,1)
u = 0 on the Boundary
"""
#s = 1.
s = 0.8


#freq = 10
#f = lambda x: np.sin(freq*np.pi*x)
#u = lambda x: (freq*np.pi)**(-2*s) * np.sin(freq*np.pi*x)

#freq = 10
#f = lambda x: np.sin(freq*np.pi/2*x + freq*np.pi/2) # f is smooth

def f(x): # f is not smooth
    conds = [x < -0.5, (x > -0.5) & (x < 0.5), x > 0.5]
    funcs = [lambda x: 0, lambda x: 1, lambda x: 0]
    return np.piecewise(x, conds, funcs)

#u = lambda x: (freq*np.pi/2)**(-2*s) * f(x)

J = JacobiPolynomials()
J.__init__(alpha=0., beta=0.)

Nquad = 1000
x,w = J.gauss_quadrature(Nquad)

fig,ax = plt.subplots(1,2)
#ax[0].plot(x, u(x), label = 'utrue')
#ax[0].plot(x, u(x, ab), label = 'utrue')

"""
requirement for N: N >= 3 ?
"""
N = np.arange(1,100,20)
l2_err = np.zeros(N.size)

Q = np.zeros(2,)
fe = np.zeros((2, np.max(N)))
for i in range (np.max(N)):
    P = J.eval([-1,1], np.arange(i,i+2))
    Q[0] = J.eval(-1,np.arange(i+2,i+3))
    Q[1] = J.eval(1,np.arange(i+2,i+3))
    fe[:,i] = np.linalg.solve(P,-Q)

Verr1 = J.eval(x, np.arange(2,np.max(N)+2))
Verr2 = fe[1,:] * J.eval(x, np.arange(1,np.max(N)+1))
Verr3 = fe[0,:] * J.eval(x, np.arange(0,np.max(N)))
Verr = Verr1 + Verr2 + Verr3
M = np.dot((w * Verr.T), Verr)

F = np.dot(w * Verr.T, f(x))

DVerr1 = J.eval(x, np.arange(2,np.max(N)+2), 1)
DVerr2 = fe[1,:] * J.eval(x, np.arange(1,np.max(N)+1), 1)
DVerr3 = fe[0,:] * J.eval(x, np.arange(0,np.max(N)), 1)
DVerr = DVerr1 + DVerr2 + DVerr3
S = np.dot((w * DVerr.T), DVerr)


Meigw, Meigv = np.linalg.eig(M)
sq = np.dot(np.dot(Meigv, np.diag(Meigw**(1/2))), np.linalg.inv(Meigv))
sqm = np.dot(np.dot(Meigv, np.diag(Meigw**(-1/2))), np.linalg.inv(Meigv))

eigw, eigv = np.linalg.eig(np.dot(np.linalg.inv(M), S))


#eig1 = np.sort((np.arange(1,max(N)+1)*np.pi)**2)
#eig2 = np.sort(eigw)
#plt.plot(np.arange(1,np.max(N)+1), np.sort((np.arange(1,max(N)+1)*np.pi)**2), label = r'$\lambda_n$')
#plt.plot(np.arange(1,np.max(N)+1), np.sort(eigw), label = r'$\lambda_n^N$')
#plt.legend()

P = np.dot(sq,eigv)
norm_array = np.linalg.norm(P, axis=0)
norm_diag = np.diag(1/norm_array)
P_sd = np.dot(P,norm_diag)


for (ind,n) in enumerate(N):
    eigw, eigv = np.linalg.eig(np.dot(np.linalg.inv(M[0:n,0:n]), S[0:n,0:n]))
    diag_s = np.diag(eigw**(-s))
    
    P = np.dot(scipy.linalg.sqrtm(M[0:n,0:n]), eigv)
    norm_diag = np.diag(1/np.linalg.norm(P, axis=0))
    
    u_coeff = np.dot(np.dot(np.dot(np.dot(eigv, diag_s), norm_diag**2), eigv.T), F[0:n])
    u_N = np.dot(Verr[:,0:n], u_coeff)
    ax[0].plot(x, u_N, label = 'N = {0:d}'.format(n))
    l2_err[ind] = np.sqrt(np.dot(w,(u_N-u(x))**2))
#    l2_err[ind] = np.sqrt(np.dot(w,(u_N-u(x, ab))**2))
    

ax[0].legend(loc = 'best')
ax[0].set_xlabel('x')
ax[0].set_ylabel('u')

ax[1].loglog(N, l2_err)
ax[1].set_xlabel('N')
ax[1].set_ylabel('L2 err')

plt.tight_layout()
