#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:53:19 2019

@author: ZexinLiu
"""

import numpy as np
from matplotlib import pyplot as plt
from families import JacobiPolynomials
import scipy


"""
2d case

(-delta)^s u = f, x in (-1,1)
u = 0 on the Boundary
"""
s = 1
freqx = 1
freqy = 2
f = lambda x, y: np.sin(freqx * np.pi * x) * np.sin(freqy * np.pi * y)
u = lambda x, y: ((freqx*np.pi)**2 + (freqy * np.pi)**2)**(-s) * np.sin(freqx * np.pi*x) * np.sin(freqy * np.pi * y)


J = JacobiPolynomials()
J.__init__(alpha=0., beta=0.)

Nquad = 30
x,wx = J.gauss_quadrature(Nquad)
y,wy = J.gauss_quadrature(Nquad)

xv, yv = np.meshgrid(x, y)
utrue = u(xv, yv)
fvalue = f(xv, yv)

W = np.tensordot(wy,wx,axes=0)

"""
requirement for N: N >= 3 ?
"""
#N = np.arange(3,30,8)
N = np.array([20])
l2_err = np.zeros(N.size)

Q = np.zeros(2,)
fe = np.zeros((2, np.max(N)))
for i in range (np.max(N)):
    P = J.eval([-1,1], np.arange(i,i+2))
    Q[0] = J.eval(-1,np.arange(i+2,i+3))
    Q[1] = J.eval(1,np.arange(i+2,i+3))
    fe[:,i] = np.linalg.solve(P,-Q)

Verr1_x = J.eval(x, np.arange(2,np.max(N)+2))
Verr2_x = fe[1,:] * J.eval(x, np.arange(1,np.max(N)+1))
Verr3_x = fe[0,:] * J.eval(x, np.arange(0,np.max(N)))
Verr_x = Verr1_x + Verr2_x + Verr3_x

Verr1_y = J.eval(y, np.arange(2,np.max(N)+2))
Verr2_y = fe[1,:] * J.eval(y, np.arange(1,np.max(N)+1))
Verr3_y = fe[0,:] * J.eval(y, np.arange(0,np.max(N)))
Verr_y = Verr1_y + Verr2_y + Verr3_y


Verr = np.zeros((Nquad, np.max(N)**2*Nquad))
for i in range(np.max(N)):
    for j in range(np.max(N)):
        phi_ij = np.tensordot(Verr_y[:,j], Verr_x[:,i], axes = 0)
        Verr[:, (i*np.max(N)*Nquad + j*Nquad):(i*np.max(N)*Nquad + (j+1)*Nquad)] = phi_ij

M = np.zeros((np.max(N)**2,np.max(N)**2))
for i in range(np.max(N)**2):
    for j in range(np.max(N)**2):
        M[i,j] = np.sum(W * np.hsplit(Verr, np.max(N)**2)[j] * np.hsplit(Verr, np.max(N)**2)[i])


DVerr1_x = J.eval(x, np.arange(2,np.max(N)+2), 1)
DVerr2_x = fe[1,:] * J.eval(x, np.arange(1,np.max(N)+1), 1)
DVerr3_x = fe[0,:] * J.eval(x, np.arange(0,np.max(N)), 1)
DVerr_x = DVerr1_x + DVerr2_x + DVerr3_x

DVerr1_y = J.eval(y, np.arange(2,np.max(N)+2), 1)
DVerr2_y = fe[1,:] * J.eval(y, np.arange(1,np.max(N)+1), 1)
DVerr3_y = fe[0,:] * J.eval(y, np.arange(0,np.max(N)), 1)
DVerr_y = DVerr1_y + DVerr2_y + DVerr3_y

Dx_Verr = np.zeros((Nquad, np.max(N)**2*Nquad))
for i in range(np.max(N)):
    for j in range(np.max(N)):
        phi_ij = np.tensordot(Verr_y[:,j], DVerr_x[:,i], axes = 0)
        Dx_Verr[:, (i*np.max(N)*Nquad + j*Nquad):(i*np.max(N)*Nquad + (j+1)*Nquad)] = phi_ij
#
S_x = np.zeros((np.max(N)**2,np.max(N)**2))
for i in range(np.max(N)**2):
    for j in range(np.max(N)**2):
        S_x[i,j] = np.sum(W * np.hsplit(Dx_Verr, np.max(N)**2)[j] * np.hsplit(Dx_Verr, np.max(N)**2)[i])


Dy_Verr = np.zeros((Nquad, np.max(N)**2*Nquad))
for i in range(np.max(N)):
    for j in range(np.max(N)):
        phi_ij = np.tensordot(DVerr_y[:,j], Verr_x[:,i], axes = 0)
        Dy_Verr[:, (i*np.max(N)*Nquad + j*Nquad):(i*np.max(N)*Nquad + (j+1)*Nquad)] = phi_ij

S_y = np.zeros((np.max(N)**2,np.max(N)**2))
for i in range(np.max(N)**2):
    for j in range(np.max(N)**2):
        S_y[i,j] = np.sum(W * np.hsplit(Dy_Verr, np.max(N)**2)[j] * np.hsplit(Dy_Verr, np.max(N)**2)[i])

S = S_x + S_y

F = np.zeros(np.max(N)**2,)
for i in range(np.max(N)**2):
    F[i] = np.sum(W * fvalue * np.hsplit(Verr, np.max(N)**2)[i])


"""
general Galerkin test for s = 1.0 to check this algorothm works or not ---> works
"""
#for (ind,n) in enumerate(N):
#    u_coeff = np.linalg.solve(S[0:n**2,0:n**2], F[0:n**2])
#    
#    X = u_coeff.reshape(n,n)
#    u_N = np.dot(np.dot(Verr_x[:,0:n], X), Verr_y[:,0:n].T).T
#    
##    Verr = np.zeros((Nquad, n**2*Nquad))
##    for i in range(n):
##        for j in range(n):
##            phi_ij = np.tensordot(Verr_y[:,j], Verr_x[:,i], axes = 0)
##            Verr[:, (i*n*Nquad + j*Nquad):(i*n*Nquad + (j+1)*Nquad)] = phi_ij
##    
##    u_N = np.zeros((Nquad,Nquad))
##    for i in range(n**2):
##        u_N += u_coeff[i] * np.hsplit(Verr, n**2)[i]
#    
#    l2_err[ind] = np.sqrt(np.sum(W*(u_N-utrue)**2))


"""
eig method for given 0<s<1 and any freqx and freqy
"""
for (ind,n) in enumerate(N):
    eigw, eigv = np.linalg.eig(np.dot(np.linalg.inv(M[0:n**2,0:n**2]), S[0:n**2,0:n**2]))
    
    eigw = eigw.real
    eigv = eigv.real
    
    diag_s = np.diag(eigw**(-s))
    
    P = np.dot(scipy.linalg.sqrtm(M[0:n**2,0:n**2]), eigv)
    norm_diag = np.diag(1/np.linalg.norm(P, axis=0))
    
    u_coeff = np.dot(np.dot(np.dot(np.dot(eigv, diag_s), norm_diag**2), eigv.T), F[0:n**2])
    
    
    X = u_coeff.reshape(n,n)
    u_N = np.dot(np.dot(Verr_x[:,0:n], X), Verr_y[:,0:n].T).T
    
#    Verr = np.zeros((Nquad, n**2*Nquad))
#    for i in range(n):
#        for j in range(n):
#            phi_ij = np.tensordot(Verr_y[:,j], Verr_x[:,i], axes = 0)
#            Verr[:, (i*n*Nquad + j*Nquad):(i*n*Nquad + (j+1)*Nquad)] = phi_ij
#    u_N = np.zeros((Nquad,Nquad))
#    for i in range(n**2):
#        u_N += u_coeff[i] * np.hsplit(Verr, n**2)[i]
    
    l2_err[ind] = np.sqrt(np.sum(W * (u_N - utrue)**2))


fig,ax = plt.subplots()
ax.loglog(N, l2_err)
ax.set_xlabel('N')
ax.set_ylabel('L2 err')
plt.tight_layout()

