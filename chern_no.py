#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.linalg

points = 1000
eigensys = np.zeros((points,points,7,6),dtype=complex)

def u(r,s,n,d):
    bra = eigensys[r%points,s%points,n+1,:]
    if d == 1:
        ket = eigensys[(r+1)%points,s%points,n+1,:]
    if d == 2:
        ket = eigensys[r%points,(s+1)%points,n+1,:]
    result = np.vdot(bra,ket)
    result = result/np.linalg.norm(result)
    return result

lattice = ti.Lattice(
PBC_i = True,
PBC_j = True,
Corners = False,
alpha = 1.5,
hal = 0.15,
M=0,
N=4)

lattice.large_hal = False
lattice.large_alpha = False

b1 = np.array([4*np.pi*np.sqrt(3)/9,0])
b2 = (2*np.pi/3)*np.array([1/np.sqrt(3),1])
length = 4*np.pi/np.sqrt(3)

for r in range(0,points):
    for s in range(0, points):
    
        k = (r*b1 + s*b2)/(length*points)
        lattice.initialize_periodic_hamiltonian(k)
        w, v = scipy.linalg.eigh(lattice.periodic_hamiltonian)
        idx = np.argsort(w)
        w = w[idx]
        v = v[:,idx]            
        eigensys[r,s,0,:] = w
        eigensys[r,s,1:7,:] = v

chern = np.zeros(6,dtype=complex)
for n in range(0,6):
    F = np.zeros((points,points),dtype=complex)
    for r in range(0,points):
        for s in range(0, points):
            f = np.log(u(r,s,n,1)*u(r+1,s,n,2)*np.conj(u(r,s+2,n,1))*np.conj(u(r,s,n,2)))
            F[r,s] = f
            
    chern[n] = np.sum(F, dtype=complex)/(2*np.pi*1j)

print(chern)