#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.optimize as optimise
from matplotlib import rc

a = 1
b = 1/6 + 0.1
t = 1
l = 1/np.sqrt(3) + 0.15
N = 96
periodic = False
res=250
phi = np.pi/2

def hamiltonian(k):
    A1 = b*t*np.roll(np.eye(6,dtype=complex),1,axis=1)
    A2 = b*l*np.exp(1j*phi)*np.roll(np.eye(6,dtype=complex),2,axis=1)
    A3 = np.zeros((6,6),dtype=complex)
    A3[0,2] = a*l*np.exp(1j*phi)*np.exp(-1j*k)
    A3[0,3] = a*t*np.exp(-1j*k)
    A3[1,3] = a*l*np.exp(1j*phi)*np.exp(-1j*k)
    A3[3,5] = a*l*np.exp(1j*phi)*np.exp(1j*k)
    A3[4,0] = a*l*np.exp(1j*phi)*np.exp(1j*k)
    A = A1 + A2 + A3
    bigA = np.kron(np.eye(N,dtype=complex),A)

    B = np.zeros((6,6),dtype=complex)
    B[0,4] = a*l*np.exp(-1j*phi)*np.exp(-1j*k)
    B[1,3] = a*l*np.exp(1j*phi)*np.exp(-1j*k)
    B[1,4] = a*t*np.exp(-1j*k)
    B[1,5] = a*l*np.exp(-1j*phi) + a*l*np.exp(-1j*phi)*np.exp(-1j*k)
    B[2,0] = a*l*np.exp(-1j*phi)
    B[2,4] = a*l*np.exp(1j*phi) + a*l*np.exp(1j*phi)*np.exp(-1j*k)
    B[2,5] = a*t
    B[3,5] = a*l*np.exp(1j*phi)
    bigB = np.kron(np.eye(N,dtype=complex),B)
    bigB = np.roll(bigB, 6, axis=1)

    if periodic == False:
        bigB[:,:6] = np.zeros((6*N,6),dtype=complex)

    hamiltonian = bigA + bigB
    hamiltonian = hamiltonian + np.transpose(np.conjugate(hamiltonian))
    
    energies = np.linalg.eigvalsh(hamiltonian)
    return energies

k = np.linspace(-np.pi,np.pi,num=res)
energies = np.zeros((res, 6*N))
for i in range(res):
    energies[i,:] = hamiltonian(k[i])

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

path = "output/phasediagram/ribbon"
if not os.path.exists(path):
            os.makedirs(path)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(20)
ax.set_ylim((-0.25,0.25))
ax.plot(k,energies,'b',linewidth=0.5)
ax.set_xlabel(r'$ak$')
ax.set_ylabel(r'$E$')
ax.set_xticks((-np.pi,-np.pi/2,0,np.pi/2,np.pi))
ax.set_xticklabels((r'$-\pi$',r'$-\frac{\pi}{2}$',0,r'$\frac{\pi}{2}$',r'$\pi$'))
fig.tight_layout()
fig.savefig(f"{path}/ribbonspectra.png", dpi=500, bbox_inches='tight')
