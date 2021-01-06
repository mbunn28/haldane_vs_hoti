#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import ti
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.optimize as optimise
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

path = "output/phasediagram/ribbon"
if not os.path.exists(path):
            os.makedirs(path)

a = 0.2
b = 1
t = 0.2
l = 1
N = 220
periodic = False
res=125
phi = np.pi/2

if a == 1 and b == 1:
    aorb_name = 'ab'
    aorb = 1
elif a == 1:
    aorb_name = 'b'
    aorb = b
else:
    aorb_name = 'a'
    aorb = a

if t == 1 and l == 1:
    torl_name = 'tl'
    torl = 1
elif t == 1:
    torl_name = 'l'
    torl = l
else:
    torl_name = 't'
    torl = t

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
    
    return hamiltonian

energy_path = f"{path}/res{res}_energies_{aorb_name}{aorb}_{torl_name}{torl}"
evecs_path = f"{path}/res{res}_evecs_{aorb_name}{aorb}_{torl_name}{torl}"

k = np.linspace(-np.pi,np.pi,num=res)

if (os.path.exists(energy_path) and os.path.exists(evecs_path)):
    energies = joblib.load(energy_path)
    evecs = joblib.load(evecs_path)
else:
    energies = np.zeros((res, 6*N))
    evecs = np.zeros((res, 6*N, 6*N),dtype=complex)

    for i in range(res):
        energies[i,:], evecs[i,:,:] = np.linalg.eigh(hamiltonian(k[i]))

    joblib.dump(energies, energy_path)
    joblib.dump(evecs, evecs_path)

evecs = np.transpose(evecs, axes=(0,2,1))
waves = np.abs(evecs)**2
mask_left = np.sum(waves[:,:,:int(np.rint(3*N))],axis=2) > 0.75
mask_right = np.sum(waves[:,:,:int(np.rint(3*N))],axis=2) < 0.25
mask_other = np.logical_not(np.logical_or(mask_left,mask_right))

_, k =np.meshgrid(np.zeros(6*N),k)

fig = plt.figure(figsize=(10,20))
ax = fig.add_subplot(111)
ax.set_aspect(2)
# ax.set_ylim((-0.25,0.25))
ax.scatter(k[mask_left],energies[mask_left],c='b',s=1)
ax.scatter(k[mask_right],energies[mask_right],c='r',s=1)
ax.scatter(k[mask_other],energies[mask_other],c='black',s=0.5,marker='x',linewidth=0.25)
ax.set_xlabel(r'$ak$')
ax.set_ylabel(r'$E$')
ax.set_xticks((-np.pi,-np.pi/2,0,np.pi/2,np.pi))
ax.set_xticklabels((r'$-\pi$',r'$-\frac{\pi}{2}$',0,r'$\frac{\pi}{2}$',r'$\pi$'))
fig.tight_layout()
fig.savefig(f"{path}/ribbonspectra.png", dpi=500, bbox_inches='tight')
