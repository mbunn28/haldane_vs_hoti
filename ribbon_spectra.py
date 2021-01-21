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

path = "output/ribbon"
if not os.path.exists(path):
            os.makedirs(path)

a = 1
b = 0.5575
t = 0.2
l = 1
N = 1000
periodic = False
res=250
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

energy_path = f"{path}/res{res}_N{N}_energies_{aorb_name}{aorb}_{torl_name}{torl}"
mask_left_path = f"{path}/res{res}_N{N}_left_{aorb_name}{aorb}_{torl_name}{torl}"
mask_right_path = f"{path}/res{res}_N{N}_right_{aorb_name}{aorb}_{torl_name}{torl}"

k = np.linspace(-np.pi,np.pi,num=res)

if (os.path.exists(energy_path) and os.path.exists(mask_left_path) and os.path.exists(mask_right_path)):
    energies = joblib.load(energy_path)
    mask_left = joblib.load(mask_left_path)
    mask_right = joblib.load(mask_right_path)
else:
    energies = np.zeros((res, 6*N))
    mask_left = np.zeros((res, 6*N),dtype=bool)
    mask_right = np.zeros((res, 6*N),dtype=bool)

    for i in range(res):
        print(f"{i}/{res}", end='\r')

        energies[i,:], evecs = np.linalg.eigh(hamiltonian(k[i]))
        evecs = np.transpose(evecs, axes=(1,0))
        waves = np.abs(evecs)**2
        mask_left[i,:] = np.sum(waves[:,:int(np.rint(3*N))],axis=1) > 0.75
        mask_right[i,:] = np.sum(waves[:,:int(np.rint(3*N))],axis=1) < 0.25

    joblib.dump(energies, energy_path)
    joblib.dump(mask_left, mask_left_path)
    joblib.dump(mask_right, mask_right_path)

<<<<<<< HEAD
mask_other = np.logical_not(np.logical_or(mask_left,mask_right))

_, k =np.meshgrid(np.zeros(6*N),k)

fig = plt.figure(figsize=(10,20))
ax = fig.add_subplot(111)
ax.set_aspect(10)
ax.set_ylim((-0.25,0.25))
ax.scatter(k[mask_left],energies[mask_left],c='b',s=1)
ax.scatter(k[mask_right],energies[mask_right],c='r',s=1)
ax.scatter(k[mask_other],energies[mask_other],c='black',s=0.5,marker='x',linewidth=0.25)
ax.set_xlabel(r'$ak$')
ax.set_ylabel(r'$E$')
ax.set_xticks((-np.pi,-np.pi/2,0,np.pi/2,np.pi))
ax.set_xticklabels((r'$-\pi$',r'$-\frac{\pi}{2}$',0,r'$\frac{\pi}{2}$',r'$\pi$'))
fig.tight_layout()

fig_path = f"{path}/res{res}_N{N}_ribbonspectrum_{aorb_name}{aorb}_{torl_name}{torl}"
fig.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')
=======
# mask_other = np.logical_not(np.logical_or(mask_left,mask_right))

# _, k =np.meshgrid(np.zeros(6*N),k)

# fig = plt.figure(figsize=(10,20))
# ax = fig.add_subplot(111)
# ax.set_aspect(2)
# # ax.set_ylim((-0.25,0.25))
# ax.scatter(k[mask_left],energies[mask_left],c='b',s=1)
# ax.scatter(k[mask_right],energies[mask_right],c='r',s=1)
# ax.scatter(k[mask_other],energies[mask_other],c='black',s=0.5,marker='x',linewidth=0.25)
# ax.set_xlabel(r'$ak$')
# ax.set_ylabel(r'$E$')
# ax.set_xticks((-np.pi,-np.pi/2,0,np.pi/2,np.pi))
# ax.set_xticklabels((r'$-\pi$',r'$-\frac{\pi}{2}$',0,r'$\frac{\pi}{2}$',r'$\pi$'))
# fig.tight_layout()

# fig_path = f"{path}/res{res}_N{N}_ribbonspectrum_{aorb_name}{aorb}_{torl_name}{torl}"
# fig.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')
>>>>>>> bc824f01d89768d31e34146f69a93377f7a7bde6
