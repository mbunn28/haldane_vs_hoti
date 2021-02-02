#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import scipy.linalg
import numpy.random
import numpy.ma
from tqdm import tqdm
from scipy.signal import argrelextrema
import zq_lib

def rule(n):
    if n < 10:
        a = 0.1*n + 0.01
        b = 1
        alph = a
    elif n >= 10:
        a = 1
        b = 0.99 - 0.1*(n-10)
        alph = 2 - b
    else:
        print('error defining params!')
        a = 0
        b = 0
        alph = 0
    return a, b, alph

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble= r'\usepackage{amsfonts}')

indep = 'Alpha'
val = 1.7
num = 20
iterations = 200
location = np.array([4,4], dtype=int)
N = 8


path_zq = "output/zq"
if not os.path.exists(path_zq):
            os.makedirs(path_zq)

N_or_res = "res"
Nphase = 600
path_phasediagram = "output/phasediagram/periodic"

x = np.linspace(0,2,num=2*Nphase+1)
x_mesh = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_xmesh")
y_mesh = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_ymesh")

j = min(range(len(x)), key=lambda i: abs(x[i]-val))
if indep == 'Lambda':
    x_vals = x_mesh[j,:]
else:
    x_vals  = y_mesh[:,j]
x_vals = x_vals[~np.isnan(x_vals)]
print(x_vals)

zq = ['z6', 'z2']
zq_phases = np.zeros((num,len(zq)))
alphs = np.zeros(num)
for n in tqdm(range(num)):
    if indep == 'Alpha':
        if 0 <= val <= 1:
            l = val
            t = 1

            a, b, alphs[n] = rule(n)
        
        else:
            t = val
            l = 1
            a, b, alphs[n] = rule(n)
    
    else:
        if 0 <= val <= 1:
            a = val
            b = 1

            l, t, alphs[n] = rule(n)
        
        else:
            b = val
            a = 1

            l, t, alphs[n] = rule(n)
    
    for j in range(len(zq)):
        lattice1 = zq_lib.zq_lattice(
            a = a,
            b = b,
            l = l,
            t = t,
            the = zq_lib.curve(0, zq=zq[j]),
            loc = location,
            zq = zq[j],
            N = N
        )
                
        M = int(3*(lattice1.N**2))
        D = 1
        energies = np.zeros((6*(N**2),iterations+1))
        ui = np.zeros(iterations,dtype=complex)

        lattice1.twist_hamiltonian()
        energies[:,0] = lattice1.energies
        evecs = lattice1.waves
        singlestates_a = evecs[:,:M]
        pa = np.matmul(singlestates_a,np.conjugate(singlestates_a.transpose()))
        phi = np.random.rand(2*M,M)
        phi = scipy.linalg.orth(phi)
        lattice1.proj = np.matmul(pa,phi)

        for i in range(iterations):

            lattice2 = zq_lib.zq_lattice(
                a = a,
                b = b,
                l = l,
                t = t,
                the = zq_lib.curve((i+1)/iterations, zq = zq[j]),
                loc = location,
                zq = zq[j],
                N = N
            )

            lattice2.twist_hamiltonian()
            energies[:,i+1] = lattice2.energies
            evecs = lattice2.waves
            singlestates_b = evecs[:,:M]
            pb = np.matmul(singlestates_b,np.conjugate(singlestates_b.transpose()))
            lattice2.proj = np.matmul(pb,phi)
            Di = np.matmul(np.conjugate(lattice1.proj.transpose()),lattice2.proj)
            det_Di = numpy.linalg.slogdet(Di)
            if det_Di == 0:
                print('error! det zero!\n')
            ui[i] = det_Di[0]
            D = D*ui[i]
            
            Nphi = np.matmul(np.conjugate(lattice2.proj.transpose()), lattice2.proj)
            det_Nphi = scipy.linalg.det(Nphi)
            if det_Nphi == 0:
                print('The overlap matrix det = 0!')
            
            lattice1 = lattice2

        zq_phase = np.angle(D)
        if zq_phase < -1e-1:
            zq_phase = zq_phase + 2*np.pi
        zq_phase = zq_phase/(2*np.pi)
        zq_phases[n,j] = zq_phase



print(zq_phases)
fig, ax = plt.subplots()
ax.plot(alphs, zq_phases[:,0],'bo-',fillstyle="none")
ax.plot(alphs, zq_phases[:,1],'k^-',fillstyle="none")
title = '$\mathbb{Z}_Q$ Berry Phase'
if indep == 'Alpha' and val <= 1:
    x_label = r'$\alpha$'
    letter = r'\lambda'
    q = np.round(val,2)
elif indep == 'Alpha' and val > 1:
    x_label = r'$\alpha$'
    letter = r't'
    q = np.round(2-val,2)
elif indep == 'Lambda' and val <= 1:
    x_label = r'$\lambda$'
    letter = r'\alpha'
    q = np.round(val,2)
else:
    x_label = r'$\lambda$'
    letter = r'\beta'
    q = np.round(2-val,2)
ax.set_title(rf'{title}: $ {letter}= {q}, N = {N},$ it $= {iterations}$')
ax.set_xlabel(x_label)
ax.set_ylabel(r'$\gamma / 2\pi$')
ax.legend({r'$\mathbb{Z}_2$', r'$\mathbb{Z}_6$'})
for i in range(len(x_vals)):
    if (min(alphs)-0.1) <= x_vals[i] <= (max(alphs)+0.1):
        ax.axvline(x_vals[i],ls='--', c='gray')

def format_func(value, tick_number):
    if value <= 1:
        return f'{value}'
    else:
        v = 2 - value
        part1 = r'$\frac{1}{'
        part2 = r'}$'
        return fr'{part1}{v}{part2}'
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

fig_path = f"{path_zq}/{indep}{val}_N{N}_iter{iterations}"
fig.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')

# fig, ax = plt.subplots()
# ax.plot(energies.transpose(),'b')
# ax.set_title(r'Energies along parameter space curve:$\lambda = 0.4, \alpha = 0.22, N = 10$')
# ax.set_xlabel(r'$\tau$ path parameter')
# ax.set_ylabel(r'$E$')
# fig_path1 = f"{path}/energies_l04_a022__N{N}"
# fig.savefig(f"{fig_path1}.png", dpi=500, bbox_inches='tight')

# fig1, ax1 = plt.subplots()
# en = energies[:M,:]
# ax1.plot(en.transpose(),'b')
# ax1.set_title(r'Energies along parameter space curve:$\lambda = 0.4, \alpha = 0.22, N = 10$')
# ax1.set_xlabel(r'$\tau$ path parameter')
# ax1.set_ylabel(r'$E$')
# fig_path1 = f"{path}/energies_l04_a022__N{N}_lowestM"
# fig1.savefig(f"{fig_path1}.png", dpi=500, bbox_inches='tight')