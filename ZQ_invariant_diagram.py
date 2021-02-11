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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble= r'\usepackage{amsfonts}')

path_zq = "output/zq"
if not os.path.exists(path_zq):
            os.makedirs(path_zq)
    
points = 50
iterations = 4
location = np.array([2,2], dtype=int)
N = 4
max_x = 2
min_x = 0

s_vals = np.linspace(0,1,num=points+1)
ones = np.ones(points)
up = np.append(s_vals, ones)
down = np.append(ones, np.flipud(s_vals))
l,a = np.meshgrid(up,up)
t,b = np.meshgrid(down,down)

zq = ['z6', 'z2']
zq_phases = np.zeros((2*points+1,2*points+1,len(zq)))
M = int(3*(N**2))
phi = np.random.rand(2*M,M)
phi = scipy.linalg.orth(phi)
for m in tqdm(range(2*points+1)):
    for n in range(2*points+1):        
        for j in range(len(zq)):
            lattice1 = zq_lib.zq_lattice(
                a = a[n,m],
                b = b[n,m],
                l = l[n,m],
                t = t[n,m],
                the = zq_lib.curve(0, zq=zq[j]),
                loc = location,
                zq = zq[j],
                N = N
            )
            
            D = 1
            lattice1.twist_hamiltonian()
            evecs = lattice1.waves
            singlestates_a = evecs[:,:M]
            pa = np.einsum('ij,jk',singlestates_a,np.conjugate(singlestates_a.transpose()))
            lattice1.proj = np.matmul(pa,phi)

            for i in range(iterations):

                lattice2 = zq_lib.zq_lattice(
                    a = a[n,m],
                    b = b[n,m],
                    l = l[n,m],
                    t = t[n,m],
                    the = zq_lib.curve((i+1)/iterations, zq = zq[j]),
                    loc = location,
                    zq = zq[j],
                    N = N
                )

                lattice2.twist_hamiltonian()
                evecs = lattice2.waves
                singlestates_b = evecs[:,:M]
                pb = np.einsum('ij,jk',singlestates_b,np.conjugate(singlestates_b.transpose()))
                lattice2.proj = np.matmul(pb,phi)
                Di = np.einsum('ij,jk',np.conjugate(lattice1.proj.transpose()),lattice2.proj)
                det_Di = numpy.linalg.slogdet(Di)
                if det_Di == 0:
                    print('error! det zero!\n')
                D = D*det_Di[0]
                
                Nphi = np.einsum('ij,jk',np.conjugate(lattice2.proj.transpose()), lattice2.proj)
                _, det_Nphi = numpy.linalg.slogdet(Nphi)
                if det_Nphi == -np.Inf:
                    print('The overlap matrix det = 0!')
                
                lattice1 = lattice2

            zq_phase = np.angle(D)
            if zq_phase < -1e-1:
                zq_phase = zq_phase + 2*np.pi
            zq_phase = 6*zq_phase/(2*np.pi)
            zq_phases[n,m,j] = zq_phase


joblib.dump(zq_phases,f'{path_zq}/zq_phases_N{N}_it{iterations}_res{points}')

N_or_res = "res"
Nphase = 600
path_phasediagram = "output/phasediagram/periodic"
x_to_plot = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_x_to_plot")
y_to_plot = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_y_to_plot")

print(zq_phases)
fig1, ax1 = plt.subplots()
x = np.linspace(0,2,num=2*points+1)
for i in range(np.shape(x_to_plot)[0]):
    ax1.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)
im = ax1.pcolormesh(x,x,zq_phases[:,:,0],cmap='Accent')
fig1.colorbar(im)
title = '$\mathbb{Z}_6$ Berry Phase'
ax1.set_title(rf'{title}: $ N = {N},$ it $= {iterations}$, $res = {points}$')
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel(r'$\lambda$')

def format_func(value, tick_number):
    if value <= 1:
        return f'{value}'
    else:
        v = np.round(2 - value, 3)
        part1 = r'$\frac{1}{'
        part2 = r'}$'
        return fr'{part1}{v}{part2}'

ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

fig_path = f"{path_zq}/diagram_N{N}_iter{iterations}"
fig1.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')

fig2, ax2 = plt.subplots()
for i in range(np.shape(x_to_plot)[0]):
    ax2.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)
im1 = ax2.pcolormesh(x,x,zq_phases[:,:,1],cmap='Accent')
fig2.colorbar(im1)
title1 = '$\mathbb{Z}_2$ Berry Phase'
ax2.set_title(rf'{title1}: $ N = {N},$ it $= {iterations}$, $res = {points}$')
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel(r'$\lambda$')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
fig_path = f"{path_zq}/diagram_N{N}_iter{iterations}_z2"
fig2.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')
