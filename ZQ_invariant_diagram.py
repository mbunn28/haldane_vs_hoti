#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg
import joblib
import os
import matplotlib as mpl
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

path_zq = "output/zq/diagrams"
if not os.path.exists(path_zq):
            os.makedirs(path_zq)
    
points = 2
iterations = 4
location = np.array([2,2], dtype=int)
N = 14
max_x = 2
min_x = 0
max_y = 2
min_y = 0

def rule(y):
    a = np.zeros(len(y))
    b = np.zeros(len(y))
    for i in range(len(y)):
        if 0 <= y[i] <= 1:
            a[i] = y[i]
            b[i] = 1
        if y[i] > 1:
            a[i] = 1
            b[i] = 2 - y[i]
    return a, b

res = points
x = np.linspace(min_x, max_x, num=points)
y = np.linspace(min_y, max_y, num=points)
a_vals, b_vals = rule(y)
l_vals, t_vals = rule(x)

l, a = np.meshgrid(l_vals,a_vals)
t, b = np.meshgrid(t_vals,b_vals)

# s_vals = np.linspace(0,1,num=points+1)
# ones = np.ones(points)
# up = np.append(s_vals, ones)
# down = np.append(ones, np.flipud(s_vals))
# l,a = np.meshgrid(up,up)
# t,b = np.meshgrid(down,down)

zq = ['z6','z2']
zq_phases = np.zeros((res,res,len(zq)))
small_energy = np.zeros((res,res,len(zq)))
M = int(3*(N**2))
phi = np.random.rand(6*(N**2),M)
phi = scipy.linalg.orth(phi)
for m in tqdm(range(res)):
    for n in range(res):        
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
            small_energy[n,m,j] = np.min(np.abs(lattice1.energies))
            evecs = lattice1.waves
            singlestates_a = evecs[:,:M]
            pa = np.einsum('ij,jk',singlestates_a,np.conjugate(singlestates_a.transpose()))
            lattice1.proj = np.einsum('ij,jk',pa,phi)

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
                if np.min(np.abs(lattice2.energies)) < small_energy[n,m,j]:
                    small_energy[n,m,j] = np.min(np.abs(lattice2.energies))
                evecs = lattice2.waves
                singlestates_b = evecs[:,:M]
                pb = np.einsum('ij,jk',singlestates_b,np.conjugate(singlestates_b.transpose()))
                lattice2.proj = np.einsum('ij,jk',pb,phi)
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
            zq_phase1 = 6*zq_phase/(2*np.pi)
            zq_phase2 = np.round(zq_phase1,2)
            if np.isclose(zq_phase1,zq_phase2) != True:
                zq_phase2 = np.NaN
            if zq_phase2 < -1e-1:
                zq_phase2 = zq_phase2 + 6
            zq_phases[n,m,j] = zq_phase2


joblib.dump(zq_phases,f'{path_zq}/zq_phases_N{N}_it{iterations}_res{points}')

N_or_res = "res"
Nphase = 600
path_phasediagram = "output/phasediagram/periodic"
x_to_plot = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_x_to_plot")
y_to_plot = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_y_to_plot")

fig1, ax1 = plt.subplots()
# x = np.linspace(0,2,num=res)
for i in range(np.shape(x_to_plot)[0]):
    ax1.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)

cmap = plt.cm.Dark2  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(6)]
cmaplist[0] = (0,0,0,0)
# create the new map
cmap = mpl.colors.ListedColormap(cmaplist)

im = ax1.pcolormesh(x,y,zq_phases[:,:,0],cmap=cmap)
cb1 = fig1.colorbar(im,cmap=cmap, format='%1i')
labels = np.arange(0,6,1)
loc    = np.array([5/12,15/12,25/12,35/12,45/12,55/12])
cb1.set_ticks(loc)
cb1.set_ticklabels(labels)
title = '$\mathbb{Z}_6$ Berry Phase'
ax1.set_title(rf'{title}: $ N = {N},$ it $= {iterations}$, $res = {points}$')
ax1.set_ylabel(r'$\alpha$')
ax1.set_xlabel(r'$\lambda$')
ax1.set_xlim(min_x,max_x)
ax1.set_ylim(min_y,max_y)

def format_func(value, tick_number):
    if value <= 1:
        return f'{np.round(value,3)}'
    else:
        v = np.round(2 - value, 3)
        part1 = r'$\frac{1}{'
        part2 = r'}$'
        return fr'{part1}{v}{part2}'

ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

fig_path = f"{path_zq}/N{N}_iter{iterations}_res{points}_z6"
fig1.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')

fig2, ax2 = plt.subplots()
for i in range(np.shape(x_to_plot)[0]):
    ax2.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)

cmap = plt.cm.tab10  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(2)]
cmaplist[0] = (0,0,0,0)
# create the new map
cmap = mpl.colors.ListedColormap(cmaplist)

im1 = ax2.pcolormesh(x,y,zq_phases[:,:,1]/3,cmap=cmap)
cb2 = fig2.colorbar(im1,cmap=cmap, format='%1i')
labels1 = [0,1]
loc1    = np.array([1/4,3/4])
cb2.set_ticks(loc1)
cb2.set_ticklabels(labels1)
title1 = '$\mathbb{Z}_2$ Berry Phase'
ax2.set_title(rf'{title1}: $ N = {N},$ it $= {iterations}$, res $= {points}$')
ax2.set_ylabel(r'$\alpha$')
ax2.set_xlabel(r'$\lambda$')
ax2.set_xlim(min_x,max_x)
ax2.set_ylim(min_y,max_y)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
fig_path1 = f"{path_zq}/N{N}_iter{iterations}_res{points}_z2"
fig2.savefig(f"{fig_path1}.png", dpi=500, bbox_inches='tight')

small_energy[small_energy>1e-2] = np.NaN

fig3, ax3 = plt.subplots()
for i in range(np.shape(x_to_plot)[0]):
    ax3.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)
im2 = ax3.pcolormesh(x,y,small_energy[:,:,0], norm = colors.LogNorm(), cmap='inferno')
fig3.colorbar(im2)
title2 = f'Small energy in {title} calc'
ax3.set_title(rf'{title2}: $ N = {N},$ it $= {iterations}$, $res = {points}$')
ax3.set_ylabel(r'$\alpha$')
ax3.set_xlabel(r'$\lambda$')
ax3.set_xlim(min_x,max_x)
ax3.set_ylim(min_y,max_y)
ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

fig_path2 = f"{path_zq}/N{N}_iter{iterations}_res{points}_z6_energy"
fig3.savefig(f"{fig_path2}.png", dpi=500, bbox_inches='tight')

fig4, ax4 = plt.subplots()
for i in range(np.shape(x_to_plot)[0]):
    ax4.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)
im3 = ax4.pcolormesh(x,y,small_energy[:,:,1], norm = colors.LogNorm(), cmap='inferno')
fig4.colorbar(im3)
title3 = f'Small energy in {title1} calc'
ax4.set_title(rf'{title3}: $ N = {N},$ it $= {iterations}$, res $= {points}$')
ax4.set_ylabel(r'$\alpha$')
ax4.set_xlabel(r'$\lambda$')
ax4.set_xlim(min_x,max_x)
ax4.set_ylim(min_y,max_y)
ax4.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax4.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
fig_path3 = f"{path_zq}/N{N}_iter{iterations}_res{points}_z2_energy"
fig4.savefig(f"{fig_path3}.png", dpi=500, bbox_inches='tight')
