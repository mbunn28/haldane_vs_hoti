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
    
points = 20
iterations = 4
location = np.array([2,2], dtype=int)
N = 3
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

joblib.dump(zq_phases,f'{path_zq}/zq_phases')

print(zq_phases)
fig1, ax1 = plt.subplots()
im = ax1.pcolormesh(zq_phases[:,:,0])
fig1.colorbar(im)
# # ax.plot(alphs, zq_phases[:,1],'k^-',fillstyle="none")
# title = '$\mathbb{Z}_Q$ Berry Phase'
# if indep == 'Alpha' and val <= 1:
#     x_label = r'$\alpha$'
#     letter = r'\lambda'
#     q = np.round(val,2)
# elif indep == 'Alpha' and val > 1:
#     x_label = r'$\alpha$'
#     letter = r't'
#     q = np.round(2-val,2)
# elif indep == 'Lambda' and val <= 1:
#     x_label = r'$\lambda$'
#     letter = r'\alpha'
#     q = np.round(val,2)
# else:
#     x_label = r'$\lambda$'
#     letter = r'\beta'
#     q = np.round(2-val,2)
# ax.set_title(rf'{title}: $ {letter}= {q}, N = {N},$ it $= {iterations}$')
# ax.set_xlabel(x_label)
# ax.set_ylabel(r'$\gamma / 2\pi$')
# # ax.legend({r'$\mathbb{Z}_6$', r'$\mathbb{Z}_2$'})
# for i in range(len(x_vals)):
#     if (min(alphs)-0.1) <= x_vals[i] <= (max(alphs)+0.1):
#         ax.axvline(x_vals[i],ls='--', c='gray')

# def format_func(value, tick_number):
#     if value <= 1:
#         return f'{value}'
#     else:
#         v = np.round(2 - value, 3)
#         part1 = r'$\frac{1}{'
#         part2 = r'}$'
#         return fr'{part1}{v}{part2}'
# ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

fig_path = f"{path_zq}/diagram_N{N}_iter{iterations}"
fig1.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')

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

# def format_func(value):
#     if value <= 1:
#         return f'{value}'
#     else:
#         v = np.round(2 - value, 3)
#         part1 = r'$\frac{1}{'
#         part2 = r'}$'
#         return fr'{part1}{v}{part2}'

# fig, ax = plt.subplots()
# ax.semilogx(it, zq_phases[:,0],'bo-',fillstyle="none")
# ax.semilogx(it, zq_phases[:,1],'k^-',fillstyle="none")
# title = '$\mathbb{Z}_Q$ Berry Phase'
# if b < 1:
# a = 2 -b
# if t < 1:
# l = 2-t

# a_val = format_func(a)
# l_val = format_func(l)
# ax.set_title(rf'{title}: $ \alpha = {a_val}, \lambda = {l_val}, N = {N}$')
# ax.set_xlabel("iterations")
# ax.set_ylabel(r'$\gamma / 2\pi$')
# ax.legend({r'$\mathbb{Z}_6$', r'$\mathbb{Z}_2$'})

# fig_path = f"{path_zq}/a{a}_l{l}_N{N}"
# fig.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')

# N_or_res = "res"
# Nphase = 600
# path_phasediagram = "output/phasediagram/periodic"

# x_mesh = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_xmesh")
# y_mesh = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_ymesh")

# x = np.zeros((1,1))
# x[0,0]=np.NaN
# y = np.zeros((1,1))
# y[0,0]=np.NaN


# m = 1
# num = np.array([0])
# for i in range(1201):
#     x_vals = x_mesh[:,i]
#     y_vals = y_mesh[:,i]
    
#     x_vals = x_vals[~np.isnan(x_vals)]
#     y_vals = y_vals[~np.isnan(y_vals)]

#     n = np.shape(x_vals)[0]
#     if n == m:
#         x_row = np.empty((np.shape(x)[0],1))
#         x_row[:] = np.NaN
#         y_row = np.empty((np.shape(y)[0],1))
#         y_row[:] = np.NaN

#         for j in range(len(num)):
#             x_row[num[j],0] = x_vals[j]
#             y_row[num[j],0] = y_vals[j]
        
#         x = np.append(x, x_row, axis=1)
#         y = np.append(y, y_row, axis=1)
#         # print(np.shape(x))

#     if (n != m and n > 0):
#         max = num[-1]
#         for j in range(n):
#             if j == 0:
#                 num = np.array([max+1])
#             else:
#                 num = np.append(num, num[-1]+1)

#         m = n
        
#         x_col =  np.empty((m,np.shape(x)[1]))
#         x_col[:] = np.NaN
        
#         x = np.append(x, x_col, axis=0)
#         y = np.append(y, x_col, axis=0)
#         # print(np.shape(x))

#         x_row = np.empty((np.shape(x)[0],1))
#         x_row[:] = np.NaN
#         y_row = np.empty((np.shape(y)[0],1))
#         y_row[:] = np.NaN

#         for j in range(len(num)):
#             x_row[num[j],0] = x_vals[j]
#             y_row[num[j],0] = y_vals[j]

#         x = np.append(x, x_row, axis=1)
#         y = np.append(y, y_row, axis=1)
#         # print(np.shape(x))

# x_row = np.empty((np.shape(x)[0],1))
# x_row[:] = np.NaN
# y_row = np.empty((np.shape(x)[0],1))
# y_row[:,0] = y[:,-1]

# for j in range(5):
#     x_row[-1-j,0] = 2

# x = np.append(x, x_row, axis=1)
# y = np.append(y, y_row, axis=1)
# y[-3,-1] = 1
# y[-4,-1] = 1

# print(x[-20:,-5:])
# print(y[-20:,-5:])

# fig, ax = plt.subplots()
# for i in range(np.shape(x)[0]):
#     plt.plot(x[i,:],y[i,:],c='k',lw=0.75)
# # plt.pcolormesh(x,x,gap, norm = colors.LogNorm(), cmap='inferno')
# # plt.scatter((2-0.693),(2-0.466),linewidth=0.1,marker='x')
# plt.title("Phase Boundary Diagram")
# ax.grid(linestyle='--')
# # ax.set_xlim([0,0.4])
# ax.set_aspect(1)

# labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, r'$\frac{1}{0.8}$',r'$\frac{1}{0.6}$',r'$\frac{1}{0.4}$',r'$\frac{1}{0.2}$',r'$\infty$']
# locs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0,1.2,1.4,1.6,1.8,2.0]
# ax.set_yticklabels(labels)
# ax.set_yticks(locs)
# ax.set_ylabel(r'$\alpha$')
# ax.set_xlim((0,2))
# ax.set_ylim((0,2))

# ax.set_xticklabels(labels)
# ax.set_xticks(locs)
# ax.set_xlabel(r'$\lambda$')

# fig.tight_layout()
# fig.savefig(f"{path}/{N_or_res}{N}_diagram_yo.png", dpi=500,bbox_inches='tight')