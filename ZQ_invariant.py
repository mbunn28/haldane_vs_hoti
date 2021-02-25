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

it = np.array([4])
#150,200,250])
# it = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#,11,12,13,14,15,16,17,18,19,20])
#,120,140,160,180,200,250,300,350,400,450,500,600,700,800,900,1000])
zq_phases = np.zeros((len(it),2))
for m in tqdm(range(len(it))):
    indep = 'Alpha'
    val = 0
    num = 1
    iterations = it[m]
    location = np.array([4,4], dtype=int)
    N = 6
    max_x = 0.1
    min_x = 0.1

    def rule(n):
        grad = (max_x-min_x)/num
        y = grad*n+min_x
        if y <= 1:
            a = y
            b = 1
        elif 1 < y < 2:
            a = 1
            b = 2 - y
        else:
            print('error defining params!')
            a = 0
            b = 0

        if a > 1 or b > 1:
            print('params > 1')

        return a, b, y

    # N_or_res = "res"
    # Nphase = 600
    # path_phasediagram = "output/phasediagram/periodic"

    # x = np.linspace(0,2,num=2*Nphase+1)
    # x_mesh = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_xmesh")
    # y_mesh = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_ymesh")

    # j = min(range(len(x)), key=lambda i: abs(x[i]-val))
    # if indep == 'Lambda':
    #     x_vals = x_mesh[j,:]
    # else:
    #     x_vals  = y_mesh[:,j]
    # x_vals = x_vals[~np.isnan(x_vals)]
    # print(x_vals)

    zq = ['z6', 'z2']
    # zq_phases = np.zeros((num,len(zq)))
    alphs = np.zeros(num)
    M = int((N**2))
    phi = np.random.rand(6*M,M)
    phi = scipy.linalg.orth(phi)
    for n in range(num):
        if indep == 'Alpha':
            if 0 <= val <= 1:
                l = val
                t = 1

                a, b, alphs[n] = rule(n)
            
            else:
                t = 2 - val
                l = 1
                a, b, alphs[n] = rule(n)
        
        else:
            if 0 <= val <= 1:
                a = val
                b = 1

                l, t, alphs[n] = rule(n)
            
            else:
                b = 2 - val
                a = 1

                l, t, alphs[n] = rule(n)
        
        # print(f'a = {a}, b= {b}, t={t}, l={l}')
        
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
            
            # print(f'lattice1: a = {lattice1.a}, b= {lattice1.b}, t={lattice1.t}, l={lattice1.l}')
            
            D = 1
            energies = np.zeros((6*(N**2),iterations+1))
            ui = np.zeros(iterations,dtype=complex)

            lattice1.twist_hamiltonian()
            energies[:,0] = lattice1.energies
            evecs = lattice1.waves
            singlestates_a = evecs[:,:M]
            pa = np.einsum('ij,jk',singlestates_a,np.conjugate(singlestates_a.transpose()))
            lattice1.proj = np.einsum('ij,jk',pa,phi)

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
                # print(f'lattice1: a = {lattice2.a}, b= {lattice2.b}, t={lattice2.t}, l={lattice2.l}')

                lattice2.twist_hamiltonian()
                energies[:,i+1] = lattice2.energies
                evecs = lattice2.waves
                singlestates_b = evecs[:,:M]
                pb = np.einsum('ij,jk',singlestates_b,np.conjugate(singlestates_b.transpose()))
                lattice2.proj = np.einsum('ij,jk',pb,phi)
                Di = np.einsum('ij,jk',np.conjugate(lattice1.proj.transpose()),lattice2.proj)
                det_Di = numpy.linalg.slogdet(Di)
                if det_Di == 0:
                    print('error! det zero!\n')
                ui[i] = det_Di[0]
                D = D*ui[i]
                
                Nphi = np.einsum('ij,jk',np.conjugate(lattice2.proj.transpose()), lattice2.proj)
                _, det_Nphi = numpy.linalg.slogdet(Nphi)
                if det_Nphi == -np.Inf:
                    print('The overlap matrix det = 0!')
                
                lattice1 = lattice2

            zq_phase = np.angle(D)
            if zq_phase < -1e-1:
                zq_phase = zq_phase + 2*np.pi
            zq_phase = zq_phase/(2*np.pi)
            zq_phases[m,j] = zq_phase



    # print(zq_phases)
    # fig, ax = plt.subplots()
    # ax.plot(alphs, zq_phases[:,0],'bo-',fillstyle="none")
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

    # fig_path = f"{path_zq}/{indep}{val}_N{N}_iter{iterations}"
    # fig.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')

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

def format_func(value):
    if value <= 1:
        return f'{value}'
    else:
        v = np.round(2 - value, 3)
        part1 = r'\frac{1}{'
        part2 = r'}'
        return fr'{part1}{v}{part2}'

fig, ax = plt.subplots()
ax.semilogx(it, zq_phases[:,0],'bo-',fillstyle="none", label=r'$\mathbb{Z}_6$')
ax.semilogx(it, zq_phases[:,1],'k^-',fillstyle="none", label=r'$\mathbb{Z}_2$')
title = '$\mathbb{Z}_Q$ Berry Phase'
if b < 1:
    a = 2 -b
if t < 1:
    l = 2-t
    
a_val = format_func(a)
l_val = format_func(l)
ax.set_title(rf'{title}: $ \alpha = {a_val}, \lambda = {l_val}, $it$ = {iterations}$')
ax.set_xlabel(r"$N$")
ax.set_ylabel(r'$\gamma / 2\pi$')
ax.legend()

fig_path = f"{path_zq}/a{a}_l{l}_N{N}"
fig.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')