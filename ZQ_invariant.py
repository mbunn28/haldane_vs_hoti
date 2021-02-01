#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble= r'\usepackage{amsfonts}')

path = "output/zq"
if not os.path.exists(path):
            os.makedirs(path)

N_or_res = "res"
Nphase = 600
path_phasediagram = "output/phasediagram/periodic"

gap = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_gap")
x = np.linspace(0,2,num=1201)
gapmask = gap < 1e-2
j = min(range(len(x)), key=lambda i: abs(x[i]-0.1))
x[~gapmask[j,:]] = np.NaN
x = x[~np.isnan(x)]
gap_vals = gap[j,:]
gap_vals[~gapmask[j,:]] = np.NaN
gap_vals = gap_vals[~np.isnan(gap_vals)]
print(gap_vals)
x = x[argrelextrema(gap_vals, np.less)[0]]
gap_vals = gap_vals[argrelextrema(gap_vals, np.less)[0]]
print(gap_vals)
for i in range(len(x)):
    if 2 > x[i] > 1:
        x[i] = 2 - x[i]
        x[i] = 1/x[i]

# x = np.round(x,2)
# x = np.delete(x, np.argwhere(np.ediff1d(x) <= 0.02) + 1) 
# x = np.unique(x)
print(x)

num = 20
z6_phases = np.zeros(num)
z2_phases = np.zeros(num)
alphs = np.zeros(num)
for n in tqdm(range(num)):
    l = 0.2
    t = 1

    a = x[1] + 0.1/(2**n)
    alphs[n] = a
    b = 1

    N = 5
    mass = 0

    PBC_i = True
    PBC_j = True
    Corners = False

    location = np.array([4,4], dtype=int)
    iterations = 100 

    def lat(i,j,s): return(6*N*i+6*j+s)

    def twist_hamiltonian(the, zq):
        vv = 100000
        h = np.zeros((6*(N)**2,6*(N)**2), dtype = complex)
        
        for i in range(N):
            for j in range(N):
                if (i == location[0] and j == location[1] and zq == 'z6'):
                    the5 = -np.sum(the)
                    h[lat(i,j,0), lat(i,j,1)] = -t*b*np.exp(-1j*the[1])
                    h[lat(i,j,0), lat(i,j,5)] = -t*b*np.exp(1j*the[0])
                    h[lat(i,j,0), lat((i+1)%N,j,3)] = -t*a

                    h[lat(i,j,2), lat(i,j,1)] = -t*b*np.exp(1j*the[2])
                    h[lat(i,j,2), lat(i,j,3)] = -t*b*np.exp(-1j*the[3])
                    h[lat(i,j,2), lat(i,(j+1)%N,5)] = -t*a

                    h[lat(i,j,4), lat(i,j,3)] = -t*b*np.exp(1j*the[4])
                    h[lat(i,j,4), lat(i,j,5)] = -t*b*np.exp(-1j*the5)
                    h[lat(i,j,4), lat((i-1)%N,(j-1)%N,1)] = -t*a

                    h[lat(i,j,0), lat(i,j,4)] = -1j*l*b*np.exp(-1j*(the[4]+the[3]+the[2]+the[1]))
                    h[lat(i,j,1), lat(i,j,5)] = -1j*l*b*np.exp(1j*(the[1]+the[0]))
                    h[lat(i,j,2), lat(i,j,0)] = -1j*l*b*np.exp(1j*(the[2]+the[1]))
                    h[lat(i,j,3), lat(i,j,1)] = -1j*l*b*np.exp(1j*(the[3]+the[2]))
                    h[lat(i,j,4), lat(i,j,2)] = -1j*l*b*np.exp(1j*(the[4]+the[3]))
                    h[lat(i,j,5), lat(i,j,3)] = -1j*l*b*np.exp(-1j*(the[3]+the[2]+the[1]+the[0]))
                
                elif (i == location[0] and j == location[1] and zq == 'z2'):
                    h[lat(i,j,0), lat(i,j,1)] = -t*b
                    h[lat(i,j,0), lat(i,j,5)] = -t*b
                    h[lat(i,j,0), lat((i+1)%N,j,3)] = -t*a

                    h[lat(i,j,2), lat(i,j,1)] = -t*b
                    h[lat(i,j,2), lat(i,j,3)] = -t*b
                    h[lat(i,j,2), lat(i,(j+1)%N,5)] = -t*a*np.exp(1j*the)

                    h[lat(i,j,4), lat(i,j,3)] = -t*b
                    h[lat(i,j,4), lat(i,j,5)] = -t*b
                    h[lat(i,j,4), lat((i-1)%N,(j-1)%N,1)] = -t*a

                    h[lat(i,j,0), lat(i,j,4)] = -1j*l*b
                    h[lat(i,j,1), lat(i,j,5)] = -1j*l*b
                    h[lat(i,j,2), lat(i,j,0)] = -1j*l*b
                    h[lat(i,j,3), lat(i,j,1)] = -1j*l*b
                    h[lat(i,j,4), lat(i,j,2)] = -1j*l*b
                    h[lat(i,j,5), lat(i,j,3)] = -1j*l*b

                else:
                    h[lat(i,j,0), lat(i,j,1)] = -t*b
                    h[lat(i,j,0), lat(i,j,5)] = -t*b
                    h[lat(i,j,0), lat((i+1)%N,j,3)] = -t*a

                    h[lat(i,j,2), lat(i,j,1)] = -t*b
                    h[lat(i,j,2), lat(i,j,3)] = -t*b
                    h[lat(i,j,2), lat(i,(j+1)%N,5)] = -t*a

                    h[lat(i,j,4), lat(i,j,3)] = -t*b
                    h[lat(i,j,4), lat(i,j,5)] = -t*b
                    h[lat(i,j,4), lat((i-1)%N,(j-1)%N,1)] = -t*a

                    h[lat(i,j,0), lat(i,j,4)] = -1j*l*b
                    h[lat(i,j,1), lat(i,j,5)] = -1j*l*b
                    h[lat(i,j,2), lat(i,j,0)] = -1j*l*b
                    h[lat(i,j,3), lat(i,j,1)] = -1j*l*b
                    h[lat(i,j,4), lat(i,j,2)] = -1j*l*b
                    h[lat(i,j,5), lat(i,j,3)] = -1j*l*b

                if N !=1:
                    h[lat(i,j,0), lat((i+1)%N,j,4)] = -1j*l*a
                    h[lat(i,j,0), lat((i+1)%N,(j+1)%N,4)] = -1j*l*a

                    h[lat(i,j,1), lat((i+1)%N,(j+1)%N,5)] = -1j*l*a
                    h[lat(i,j,1), lat(i,(j+1)%N,5)] = -1j*l*a

                    h[lat(i,j,2), lat(i,(j+1)%N,0)] = -1j*l*a
                    h[lat(i,j,2), lat((i-1)%N,j,0)] = -1j*l*a

                    h[lat(i,j,3), lat((i-1)%N,j,1)] = -1j*l*a
                    h[lat(i,j,3), lat((i-1)%N,(j-1)%N,1)] = -1j*l*a

                    h[lat(i,j,4), lat((i-1)%N,(j-1)%N,2)] = -1j*l*a
                    h[lat(i,j,4), lat(i,(j-1)%N,2)] = -1j*l*a

                    h[lat(i,j,5), lat(i,(j-1)%N,3)] = -1j*l*a
                    h[lat(i,j,5), lat((i+1)%N,j,3)] = -1j*l*a

                for s in [0,2,4]:
                    h[lat(i,j,s), lat(i,j,s)] = +mass/2
                for s in [1,3,5]:
                    h[lat(i,j,s), lat(i,j,s)] = -mass/2



        if PBC_i == False:
            for j in range(N):
                h[lat(N-1,j,0), lat(0,j,3)] = 0
                h[lat(0,j,4), lat(N-1,(j-1)%N,1)] = 0

                h[lat(0,(j+1)%N,3), lat(N-1,j,1)] = 0
                h[lat(N-1,j,0), lat(0,j,4)] = 0
                h[lat(0,(j+1)%N,4), lat(N-1,j,2)] = 0
                h[lat(N-1,j,5), lat(0,j,3)] = 0

                h[lat(N-1,j,1), lat(0,(j+1)%N,5)] = 0
                h[lat(N-1,j,0), lat(0,(j+1)%N,4)] = 0
                h[lat(0,j,3), lat(N-1,j,1)] = 0
                h[lat(0,j,2), lat(N-1,j,0)] = 0

        if PBC_j == False:
            for i in range(N):
                h[lat(i,N-1,2), lat(i,0,5)] = 0
                h[lat(i,0,4), lat((i-1)%N,N-1,1)] = 0

                h[lat((i+1)%N,0,3), lat(i,N-1,1)] = 0
                h[lat(i,N-1,1), lat(i,0,5)] = 0
                h[lat((i+1)%N,0,4), lat(i,N-1,2)] = 0
                h[lat(i,N-1,2), lat(i,0,0)] = 0

                h[lat(i,N-1,1), lat((i+1)%N,0,5)] = 0
                h[lat(i,0,5), lat(i,N-1,3)] = 0
                h[lat(i,N-1,0), lat((i+1)%N,0,4)] = 0
                h[lat(i,0,4), lat(i,N-1,2)] = 0

        #dimer geometry
        if PBC_j == False and Corners == True:
            for i in range(0,N):
                for s in [0,3,4,5]:
                    h[lat(i,0,s),lat(i,0,s)] = vv

                for s in [0,1,2,3]:
                    h[lat(i,N-1,s),lat(i,N-1,s)] = vv


        if PBC_i==False and Corners == True:
            for j in range(0,N):
                for s in [2,3,4,5]:
                    h[lat(0,j,s),lat(0,j,s)] = vv

                for s in [0,1,2,5]:
                    h[lat(N-1,j,s),lat(N-1,j,s)] = vv

        h = np.conjugate(h.transpose()) + h

        ener, evecs = scipy.linalg.eigh(h)
        if (ener[3*(N**2)]-ener[3*(N**2)-1]) < 1e-3:
            print('energy very small!\n')
        return ener, evecs

    def u(the1,the2,n):
        evecs1 = twist_hamiltonian(the1)
        evecs2 = twist_hamiltonian(the2)
        res = np.vdot(evecs1[n,:],evecs2[n,:])
        res = res/np.abs(res)
        return res

    def curve(tau):
        CoG = (2*np.pi/6)*np.array([1,1,1,1,1])
        # e0 = np.zeros(6,dtype=float)
        e1 = np.array([2*np.pi,0,0,0,0])
        # e2 = np.array([0,2*np.pi,0,0,0])
        # e3 = np.array([0,0,2*np.pi,0,0])
        # e4 = np.array([0,0,0,2*np.pi,0])
        # e5 = np.array([0,0,0,0,2*np.pi])

        if tau <= 0.5:
            the = 2*tau*CoG
        elif tau <= 1:
            the = 2*(1-tau)*CoG+(2*tau-1)*e1
        else:
            print("error! poorly parameterised curve")

        return the

    M = int(3*(N**2))
    # D = np.eye(M, dtype=complex)
    D = 1
    energies = np.zeros((6*(N**2),iterations+1))
    ui = np.zeros(iterations,dtype=complex)
    energies[:,0], evecs = twist_hamiltonian(curve(0),zq='z6')
    singlestates_a = evecs[:,:M]
    pa = np.matmul(singlestates_a,np.conjugate(singlestates_a.transpose()))
    phi = np.random.rand(2*M,M)
    phi = scipy.linalg.orth(phi)
    ua = np.matmul(pa,phi)
    for i in range(iterations):
        # print(f"{iterations*(2*n+1)+i+1}/{2*iterations*num}", end='\r')

        energies[:,i+1],evecs = twist_hamiltonian(curve((i+1)/iterations),zq='z6')
        singlestates_b = evecs[:,:M]
        pb = np.matmul(singlestates_b,np.conjugate(singlestates_b.transpose()))
        ub = np.matmul(pb,phi)
        Di = np.matmul(np.conjugate(ua.transpose()),ub)
        det_Di = numpy.linalg.slogdet(Di)
        # print(det_Di)
        if det_Di == 0:
            print('error! det zero!\n')
        # U_i = det_Di/np.abs(det_Di)
        # print(U_i)
        ui[i] = det_Di[0]
        D = D*ui[i]
        # print(D)
        ua = ub

        Nphi = np.matmul(np.conjugate(ub.transpose()), ub)
        det_Nphi = scipy.linalg.det(Nphi)
        if det_Nphi == 0:
            print('The overlap matrix det = 0!')

    z6_phase = np.angle(D)
    if z6_phase < -1e-1:
        z6_phase = z6_phase + 2*np.pi
    z6_phase = z6_phase/(2*np.pi)
    z6_phases[n] = z6_phase

    D = 1
    _, evecs = twist_hamiltonian(0,zq='z2')
    singlestates_a = evecs[:,:M]
    pa = np.matmul(singlestates_a,np.conjugate(singlestates_a.transpose()))
    ua = np.matmul(pa,phi)
    for i in range(iterations):
        # print(f"{iterations*(2*n+1)+i+1}/{2*iterations*num}", end='\r')

        theta = 2*np.pi*(i+1)/iterations
        _, evecs = twist_hamiltonian(theta,zq='z2')
        singlestates_b = evecs[:,:M]
        pb = np.matmul(singlestates_b,np.conjugate(singlestates_b.transpose()))
        ub = np.matmul(pb,phi)
        Di = np.matmul(np.conjugate(ua.transpose()),ub)
        det_Di = numpy.linalg.slogdet(Di)
        U_i = det_Di[0]
        D = D*U_i
        ua = ub

        z2_phase = np.angle(D)
        if z2_phase < -1e-3:
            z2_phase = z2_phase + 2*np.pi
        z2_phase = z2_phase/(2*np.pi)
        z2_phases[n] = z2_phase

    ui_angles = np.angle(ui)
    # print(ui_angles*np.pi)


print(z6_phases)
print(z2_phases)
fig, ax = plt.subplots()
ax.plot(alphs, z6_phases,'bo-',fillstyle="none")
ax.plot(alphs, z2_phases,'k^-',fillstyle="none")
title = '$\mathbb{Z}_Q$ Berry Phase'
ax.set_title(rf'{title}: $\lambda = {l}, N = {N},$ it $= {iterations}$')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\gamma / 2\pi$')
ax.legend({r'$\mathbb{Z}_2$', r'$\mathbb{Z}_6$'})
for i in range(len(x)):
    if (min(alphs)-0.1) <= x[i] <= (max(alphs)+0.1):
        ax.axvline(x[i],ls='--', c='gray')
fig_path = f"{path}/a{l}_N{N}_iter{iterations}"
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