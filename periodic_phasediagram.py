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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

path = "output/phasediagram/periodic"
if not os.path.exists(path):
            os.makedirs(path)

res = 600
# points = 250

run = 2*res + 1

# r_vals = np.arange(points)
# r1,r2 = np.meshgrid(r_vals,r_vals)
b1 = np.array([0,4*np.pi*np.sqrt(3)/9])
b2 = (2*np.pi/9)*np.array([3,np.sqrt(3)])
# kx = (b1[0]*r1 + b2[0]*r2)/points
# ky = (b1[1]*r1 + b2[1]*r2)/points

s_vals = np.linspace(0,1,num=res+1)
ones = np.ones(res)
up = np.append(s_vals, ones)
down = np.append(ones, np.flipud(s_vals))
l,a = np.meshgrid(up,up)
t,b = np.meshgrid(down,down)

# l_min = 0.5
# l_max = 1.0

# lvals = np.linspace(l_min,l_max,num=res)

# b_min = 0
# b_max = 0.5

# bvals = np.linspace(b_min,b_max,num=res)

# l,b = np.meshgrid(lvals, np.flipud(bvals))
# ones = np.ones((res,res))
# a = ones
# t = ones

def hamil(n,m,kvals):
    if kvals.size != 2:
        kx = kvals[:,0]
        ky = kvals[:,1]
    else:
        kx = kvals[0]
        ky = kvals[1]

    hamiltonians = np.zeros((kx.size,6,6), dtype=complex)
    phi = np.pi/2

    hamiltonians[:,1,0] = t[n,m]*b[n,m]
    hamiltonians[:,5,0] = t[n,m]*b[n,m]
    hamiltonians[:,2,1] = t[n,m]*b[n,m]
    hamiltonians[:,3,2] = t[n,m]*b[n,m]
    hamiltonians[:,4,3] = t[n,m]*b[n,m]
    hamiltonians[:,5,4] = t[n,m]*b[n,m]

    hamiltonians[:,2,0] = l[n,m]*np.exp(-1j*phi)*(b[n,m]+a[n,m]*(np.exp(3*1j*kx)+np.exp(1.5*1j*(kx+ky*np.sqrt(3)))))
    hamiltonians[:,3,0] = t[n,m]*a[n,m]*np.exp(1.5*1j*(kx+ky*np.sqrt(3)))
    hamiltonians[:,4,0] = l[n,m]*np.exp(1j*phi)*(b[n,m]+a[n,m]*(np.exp(1.5*1j*(kx+ky*np.sqrt(3)))+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))

    hamiltonians[:,3,1] = l[n,m]*np.exp(-1j*phi)*(b[n,m]+a[n,m]*(np.exp(1.5*1j*(kx+ky*np.sqrt(3)))+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
    hamiltonians[:,4,1] = t[n,m]*a[n,m]*np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))
    hamiltonians[:,5,1] = l[n,m]*np.exp(1j*phi)*(b[n,m]+a[n,m]*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))

    hamiltonians[:,4,2] = l[n,m]*np.exp(-1j*phi)*(b[n,m]+a[n,m]*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
    hamiltonians[:,5,2] = t[n,m]*a[n,m]*np.exp(-3*1j*kx)

    hamiltonians[:,5,3] = l[n,m]*np.exp(-1j*phi)*(b[n,m]+a[n,m]*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx+ky*np.sqrt(3)))))
    hamiltonians = hamiltonians + np.conjugate(np.swapaxes(hamiltonians,1,2))
    evals = np.abs(np.linalg.eigvalsh(hamiltonians))
    return np.amin(evals, axis=1)

def reduced_zone1(parameter,n,m):
    kpoint = parameter*(b1+b2)
    energy = hamil(n,m,kpoint)
    return energy

def reduced_zone2(parameter,n,m):
    kpoint = parameter*b1
    energy = hamil(n,m,kpoint)
    return energy

# fig = plt.figure()
# x = np.linspace(1,100,num=100)
# x = 2*x/300
# y_vals = np.zeros(100)
# for i in range(100):
#     y_vals[i] = reduced_zone1(x[i],2,2)
# plt.plot(x,y_vals)
# fig.savefig(f"{path}/reduced1.png", dpi=300)

def min_en(n,m):
    Kpoint = (1/3)*(b1+b2)
    Kdashpoint = (1/3)*(b1+b2)
    Mpoint = (1/2)*b1
    Gamma = np.array([0,0])
    energies = hamil(n,m,np.array([Kdashpoint, Kpoint,Gamma, Mpoint]))
    min_energy = np.amin(energies)
    lam_val = np.argmin(energies)
    if lam_val ==3:
        lam = 2+np.sqrt(3)/2
    else:
        lam = lam_val
    
    if min_energy != 0:
        result = optimise.minimize(reduced_zone1,0.1,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun[0]
            lam = 3*((2/3) - result.x)
        result = optimise.minimize(reduced_zone1,0.2,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun[0]
            lam = 3*((2/3) - result.x)
        result = optimise.minimize(reduced_zone1,7/18,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun[0]
            lam = 3*((2/3) - result.x)
        result = optimise.minimize(reduced_zone1,4/9,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun[0]
            lam = 3*((2/3) - result.x)
        result = optimise.minimize(reduced_zone1,5/9,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun[0]
            lam = 3*((2/3) - result.x)
        result = optimise.minimize(reduced_zone1,11/18,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun[0]
            lam = 3*((2/3) - result.x)

        result = optimise.minimize(reduced_zone2,0.15,args=(n,m), bounds=[(0,1/2)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun
            lam = 2 + np.sqrt(3)*result.x
        result = optimise.minimize(reduced_zone2,0.35,args=(n,m), bounds=[(0,1/2)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun
            lam = 2 + np.sqrt(3)*result.x
        
    return min_energy, lam 

def check_eval(n,m):
    if (l[n,m]<=0.6) and (b[n,m]==1) and (a[n,m] >= (-3/5)*l[n,m] + 0.3) and (a[n,m] <= -1.5*(l[n,m]-0.1)+1):
        check = True
    elif (l[n,m]>0.6) and (b[n,m] == 1) and (t[n,m]==1) and (a[n,m] >= l[n,m] - 0.75) and (a[n,m] <= (5/8)*l[n,m]-1/8):
        check = True
    elif (t[n,m]<0.1) or ((l[n,m]==1) and (a[n,m] >= -(2/9)*(t[n,m]-0.1)+0.45) and (a[n,m] <= -(5/9)*(t[n,m]-1)+0.5)):
        check = True
    elif (a[n,m]==1) and (l[n,m] ==1) and ((b[n,m] >= -(1/6)*(t[n,m]-1)+0.3) and (b[n,m] <= -(5/9)*(t[n,m]-1)+0.5)):
        check = True
    elif (l[n,m] >= 0.6) and (t[n,m]==1) and (b[n,m] >= 0.75*(l[n,m]-1) + 0.3) and (b[n,m] <= 0.5*(l[n,m]-1)+0.5):
        check = True
    elif (0.3 <= l[n,m] < 0.6) and (b[n,m] <= (6/7)*(l[n,m]-0.25)):
        check = True
    elif (l[n,m] < 0.4) and (a[n,m]==1) and (b[n,m] >= (-5/3)*(l[n,m] -0.2)) and (b[n,m] <= (-61/21)*(l[n,m]-0.1)+1):
        check = True
    else:
        check = False
    return check 

gap = np.zeros((run,run))
kgap = np.zeros((run,run))
for n in range(0,run):
    for m in range(0,run):

        print(f"{n*(run)+m}/{(run)**2}", end='\r')

        check = check_eval(n,m)
        if check == True:
            gap[n,m], kgap[n,m] = min_en(n,m)
        if check == False:
            gap[n,m] = np.NaN
            kgap[n,m] = np.NaN

x = np.linspace(0,2,num=run)
gap[gap>0.01]= np.NaN
joblib.dump(gap, f"{path}/res{res}_gap_topleft")
joblib.dump(kgap, f"{path}/res{res}_kgap_topleft")
joblib.dump(x, f"{path}/res{res}_x")

# fig = plt.figure()
# plt.pcolormesh(x, x, gap, norm = colors.LogNorm(), cmap='inferno')
# fig.savefig(f"{path}/periodictopleft.png", dpi=500)

# vmax = (2 + np.sqrt(3)/2)
# kgap[gap>0.01] = np.NaN
# cmap1 = colors.LinearSegmentedColormap.from_list('mycmap', [(0/vmax,    '#984ea3'), (0.5/vmax,    '#e41a1c'), (1/vmax, '#4daf4a'), (2/vmax, '#377eb8'), ((2 + np.sqrt(3)/2)/vmax,    '#e41a1c')], N=256)
# fig1, ax = plt.subplots()
# im = ax.pcolormesh(x,x,kgap, cmap=cmap1, vmin=0, vmax=vmax)
# cbar = fig1.colorbar(im)
# ax.set(aspect=1)
# cbar.set_ticks([0,0.5,1,2,vmax]) # Integer colorbar tick locations
# cbar.set_ticklabels(["K\'", "M", "K", "$\Gamma$","M"])
# fig1.savefig(f"{path}/kpointstopleft.png", dpi=500)