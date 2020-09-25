#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.optimize as optimise


path = "output/phasediagram/periodic"
if not os.path.exists(path):
            os.makedirs(path)

res = 300
# points = 250

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
    energies = hamil(n,m,np.array([Kpoint, Kdashpoint,Mpoint,Gamma]))
    min_energy = np.amin(energies)
    if min_energy != 0:
        result = optimise.minimize(reduced_zone1,0.15,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun[0]
        result = optimise.minimize(reduced_zone1,0.5,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun[0]

        result = optimise.minimize(reduced_zone2,0.25,args=(n,m), bounds=[(0,1/2)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun
    return min_energy

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

gap = np.zeros((2*res+1,2*res+1))
for n in range(0,2*res+1):
    for m in range(0,2*res+1):

        print(f"{n*(2*res+1)+m}/{(2*res+1)**2}", end='\r')

        check = check_eval(n,m)
        if check == True:
            gap[n,m] = min_en(n,m)
        if check == False:
            gap[n,m] = np.NaN

x = np.linspace(0,2,num=2*res+1)
gap[gap>0.01]= np.NaN
joblib.dump(gap, f"{path}/res{res}_gap")
joblib.dump(x, f"{path}/res{res}_x")

fig = plt.figure()
plt.pcolormesh(x, x, gap, norm = colors.LogNorm(), cmap='inferno')
fig.savefig(f"{path}/periodic.png", dpi=500)