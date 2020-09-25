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


res = 5
points = 250

r_vals = np.arange(points)
r1,r2 = np.meshgrid(r_vals,r_vals)
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
    print(energy)
    return energy

def reduced_zone2(parameter,n,m):
    kpoint = parameter*b1
    energy = hamil(n,m,kpoint)
    return energy

n=2
m=2
result = optimise.minimize(reduced_zone1,1/3,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
print(result)

fig = plt.figure()
x = np.linspace(1,100,num=100)
x = 2*x/300
y_vals = np.zeros(100)
for i in range(100):
    y_vals[i] = reduced_zone1(x[i],2,2)
plt.plot(x,y_vals)
fig.savefig(f"{path}/reduced1.png", dpi=300)