#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

lattice = ti.Lattice(
PBC_i = True,
PBC_j = True,
Corners = False,
alpha = 0,
hal = 0,
M=0,
N=11)

t = 250
points = 200
gap = np.zeros((2*t+1,2*t+1,3))
hal_val = np.zeros(2*t+1)
alph_val = np.zeros(2*t+1)

b1 = np.array([0,4*np.pi*np.sqrt(3)/9])
b2 = (2*np.pi/9)*np.array([3,np.sqrt(3)])
# M = np.array([0,2*np.pi*np.sqrt(3)/9])
# K1 = np.array([-0.5,2*np.pi*np.sqrt(3)/9])
# K2 = np.array([0.5,2*np.pi*np.sqrt(3)/9])
# kpoints = [M, K1, K2]

for r in range(0,points):
    for s in range(0, points):
    
        kpoint = (r*b1 + s*b2)/(points)
        print(f"{r*points+s}/{(points)**2}", end='\r')

        for k in range(0,t+1):
            
            lattice.hal = k/t
            hal_val[k] = lattice.hal
            if k != t:
                hal_val[2*t-k] = 2-lattice.hal

            for m in range(0,t+1):

                lattice.large_alpha = False

                lattice.alpha = m/t
                alph_val[m] = lattice.alpha
                if m != t:
                    alph_val[2*t-m] = 2 - lattice.alpha

                if (lattice.hal<=0.6) and (lattice.alpha >= (-3/5)*lattice.hal + 0.3) and (lattice.alpha <= -1.5*(lattice.hal-0.1)+1):
                    e = lattice.min_periodic_energy(kpoint)
                    if (r == 0 and s ==0) or e < gap[k,m,0]:
                        gap[k,m,0] = e
                        gap[k,m,1] = kpoint[0]
                        gap[k,m,2] = kpoint[1]
                elif (lattice.hal>0.6) and (lattice.alpha >= lattice.hal - 0.7) and (lattice.alpha <= (5/8)*lattice.hal-1/8):
                    e = lattice.min_periodic_energy(kpoint)
                    if (r == 0 and s ==0) or e < gap[k,m,0]:
                        gap[k,m,0] = e
                        gap[k,m,1] = kpoint[0]
                        gap[k,m,2] = kpoint[1]
                else:
                    gap[k,m,0] = np.NaN
                    gap[k,m,1] = np.NaN
                    gap[k,m,2] = np.NaN 
                
                lattice.large_hal = True

                if k != t:
                    if (lattice.hal<0.1) or ((lattice.alpha >= -(2/9)*(lattice.hal-0.1)+0.45) and (lattice.alpha <= -(5/9)*(lattice.hal-1)+0.5)):
                        e = lattice.min_periodic_energy(kpoint)
                        if (r == 0 and s == 0) or e < gap[2*t-k,m,0]:
                            gap[2*t-k,m,0] = e
                            gap[2*t-k,m,1] = kpoint[0]
                            gap[2*t-k,m,2] = kpoint[1]
                    else:
                        gap[2*t-k,m,0] = np.NaN
                        gap[2*t-k,m,1] = np.NaN
                        gap[2*t-k,m,2] = np.NaN

                lattice.large_alpha = True

                if k != t and m != t:     
                    if (lattice.hal<0.1) or ((lattice.alpha >= -(1/6)*(lattice.hal-1)+0.3) and (lattice.alpha <= -(5/9)*(lattice.hal-1)+0.5)):
                        e = lattice.min_periodic_energy(kpoint)
                        if (r == 0 and s == 0) or e < gap[2*t-k,2*t-m,0]:
                            gap[2*t-k,2*t-m,0] = e
                            gap[2*t-k,2*t-m,1] = kpoint[0]
                            gap[2*t-k,2*t-m,2] = kpoint[1]
                    else:
                        gap[2*t-k,2*t-m,0] = np.NaN
                        gap[2*t-k,2*t-m,1] = np.NaN
                        gap[2*t-k,2*t-m,2] = np.NaN

                lattice.large_hal = False

                if m != t:
                    if (lattice.hal >= 0.6) and (lattice.alpha >= 0.75*(lattice.hal-1) + 0.3) and (lattice.alpha <= 0.5*(lattice.hal-1)+0.5):
                        e = lattice.min_periodic_energy(kpoint)
                        if (r == 0 and s == 0) or e < gap[k,2*t-m,0]:
                            gap[k,2*t-m,0] = e 
                            gap[k,2*t-m,1] = kpoint[0]
                            gap[k,2*t-m,2] = kpoint[1]               
                    elif (0.3 <= lattice.hal < 0.6) and (lattice.alpha <= (6/7)*(lattice.hal-0.25)):
                        e = lattice.min_periodic_energy(kpoint)
                        if (r == 0 and s == 0) or e < gap[k,2*t-m,0]:
                            gap[k,2*t-m,0] = e
                            gap[k,2*t-m,1] = kpoint[0]
                            gap[k,2*t-m,2] = kpoint[1]   
                    elif (lattice.hal < 0.4) and (lattice.alpha >= (-5/3)*(lattice.hal -0.2)) and (lattice.alpha <= (-61/21)*(lattice.hal-0.1)+1):
                        e = lattice.min_periodic_energy(kpoint)
                        if (r == 0 and s == 0) or e < gap[k,2*t-m,0]:
                            gap[k,2*t-m,0] = e 
                            gap[k,2*t-m,1] = kpoint[0]
                            gap[k,2*t-m,2] = kpoint[1]   
                    else:
                        gap[k,2*t-m,0] = np.NaN
                        gap[k,2*t-m,1] = np.NaN
                        gap[k,2*t-m,2] = np.NaN

path = "output/phasediagram/periodic"
if not os.path.exists(path):
            os.makedirs(path)

joblib.dump(gap, f"{path}/res{t}_gap")
joblib.dump(hal_val, f"{path}/res{t}_hal_val")
joblib.dump(alph_val, f"{path}/res{t}_alph_val")
# print(gap)
# print(hal_val)
# print(alph_val)

# fig = plt.figure()
# plt.pcolormesh(hal_val, alph_val, np.transpose(gap[:,:,0]), norm = colors.LogNorm(), cmap='inferno')
# fig.savefig(f"{path}/periodic.png", dpi=1200)