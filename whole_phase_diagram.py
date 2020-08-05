#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

lattice = ti.Lattice(
PBC_i = True,
PBC_j = True,
Corners = False,
alpha = 0,
hal = 0,
M=0,
N=14)

t=250
gap = np.zeros((2*t+1,2*t+1))
hal_val = np.zeros(2*t+1)
alph_val = np.zeros(2*t+1)

for k in range(0,t+1):
    
    lattice.hal = k/t
    hal_val[k] = lattice.hal
    if k != t:
        hal_val[2*t-k] = 2-lattice.hal

    for m in range(0,t+1):

        print(f"{k*(t+1)+m}/{(t+1)**2}", end='\r')

        lattice.alpha = m/t
        alph_val[m] = lattice.alpha
        if m != t:
           alph_val[2*t-m] = 2 - lattice.alpha

        if (lattice.hal<=0.6) and (lattice.alpha >= (-3/5)*lattice.hal + 0.3) and (lattice.alpha <= -1.5*(lattice.hal-0.1)+1):
            gap[k,m] = lattice.min_energy()
        elif (lattice.hal>0.6) and (lattice.alpha >= lattice.hal - 0.7) and (lattice.alpha <= (5/8)*lattice.hal-1/8):
            gap[k,m] = lattice.min_energy()
        else:
            gap[k,m] = np.NaN
        
        lattice.large_hal = True

        if k != t:
            if (lattice.hal<0.1) or ((lattice.alpha >= -(2/9)*(lattice.hal-0.1)+0.5) and (lattice.alpha <= -(2/9)*(lattice.hal-1)+0.5)):
                gap[2*t-k,m] = lattice.min_energy()
            else:
                gap[2*t-k,m] = np.NaN

        lattice.large_alpha = True

        if k != t and m != t:     
            if (lattice.hal<0.1) or ((lattice.alpha >= -(1/6)*(lattice.hal-1)+0.3) and (lattice.alpha <= -(5/9)*(lattice.hal-1)+0.5)):
                gap[2*t-k,2*t-m] = lattice.min_energy()
            else:
                gap[2*t-k,2*t-m] = np.NaN

        lattice.large_hal = False

        if m != t:
            if (lattice.hal >= 0.6) and (lattice.alpha >= 0.75*(lattice.hal-1) + 0.3) and (lattice.alpha <= 0.5*(lattice.hal-1)+0.5):
                gap[k,2*t-m] = lattice.min_energy()
            elif (0.3 <= lattice.hal < 0.6) and (lattice.alpha <= (6/7)*(lattice.hal-0.25)):
                gap[k,2*t-m] = lattice.min_energy()
            elif (lattice.hal < 0.4) and (lattice.alpha >= (-5/3)*(lattice.hal -0.2)) and (lattice.alpha <= (-61/21)*(lattice.hal-0.1)+1):
                gap[k,2*t-m] = lattice.min_energy()
            else:
                gap[k,2*t-m] = np.NaN

path = "output/phasediagram"
if not os.path.exists(path):
            os.makedirs(path)

joblib.dump(gap, f"{path}/N{lattice.N}_gap")
joblib.dump(hal_val, f"{path}/N{lattice.N}_hal_val")
joblib.dump(alph_val, f"{path}/N{lattice.N}_alph_val")
# print(gap)
# print(hal_val)
# print(alph_val)