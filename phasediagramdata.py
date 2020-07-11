#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os

lattice = ti.Lattice(
PBC_i = True,
PBC_j = True,
Corners = False,
alpha = 0,
hal = 0,
M=0,
N=16)

t=150
gap = np.zeros((2*t+1,t+1))
hal_val = np.zeros(2*t+1)
alph_val = np.zeros(t+1)
lattice.large_alpha = True

for k in range(0,t):
    
    lattice.hal = k/t
    hal_val[k] = lattice.hal
    hal_val[2*t-k] = 2-lattice.hal

    for m in range(0,t+1):

        print(f"{k*(t+1)+m}/{t*(t+1)}", end='\r')
        if m == t:
            lattice.alpha = 1
        else:
            lattice.alpha = m/t
            alph_val[t-m] = 2-lattice.alpha

        # lattice.large_hal= False
        # lattice.large_alpha = False
        # gap[k,m] = lattice.min_energy()

        lattice.large_hal = True
        # gap[2*t-k,m] = lattice.min_energy()

        if (lattice.hal<0.1) or ((lattice.alpha >= -(1/6)*(lattice.hal-1)+0.3) and (lattice.alpha <= -(1/9)*(lattice.hal-1)+0.5)):
            gap[2*t-k,t-m] = lattice.min_energy()
        else:
            gap[2*t-k,t-m] = np.NaN

        lattice.large_hal = False

        if (lattice.hal >= 0.6) and (lattice.alpha >= 0.75*(lattice.hal-1) + 0.3) and (lattice.alpha <= 0.75*(lattice.hal-1)+0.6):
            gap[k,t-m] = lattice.min_energy()
        elif (0.25 <= lattice.hal < 0.6) and (lattice.alpha <= (6/7)*(lattice.hal-0.25)):
            gap[k,t-m] = lattice.min_energy()
        else:
            gap[k,t-m] = np.NaN

lattice.large_hal= False
lattice.hal=1
hal_val[t]=1

for m in range(0,t):

    # lattice.large_alpha = False 
    lattice.alpha = m/t
    # gap[t,m] = lattice.min_energy()

    # lattice.large_alpha = True

    if (lattice.alpha <= 0.5) and (lattice.alpha >= 0.2):
        gap[t,t-m] = lattice.min_energy()
    else:
        gap[t,t-m] = np.NaN

lattice.hal=1
lattice.alpha=1
alph_val[0]=1
# gap[t,0] = lattice.min_energy()
gap[t,0]=np.NaN

path = "output/phasediagram"
if not os.path.exists(path):
            os.makedirs(path)

joblib.dump(gap, f"{path}/N{lattice.N}_gap")
joblib.dump(hal_val, f"{path}/N{lattice.N}_hal_val")
joblib.dump(alph_val, f"{path}/N{lattice.N}_alph_val")
# print(gap)
# print(hal_val)
# print(alph_val)