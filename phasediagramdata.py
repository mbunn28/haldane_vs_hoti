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
N=2)

t=4 #multiples of twenty only
delta = 1/t
alph_val = np.zeros(t+1)
lattice.large_alpha = True

for V in range(0,5):
    print(V)
    print("\n")
    Vmin = [0, 0.1, 0.55, 0.75, 0.25]
    Vmax = [0.1, 0.55, 1.0, 1.0, 0.75]
    v_vals = np.arange(Vmin[V],Vmax[V],delta)
    VLarge = [True, True, True, False, False]

    gap = np.zeros((v_vals.size,t+1))
    hal_val = np.zeros(v_vals.size)

    for k in range(0, v_vals.size):
        
        lattice.hal = v_vals[k]
        lattice.large_hal = VLarge[V]

        if lattice.large_hal == False:
            hal_val[k] = lattice.hal
        elif lattice.large_hal == True:
            hal_val[v_vals.size-k-1] = 2-lattice.hal
        else:
            print('broken')

        for m in range(0,t+1):

            print(f"{k*(t+1)+m}/{v_vals.size*(t+1)}", end='\r')

            if m == t:
                lattice.alpha = 1
                alph_val[0] = 1
            else:
                lattice.alpha = m/t
                alph_val[t-m] = 2-lattice.alpha

            if lattice.large_hal == True:
                if (lattice.hal<0.1) or ((lattice.alpha >= -(1/6)*(lattice.hal-1)+0.3) and (lattice.alpha <= -(5/9)*(lattice.hal-1)+0.5)):
                    gap[v_vals.size-k-1,t-m] = lattice.min_energy()
                else:
                    gap[v_vals.size-k-1,t-m] = np.NaN

            elif lattice.large_hal == False:
                if (lattice.hal >= 0.6) and (lattice.alpha >= 0.75*(lattice.hal-1) + 0.3) and (lattice.alpha <= 0.5*(lattice.hal-1)+0.5):
                    gap[k,t-m] = lattice.min_energy()
                elif (0.25 <= lattice.hal < 0.6) and (lattice.alpha <= (6/7)*(lattice.hal-0.25)):
                    gap[k,t-m] = lattice.min_energy()
                else:
                    gap[k,t-m] = np.NaN

    # path = "output/phasediagram"
    # if not os.path.exists(path):
    #             os.makedirs(path)

    # joblib.dump(gap, f"{path}/N{lattice.N}_gap_v{V}")
    # joblib.dump(hal_val, f"{path}/N{lattice.N}_hal_val_v{V}")
    # joblib.dump(alph_val, f"{path}/N{lattice.N}_alph_val_v{V}")
    # print(gap)
    # print(hal_val)
    # print(alph_val)