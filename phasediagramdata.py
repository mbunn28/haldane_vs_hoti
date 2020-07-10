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
N=25)

t=150
gap = np.zeros((2*t+1,2*t+1))
hal_val = np.zeros(2*t+1)
alph_val = np.zeros(2*t+1)
for k in range(0,t):
    lattice.hal = k/t
    hal_val[k] = lattice.hal
    hal_val[2*t-k] = 2-lattice.hal

    for m in range(0,t):

        lattice.alpha = m/t
        alph_val[m] = lattice.alpha
        alph_val[2*t-m] = 2-lattice.alpha

        lattice.large_hal= False
        lattice.large_alpha = False

        lattice.initialize_hamiltonian()
        lattice.eigensystem()
        gap[k,m] = np.min(np.abs(lattice.energies))

        lattice.large_hal = True

        lattice.initialize_hamiltonian()
        lattice.eigensystem()
        gap[2*t-k,m] = np.min(np.abs(lattice.energies))

        lattice.large_alpha = True

        lattice.initialize_hamiltonian()
        lattice.eigensystem()
        gap[2*t-k,2*t-m] = np.min(np.abs(lattice.energies))

        lattice.large_hal = False

        lattice.initialize_hamiltonian()
        lattice.eigensystem()
        gap[k,2*t-m] = np.min(np.abs(lattice.energies))

lattice.large_hal= False
lattice.hal=1
hal_val[t]=1

for m in range(0,t):

    lattice.large_alpha = False 
    lattice.alpha = m/t

    lattice.initialize_hamiltonian()
    lattice.eigensystem()
    gap[t,m] = np.min(np.abs(lattice.energies))

    lattice.large_alpha = True

    lattice.initialize_hamiltonian()
    lattice.eigensystem()
    gap[t,2*t-m] = np.min(np.abs(lattice.energies))

lattice.hal=1
lattice.alpha=1
alph_val[t]=1
lattice.initialize_hamiltonian()
lattice.eigensystem()
gap[t,t] = np.min(np.abs(lattice.energies))

path = "output/phasediagram"
if not os.path.exists(path):
            os.makedirs(path)

joblib.dump(gap, f"{path}/N{lattice.N}_gap")
joblib.dump(hal_val, f"{path}/N{lattice.N}_hal_val")
joblib.dump(alph_val, f"{path}/N{lattice.N}_alph_val")