#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib

lattice = ti.Lattice(
PBC_i = False, 
PBC_j = False,
Corners = True,
alpha = 0.5575,
hal = 0.2,
M=0,
N=24)

lattice.large_alpha = True
lattice.large_hal = True

# folder = "data/11052020/PBC/t1.9_a1.9_N12"

# lattice.energies = joblib.load( f"{folder}_energies")
# lattice.waves = joblib.load(f"{folder}_waves")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

lattice.single_state()
lattice.densityofstates(r=[-0.1,0.1])

if lattice.large_hal == True:
    p = np.round(2-lattice.hal,4)
else:
    p = np.round(lattice.hal,4)
if lattice.large_alpha == True:
    q = np.round(2-lattice.alpha,4)
else:
    q = np.round(lattice.alpha,4)

print(f"a = {q}, t = {p}\n")
print(f"PBC_i = {lattice.PBC_i}\n")
print(f"PBC_j = {lattice.PBC_j}\n")
print(f"Corners = {lattice.Corners}\n")

# index = ti.find_mode(lattice.energies, 13)
# betweenbands = np.round(np.abs(lattice.energies),4) < np.round(np.abs(lattice.energies[index[1]]),4)
# s = np.count_nonzero(betweenbands)
# print(s)
# print(np.shape(lattice.energies))

# lattice.initialize_hamiltonian()
# E = 0.3
# lattice.energies = lattice.energies - E
# lattice.plot_mode(0, shift=E)