#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib

lattice = ti.Lattice(
PBC_i = True, 
PBC_j = True,
Corners = False,
alpha = 0.6,
hal = 0.4,
M=0,
N=32)

lattice.large_alpha = False
lattice.large_hal = True

# folder = "data/11052020/PBC/t1.9_a1.9_N12"

# lattice.energies = joblib.load( f"{folder}_energies")
# lattice.waves = joblib.load(f"{folder}_waves")

lattice.single_state()
# lattice.densityofstates(r=[-0.1,0.1])

# index = ti.find_mode(lattice.energies, 13)
# betweenbands = np.round(np.abs(lattice.energies),4) < np.round(np.abs(lattice.energies[index[1]]),4)
# s = np.count_nonzero(betweenbands)
# print(s)
# print(np.shape(lattice.energies))

# lattice.initialize_hamiltonian()
# E = 0.3
# lattice.energies = lattice.energies - E
# lattice.plot_mode(0, shift=E)