#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib
import scipy.stats as st

lattice = ti.Lattice(
PBC_i = False,
PBC_j = False,
Corners = False,
alpha = 0.1,
hal = 0.1,
M=0,
N=20)

lattice.large_alpha = True
lattice.large_hal = True

lattice.initialize_hamiltonian()
[newpath, name] = lattice.make_names()
lattice.energies = joblib.load(f"{newpath}/t{lattice.hal}_a{lattice.alpha}_energies")
lattice.waves = joblib.load(f"{newpath}/t{lattice.hal}_a{lattice.alpha}_waves")

# plot the energy eigenstate closest to a particular energy value
E = +0.12 # desired energy value 
lattice.energies = lattice.energies - E
lattice.plot_mode(0, E)
lattice.energies = lattice.energies + E

# print(st.mode(np.round(lattice.energies,2)))