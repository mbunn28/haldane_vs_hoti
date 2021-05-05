#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib
from tqdm.auto import trange

lattice = ti.Lattice(
PBC_i = False, 
PBC_j = False,
Corners = False,
a = 1,
b = 0.2,
t = 1,
l = 0.05,
M=0,
N=20)

lattice.fivesites=True
# folder = "data/11052020/PBC/t1.9_a1.9_N12"

# lattice.energies = joblib.load( f"{folder}_energies")
# lattice.waves = joblib.load(f"{folder}_waves")
# t_vals = [0.1,0.2,0.3]
# for i in range(len(t_vals)):
# [_,_,p,q] = lattice.make_names()

# print(f"a = {p}, l = {q}")
# print(f"PBC_i = {lattice.PBC_i}")
# print(f"PBC_j = {lattice.PBC_j}")
# print(f"Corners = {lattice.Corners}")

lattice.single_state()
# lattice.energy_plot(r=0.5)
# lattice.a = t_vals[i]
# lattice.energy_spectrum('t',t=200,min_val=0.2,max_val=0.6)
        

# index = ti.find_mode(lattice.energies, 13)
# betweenbands = np.round(np.abs(lattice.energies),4) < np.round(np.abs(lattice.energies[index[1]]),4)
# s = np.count_nonzero(betweenbands)
# print(s)
# print(np.shape(lattice.energies))

# lattice.initialize_hamiltonian()
# E = -1.8
# lattice.energies = lattice.energies - E
# lattice.plot_mode(0, shift=E)