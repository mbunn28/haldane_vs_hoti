#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib

lattice = ti.Lattice(
PBC_i = True, 
PBC_j = False,
Corners = True,
a = 1,
b = 0.3,
t = 1,
l = 0.05,
M=0,
N=10)

# folder = "data/11052020/PBC/t1.9_a1.9_N12"

# lattice.energies = joblib.load( f"{folder}_energies")
# lattice.waves = joblib.load(f"{folder}_waves")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

[_,_,p,q] = lattice.make_names()

print(f"a = {p}, l = {q}")
print(f"PBC_i = {lattice.PBC_i}")
print(f"PBC_j = {lattice.PBC_j}")
print(f"Corners = {lattice.Corners}")

lattice.single_state()
lattice.energy_plot(r=[-0.5,0.5])


# index = ti.find_mode(lattice.energies, 13)
# betweenbands = np.round(np.abs(lattice.energies),4) < np.round(np.abs(lattice.energies[index[1]]),4)
# s = np.count_nonzero(betweenbands)
# print(s)
# print(np.shape(lattice.energies))

# lattice.initialize_hamiltonian()
# E = 0.3
# lattice.energies = lattice.energies - E
# lattice.plot_mode(0, shift=E)