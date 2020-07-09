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
alpha = 0,
hal = 0.1,
M=0,
N=20)

lattice.large_alpha = True
lattice.large_hal = False

# lattice.energy_spectrum('Lambda',set_val=0,t=100,max_val=1)
indep ='Lambda'
set_val=0
t=100
max_val=1

[newpath, name] = lattice.make_names("Energy vs Lambda", output="output")
q=set_val
large = "large"

vals = joblib.load( f"{newpath}/M{lattice.M}/{indep}{q}{large}_N{lattice.N}_xvals")
uniques = joblib.load(f"{newpath}/M{lattice.M}/{indep}{q}{large}_N{lattice.N}_evals")

fig = plt.figure()
for m in range(0,uniques.shape[0]):
    plt.plot(vals, uniques[m,:], lattice.colour(), alpha=0.7, linewidth=0.1)

plt.xlabel(indep)
plt.ylabel("E/t0")
plt.axvline(linewidth=0.5, color='k', x=1/(2*np.sqrt(3)))
plt.axvline(linewidth=0.5, color='k', x=1/np.sqrt(3))

plt.title(f"{name}, M = {lattice.M}")    
fig.savefig(f"{newpath}/M{lattice.M}/{indep}{q}{large}_N{lattice.N}.pdf")
plt.close(fig)
