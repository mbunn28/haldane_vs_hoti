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
alpha = 0.1,
hal = 0.1,
M=0,
N=20)

lattice.large_alpha = True
lattice.large_hal = True

#lattice.phase_diagram(s=50,t=50)
# lattice.single_state()
# [newpath, name] = lattice.make_names()

# joblib.dump(lattice.energies, f"{newpath}/t{lattice.hal}_a{lattice.alpha}_energies")
# joblib.dump(lattice.waves, f"{newpath}/t{lattice.hal}_a{lattice.alpha}_waves")
lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
#lattice.plot_groundstate()
#lattice.energy_spectrum_full(indep = "Lambda", set_val = 0, t = 1000)
# x = lattice.min_energy(0)
# print(x)