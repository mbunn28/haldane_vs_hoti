#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

lattice = ti.Lattice(
PBC_i = False,
PBC_j = False,
Corners = False,
alpha = 0.1,
hal = 1,
M=0,
N=5)

lattice.large_alpha = False
lattice.large_hal = False

#lattice.phase_diagram(s=50,t=50)
lattice.single_state()
# lattice.energy_spectrum(indep='Lambda',set_val=0)
#lattice.plot_groundstate()
#lattice.energy_spectrum_full(indep = "Lambda", set_val = 0, t = 1000)
