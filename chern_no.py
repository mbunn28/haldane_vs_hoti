#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

lattice = ti.Lattice(
PBC_i = True,
PBC_j = True,
Corners = False,
alpha = 1,
hal = 0.05,
M=0,
N=4)

lattice.large_hal = False
lattice.large_alpha = False

b1 = np.array([4*np.pi*np.sqrt(3)/9,0])
b2 = (2*np.pi/3)*np.array([1/np.sqrt(3),1])

k = np.array([4*np.pi/9,0])
lattice.initialize_periodic_hamiltonian(k)
