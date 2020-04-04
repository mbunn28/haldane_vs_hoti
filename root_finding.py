#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

N = 200
values = np.zeros(4*N)
xvals = np.zeros(4*N)
for i in range(0,N):
    lattice = ti.Lattice(
    PBC_i = True,
    PBC_j = True,
    Corners = False,
    alpha = 0.5,
    hal = 0.005*i,
    M=0,
    N=7)

    lattice.large_hal = False
    lattice.large_alpha=False
    res = opt.minimize_scalar(lattice.min_energy, bounds=(0,1), method='bounded', tol=1e-6)
    values[i] = res.x
    xvals[i] = 0.005*i
    if i < 2:
        values[2*N-i-1] = np.nan
        xvals[2*N-i-1] = 2-0.005*i
        values[i+2*N] = np.nan
        xvals[i+2*N] = 2-0.005*i
        lattice.large_alpha=True
        lattice.large_hal=True
    else:
        lattice.large_hal = True

        res = opt.minimize_scalar(lattice.min_energy, bounds=(0,1), method='bounded', tol=1e-6)
        values[2*N-i-1] = res.x
        xvals[2*N-i-1] = 2-0.005*i

        lattice.large_alpha=True

        res = opt.minimize_scalar(lattice.min_energy, bounds=(0,1), method='bounded', tol=1e-6)
        values[i+2*N] = 2-res.x
        xvals[i+2*N] = 2-0.005*i

    lattice.large_hal = False
    zero = False
    k = 0.01
    while (zero is False) and (k<=1):
        res = opt.minimize_scalar(lattice.min_energy, bounds=(1-k,1), method='bounded', tol=1e-6)
        if lattice.min_energy(res.x) < 1e-6:
            zero = True
        else:
            k = k + 0.01

    values[4*N-i-1] = 2-res.x
    xvals[4*N-i-1] = 0.005*i    

    lattice.large_hal=False

    print(f"{i}/{N}", end='\r')

joblib.dump(values, "values")
joblib.dump(xvals, "xvals")