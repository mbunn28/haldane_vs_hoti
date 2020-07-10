#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

N = 14
path = "output/phasediagram"
gap = joblib.load(f"{path}/N{N}_gap")
hal_val = joblib.load(f"{path}/N{N}_hal_val")
alph_val = joblib.load(f"{path}/N{N}_alph_val")

# gap[gap>0.01]= np.NaN
fig = plt.figure()
plt.pcolormesh(hal_val, alph_val, np.transpose(gap))
fig.savefig(f"{path}/N{N}_diagram.pdf")