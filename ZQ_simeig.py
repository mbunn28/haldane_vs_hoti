#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg
import joblib
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import scipy.linalg
import numpy.random
import numpy.ma
from tqdm.auto import trange
from scipy.signal import argrelextrema
import zq_lib

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble= r'\usepackage{amsfonts}')

path_zq = "output/zq/simeig"
if not os.path.exists(path_zq):
            os.makedirs(path_zq)


location = np.array([2,2], dtype=int)
N = 5

M = int(3*(N**2))
lattice1 = zq_lib.zq_lattice(
    a = 0.4,
    b = 1,
    l = 0.5,
    t = 1,
    the = zq_lib.curve(1/2, zq='z6'),
    loc = location,
    zq = 'z6',
    N = N
)
lattice1.twist_hamiltonian()
Z6 = np.roll(np.eye(6),1,axis=0)
U_Z6 = np.kron(np.eye(N*N),Z6)
print(U_Z6[:6,:6])
print(np.count_nonzero(np.einsum('ij,jk->ik',U_Z6,lattice1.h) - np.einsum('ij,jk->ik',lattice1.h,U_Z6)))

