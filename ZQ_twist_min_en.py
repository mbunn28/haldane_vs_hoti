#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg
import joblib
import os
import scipy.linalg
import numpy.random
from tqdm import tqdm
import zq_lib

def rule(y):
    a = np.zeros(len(y))
    b = np.zeros(len(y))
    for i in range(len(y)):
        if 0 <= y[i] <= 1:
            a[i] = y[i]
            b[i] = 1
        if y[i] > 1:
            a[i] = 1
            b[i] = 2 - y[i]
    return a, b

def main():
    
    # TODO implement argparse
    iterations = 40
    location = np.array([2,2], dtype=int)
    N = 18
    a = 1
    b = 0.5
    t = 1
    l = 0.2
    filling = 'half'

    zq = ['z2']
    if filling == 'half':
        M = int(3*(N**2))
    elif filling == 'third':
        M = int(2*(N**2))
    elif filling == 'sixth':
        M = int(N**2)

    def make_filenames():
        path_zq = f"output/zq/min_energy/{filling}"
        if not os.path.exists(path_zq):
            os.makedirs(path_zq)
        small_energy_path = f'{path_zq}/small_energy_N{N}_it{iterations}'
        return [path_zq,small_energy_path]

    _,small_energy_path = make_filenames()

    if (os.path.exists(small_energy_path)):
        small_energy = joblib.load(small_energy_path)
        return

    small_energy = np.zeros((iterations+1,len(zq)))

    phi = np.random.rand(6*(N**2),M)
    phi = scipy.linalg.orth(phi)

    def compute(j):
        def create_lattice(theta):
            lattice = zq_lib.zq_lattice(
                    a = a,
                    b = b,
                    l = l,
                    t = t,
                    the = zq_lib.curve(theta, zq=zq[j]),
                    loc = location,
                    zq = zq[j],
                    N = N  
            )
            return lattice

        def get_gap(lattice, M):
            gap = lattice.energies[M]-lattice.energies[M-1]
            return gap

        def get_states(lattice, M):
            evecs = lattice.waves
            singlestates = evecs[:, :M]
            return singlestates

        def do_integral(lattice1, lattice2):
            Di = np.einsum('ij,jk',np.conjugate(lattice1.proj.transpose()),lattice2.proj,optimize='greedy')
            det_Di = numpy.linalg.slogdet(Di)
            if det_Di == 0:
                raise ValueError('error! det zero!')
            D = det_Di[0]
            
            # Nphi = np.einsum('ij,jk',np.conjugate(lattice2.proj.transpose()), lattice2.proj)
            # _, det_Nphi = numpy.linalg.slogdet(Nphi)
            # if det_Nphi == -np.Inf:
                # raise ValueError('The overlap matrix det = 0!')

            return D

        Ts = np.linspace(0, 1, iterations+1)
        lattices = [create_lattice(theta) for theta in Ts]
        for lattice in lattices:
            lattice.twist_hamiltonian()
        gaps = [get_gap(lattice, M) for lattice in lattices]
        small_energy[:,j] = gaps
        return 
        
    for j in tqdm(range(len(zq))):
        compute(j)

    print(small_energy)
    joblib.dump(small_energy,small_energy_path)

    return 

if __name__ == "__main__":
    main()