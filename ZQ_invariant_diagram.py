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
    points = 100
    iterations = 4
    location = np.array([2,2], dtype=int)
    N = 12
    max_x = 2
    min_x = 1
    max_y = 2
    min_y = 1

    zq = ['z6']
    M = int(3*(N**2))

    x = np.linspace(min_x, max_x, num=points)
    y = np.linspace(min_y, max_y, num=points)

    def make_filenames():
        path_zq = "output/zq/diagrams"
        if not os.path.exists(path_zq):
            os.makedirs(path_zq)
        zq_phases_path = f'{path_zq}/zq_phases_N{N}_it{iterations}_res{points}'
        small_energy_path = f'{path_zq}/small_energy_N{N}_it{iterations}_res{points}'
        small_energy_loc_path = f'{path_zq}/smallen_loc_N{N}_it{iterations}_res{points}'
        return [path_zq,zq_phases_path,small_energy_path,small_energy_loc_path]

    _,zq_phases_path,small_energy_path,small_energy_loc_path = make_filenames()

    if (os.path.exists(zq_phases_path)):
        zq_phases = joblib.load(zq_phases_path)
        small_energy = joblib.load(small_energy_path)
        small_energy_loc = joblib.load(small_energy_loc_path)
        return

    a_vals, b_vals = rule(y)
    l_vals, t_vals = rule(x)
    l, a = np.meshgrid(l_vals,a_vals)
    t, b = np.meshgrid(t_vals,b_vals)


    zq_phases = np.zeros((points,points,len(zq)))
    small_energy = np.zeros((points,points,len(zq)))
    small_energy_loc = np.zeros((points,points,len(zq)))

    phi = np.random.rand(6*(N**2),M)
    phi = scipy.linalg.orth(phi)

    def compute(m, n, j):
        def create_lattice(theta):
            lattice = zq_lib.zq_lattice(
                    a = a[n,m],
                    b = b[n,m],
                    l = l[n,m],
                    t = t[n,m],
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
        gap = min(gaps)
        small_energy[n,m,j] = gap
        small_energy_loc[n,m,j] = np.argmin(gaps)/iterations

        for lattice in lattices:
            singlestate = get_states(lattice, M)
            pa = np.einsum('ij,jk', singlestate, np.conjugate(singlestate.transpose()),optimize='greedy')
            lattice.proj = np.einsum('ij,jk', pa, phi,optimize='greedy')

        D = [do_integral(lattice1, lattice2) for lattice1, lattice2 in zip(lattices, lattices[1:])]
        D = np.prod(D)

        zq_phase = np.angle(D)
        zq_phase1 = 6*zq_phase/(2*np.pi)
        zq_phase2 = np.round(zq_phase1,2)
        if np.isclose(zq_phase1,zq_phase2) != True:
            zq_phase2 = np.NaN
        if zq_phase2 < -1e-1:
            zq_phase2 = zq_phase2 + 6
        zq_phases[n,m,j] = zq_phase2
        
    with tqdm(total=points * points * len(zq)) as pbar:
        for m, n, j in np.ndindex(points, points, len(zq)):
            compute(m, n,j)
            pbar.update(1)

    joblib.dump(zq_phases,zq_phases_path)
    joblib.dump(small_energy,small_energy_path)
    joblib.dump(small_energy_loc,small_energy_loc_path)

    return 

if __name__ == "__main__":
    main()