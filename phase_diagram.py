#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti


def main():
    lattice = ti.Lattice(
        PBC_i = False, PBC_j = False,
        cornertype = 'Five Sites',
        a = 0.1, b = 1,
        l = 1, t = 1,
        M=0,
        N=5
    )

    lattice.colourcode = True
    lattice.corner_p = 0.75
    lattice.edge_p = 0.85

    lattice.energy_spectrum(indep='t',t=50,min_val=0,max_val=1)
    # lattice.single_state()

    return

if __name__ == "__main__":
    main()