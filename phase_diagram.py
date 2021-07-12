#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti


def main():
    lattice = ti.Lattice(
        PBC_i = False, PBC_j = False,
        cornertype = 'Hemaxer',
        a = 1, b = 0.1,
        l = 0, t = 1,
        M=0,
        N=10
    )

    lattice.colourcode = True
    lattice.corner_p = 0.3
    lattice.edge_p = 0.55    
    
    lattice.energy_spectrum(indep='l',t=500,min_val=0,max_val=0.4,yrange=[-2,2])
    # lattice.single_state()

    return

if __name__ == "__main__":
    main()