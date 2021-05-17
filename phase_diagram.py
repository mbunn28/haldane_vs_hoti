#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti


def main():
    lattice = ti.Lattice(
        PBC_i = False, PBC_j = False,
        cornertype = 'Hexamer',
        a = 1, b = 0.25,
        l = 0, t = 1,
        M=0,
        N=17
    )

    lattice.colourcode = True
    lattice.corner_p = 0.6
    lattice.edge_p = 0.6

    lattice.energy_spectrum(indep='l',t=250,max_val=0.6)

    return

if __name__ == "__main__":
    main()