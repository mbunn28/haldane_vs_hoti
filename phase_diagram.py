#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np

def main():
    lattice = ti.Lattice(
        PBC_i = True, PBC_j = True,
        cornertype = 'Hexamer',
        a = 1, b = 0.2,
        l = 0.3, t = 1,
        M=0,
        N= 10
    )

    lattice.initialize_hamiltonian()
    lattice.eigensystem()
    V_0 = 20
    y_lim = np.amax(lattice.energies)+0.3

    imp_loc = [np.array([[3,3,1]])]#,np.array([[3,3,1,4,4,4]]),np.array([[3,3,1,3,3,2]]),np.array([[3,3,1,4,4,3]]),np.array([[3,3,1,3,3,3]])]
    bond_typ = [np.array(['Site'])]#,np.array(['Bond']),np.array(['Bond']),np.array(['Bond']),np.array(['Bond'])]
    keys = ['site']#, "dimer","hexamer","dimer_sec","hexamer_sec"]
    for i in range(5):
        lattice.impurity_spectrum(
            t=1000,
            min_val = np.array([-V_0]),max_val=np.array([V_0]),
            impurity_loc = imp_loc[i],
            imp_type = bond_typ[i],
            keyword = keys[i],
            yrange=[-y_lim,y_lim]
        )

    # imp_loc = [np.array([[3,3,1,3,3,2],[3,3,2,3,3,3],[3,3,3,3,3,4],[3,3,4,3,3,5],[3,3,5,3,3,0],[3,3,0,3,3,1]])]
    # #                np.array([[3,3,1,3,3,3],[3,3,3,3,3,5],[3,3,5,3,3,1]]),
    # #                np.array([[3,3,1,3,3,2],[3,3,2,3,3,3],[3,3,3,3,3,4],[3,3,4,3,3,5],[3,3,5,3,3,0],[3,3,0,3,3,1],[3,3,1,3,3,3],[3,3,3,3,3,5],[3,3,5,3,3,1],[3,3,2,3,3,4],[3,3,4,3,3,0],[3,3,0,3,3,2]])]
    # bond_typ = [np.array(['Bond','Bond','Bond','Bond','Bond','Bond'])]#,np.array(['Bond','Bond','Bond']),np.array(['Bond','Bond','Bond','Bond','Bond','Bond','Bond','Bond','Bond'])]
    # keys = ["full_hexamer_fullrange"]#,"triangle","hexandtri"]
    # for i in range(len(keys)):
    #     lattice.impurity_spectrum(
    #         t=1000,
    #         min_val = -V_0*np.ones(len(bond_typ[i])),max_val=V_0*np.ones(len(bond_typ[i])),
    #         impurity_loc = imp_loc[i],
    #         imp_type = bond_typ[i],
    #         keyword = keys[i],
    #         yrange=[-y_lim,y_lim]
    #     )
    # lattice.impurity_spectrum(
    #     t=200,
    #     min_val = np.array([-V_0]),max_val=np.array([V_0]),
    #     impurity_loc = np.array([[3,3,1,3,3,3]]),
    #     imp_type = np.array(['Bond']),
    #     keyword = 'hexamer_sec_bigrange',
    #     yrange=[-y_lim,y_lim]
    # )


    # for i in range(len(keys)):
    #     lattice.output = f"impurities/{keys[i]}"
    #     lattice.impurity_loc = imp_loc[i] #np.array([[1,1,0]])
    #     lattice.impurity_type = bond_typ[i]
    #     lattice.impurity = 10*np.ones(len(bond_typ[i]))
    #     lattice.single_state(y=1)
    #     a = lattice.find_energysize()
    #     indicies = [0,1,2,98,99,100,101,198,199,200,201,298,299,300,301]
    #     for i in range(len(indicies)):
    #         lattice.plot_estate(indicies[i])
    #     lattice.colourcode = False
    #     lattice.corner_p = 0.3
    #     lattice.edge_p = 0.6
    #     lattice.single_state()

    return

if __name__ == "__main__":
    main()