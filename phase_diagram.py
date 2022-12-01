#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
from numpy.random import randint
from numpy.random import random

def main():
    lattice = ti.Lattice(
        PBC_i = True, PBC_j = True,
        cornertype = 'Hexamer',
        a = 1, b = 1,
        l = 0.1, t = 1,
        M=0,
        N=15
    )

    # lattice.initialize_hamiltonian()
    # lattice.eigensystem()
    V_0 = 50
    # y_lim = np.amax(lattice.energies)+0.3

    N_imp = 20
    imp_loc = np.zeros((N_imp,3))
    for i in range(N_imp):
        imp_loc[i,0] = randint(lattice.N)
        imp_loc[i,1] = randint(lattice.N)
        imp_loc[i,2] = randint(6)
    # imp_strength = np.ones(N_imp)
    # file_keyword = 'const'
    imp_strength = np.empty(N_imp)
    for i in range(N_imp):
        imp_strength[i] = 0.5*random()*(2*randint(0,2)-1)
    file_keyword = 'rand'
    #imp_loc = [np.array([[3,3,1]])]#,np.array([[3,3,1,4,4,4]]),np.array([[3,3,1,3,3,2]]),np.array([[3,3,1,4,4,3]]),np.array([[3,3,1,3,3,3]])]
    bond_typ = np.array(['Site']*N_imp)#,np.array(['Bond']),np.array(['Bond']),np.array(['Bond']),np.array(['Bond'])]
    #keys = ['site']*N_imp#, "dimer","hexamer","dimer_sec","hexamer_sec"]
    [file_path, _,_,_]=lattice.make_names('Impurity Spectrum', output=f'output/impurities_{file_keyword}')
    f = open(f'{file_path}/data','w')
    f.write("Impurity Loc \tRelative Strength\n")
    for i in range(N_imp):
        lattice.impurity_spectrum(
            t=101,
            field_max = V_0,
            field_type = file_keyword,
            impurity_loc = imp_loc[:(i+1),:],
            imp_type = bond_typ[:(i+1)],
            keyword = f"{i+1}siteimp",
            file_keyword = file_keyword,
            imp_strength = imp_strength
            # yrange=[-y_lim,y_lim]
        )
        f.write(f'{imp_loc[i,:]}\t{imp_strength[i]}\n')
   
    f.close()

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