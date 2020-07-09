#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib

lattice = ti.Lattice(
PBC_i = False,
PBC_j = False,
Corners = False,
alpha = 0.1,
hal = 0.1,
M=0,
N=12)

folder = 'data/11052020'
lattice.large_alpha = True
lattice.large_hal = True

lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Alpha',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0, t=400)

lattice.large_alpha = False


[new_path,name] = lattice.make_names(output=folder)
[p,q]= lattice.parvals_forlabels()
lattice.energies = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
lattice.waves = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")
lattice.initialize_hamiltonian()
lattice.densityofstates()

lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Alpha',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0, t=400)
lattice.large_alpha = True

lattice.PBC_i = True
lattice.PBC_j = True 


[new_path,name] = lattice.make_names(output=folder)
[p,q]= lattice.parvals_forlabels()
lattice.energies = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
lattice.waves = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")
lattice.initialize_hamiltonian()
lattice.densityofstates()

lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Alpha',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0, t=400)

lattice.large_alpha = False



[new_path,name] = lattice.make_names(output=folder)
[p,q]= lattice.parvals_forlabels()
lattice.energies = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
lattice.waves = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")
lattice.initialize_hamiltonian()
lattice.densityofstates()

lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Alpha',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0, t=400)
lattice.large_alpha = True

lattice.PBC_j = False


[new_path,name] = lattice.make_names(output=folder)
[p,q]= lattice.parvals_forlabels()
lattice.energies = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
lattice.waves = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")
lattice.initialize_hamiltonian()
lattice.densityofstates()

lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Alpha',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0, t=400)

lattice.large_alpha = False


[new_path,name] = lattice.make_names(output=folder)
[p,q]= lattice.parvals_forlabels()
lattice.energies = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
lattice.waves = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")
lattice.initialize_hamiltonian()
lattice.densityofstates()

lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Alpha',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0, t=400)
lattice.large_alpha = True

lattice.PBC_i = False

lattice.Corners = True


[new_path,name] = lattice.make_names(output=folder)
[p,q]= lattice.parvals_forlabels()
lattice.energies = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
lattice.waves = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")
lattice.initialize_hamiltonian()
lattice.densityofstates()

lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Alpha',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0, t=400)

lattice.large_alpha = False


[new_path,name] = lattice.make_names(output=folder)
[p,q]= lattice.parvals_forlabels()
lattice.energies = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
lattice.waves = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")
lattice.initialize_hamiltonian()
lattice.densityofstates()

lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Alpha',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0, t=400)
lattice.large_alpha = True

lattice.PBC_i = True


[new_path,name] = lattice.make_names(output=folder)
[p,q]= lattice.parvals_forlabels()
lattice.energies = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
lattice.waves = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")
lattice.initialize_hamiltonian()
lattice.densityofstates()

lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Alpha',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0, t=400)

lattice.large_alpha = False


[new_path,name] = lattice.make_names(output=folder)
[p,q]= lattice.parvals_forlabels()
lattice.energies = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
lattice.waves = joblib.load( f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")
lattice.initialize_hamiltonian()
lattice.densityofstates()

lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Alpha',data = folder ,set_val=0.1, t=400)
lattice.energy_spectrumfromdata(indep='Lambda',data = folder ,set_val=0, t=400)
lattice.large_alpha = True

lattice.PBC_i = False

lattice.Corners = False
