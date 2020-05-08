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
N=2)

lattice.large_alpha = True
lattice.large_hal = True

lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Alpha',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Lambda',set_val=0, t=400)

lattice.large_alpha = False

lattice.eigensystem()
[new_path,name] = lattice.make_names()
[p,q]= lattice.parvals_forlabels()
joblib.dump(lattice.energies, f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
joblib.dump(lattice.waves, f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")

lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Alpha',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Lambda',set_val=0, t=400)
lattice.large_alpha = True

lattice.PBC_i = True
lattice.PBC_j = True 

lattice.eigensystem()
[new_path,name] = lattice.make_names()
[p,q]= lattice.parvals_forlabels()
joblib.dump(lattice.energies, f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
joblib.dump(lattice.waves, f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")

lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Alpha',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Lambda',set_val=0, t=400)

lattice.large_alpha = False

lattice.eigensystem()
[new_path,name] = lattice.make_names()
[p,q]= lattice.parvals_forlabels()
joblib.dump(lattice.energies, f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
joblib.dump(lattice.waves, f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")

lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Alpha',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Lambda',set_val=0, t=400)
lattice.large_alpha = True

lattice.PBC_j = False

lattice.eigensystem()
[new_path,name] = lattice.make_names()
[p,q]= lattice.parvals_forlabels()
joblib.dump(lattice.energies, f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
joblib.dump(lattice.waves, f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")

lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Alpha',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Lambda',set_val=0, t=400)

lattice.large_alpha = False

lattice.eigensystem()
[new_path,name] = lattice.make_names()
[p,q]= lattice.parvals_forlabels()
joblib.dump(lattice.energies, f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
joblib.dump(lattice.waves, f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")

lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Alpha',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Lambda',set_val=0, t=400)
lattice.large_alpha = True

lattice.PBC_i = False

lattice.Corners = True

lattice.eigensystem()
[new_path,name] = lattice.make_names()
[p,q]= lattice.parvals_forlabels()
joblib.dump(lattice.energies, f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
joblib.dump(lattice.waves, f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")

lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Alpha',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Lambda',set_val=0, t=400)

lattice.large_alpha = False

lattice.eigensystem()
[new_path,name] = lattice.make_names()
[p,q]= lattice.parvals_forlabels()
joblib.dump(lattice.energies, f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
joblib.dump(lattice.waves, f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")

lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Alpha',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Lambda',set_val=0, t=400)
lattice.large_alpha = True

lattice.PBC_i = True

lattice.eigensystem()
[new_path,name] = lattice.make_names()
[p,q]= lattice.parvals_forlabels()
joblib.dump(lattice.energies, f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
joblib.dump(lattice.waves, f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")

lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Alpha',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Lambda',set_val=0, t=400)

lattice.large_alpha = False

lattice.eigensystem()
[new_path,name] = lattice.make_names()
[p,q]= lattice.parvals_forlabels()
joblib.dump(lattice.energies, f"{new_path}/t{p}_a{q}_N{lattice.N}_energies")
joblib.dump(lattice.waves, f"{new_path}/t{p}_a{q}_N{lattice.N}_waves")

lattice.energy_spectrum(indep='Lambda',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Alpha',set_val=0.1, t=400)
lattice.energy_spectrum(indep='Lambda',set_val=0, t=400)
lattice.large_alpha = True

lattice.PBC_i = False

lattice.Corners = False
