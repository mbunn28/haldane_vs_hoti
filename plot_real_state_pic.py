#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib import cm
from matplotlib import colors


def main():
    lattice = ti.Lattice(
        PBC_i = False, PBC_j = False,
        cornertype = 'Five Sites',
        a = 0.2, b = 1,
        l = 0.2, t = 1,
        M=0,
        N=10
    )

    lattice.colourcode = True
    lattice.corner_p = 0.1
    lattice.edge_p = 2   

    lattice.initialize_hamiltonian()
    lattice.eigensystem()

    def plot_estate(lattice, ax, m,anc, wave_max):
        en = np.arange(len(lattice.energies))
        mode = [i for i, e in enumerate(en) if e == m]
        # rows, columns = 1, 1
        # fig, ax_array = plt.subplots(1, 1, squeeze=False,figsize=(1.7,6))
        # count = 0
        lattice.energies[np.round(lattice.energies,4) == 0] = 0

        # for l, ax_row in enumerate(ax_array):
        #     for k,axes in enumerate(ax_row):
        psi = np.transpose(lattice.waves)[mode[0]] #wavefunction
        proba = (np.abs(psi))**2
        proba = proba/wave_max

        # ax.set_title(rf"$E$: {np.round(lattice.energies[mode[0]],4)}", fontsize=10)

        cmap = cm.get_cmap('inferno_r')
        normalize = colors.Normalize(vmin=0, vmax=1)
        colours = [cmap(normalize(value)) for value in proba]

        #plot the probability distribution:
        x  = np.zeros(6*(lattice.N)**2)
        y = np.zeros(6*(lattice.N)**2)
        for i in range(lattice.N):
            for j in range(lattice.N):
                for l in range(6):
                    if lattice.h[lattice.lat(i,j,l),lattice.lat(i,j,l)] < 99:
                        x[lattice.lat(i,j,l)] = ti.pos(i,j,l)[0]
                        y[lattice.lat(i,j,l)] = ti.pos(i,j,l)[1]
                        circle = Circle(ti.pos(i,j,l),0.5,color=colours[lattice.lat(i,j,l)],alpha=1,ec=None,zorder=1)
                        ax.add_artist(circle)

        ax.set_ylim(ti.pos(0,lattice.N-1,3)[1]-4,ti.pos(lattice.N-1,0,0)[1]+4)
        ax.set_xlim(ti.pos(0,0,5)[0]-4,ti.pos(lattice.N-1,lattice.N-1,1)[0]+4)
        ax.set_yticks([])
        ax.set_xticks([])
        # ax.set_aspect('equal',adjustable='box',anchor=anc)

        sc = plt.scatter(x,y,s=0, c=proba, cmap= 'inferno_r',vmin=min(proba), vmax=max(proba), facecolors='none')
        # cb = ti.colorbar(sc)
        # cb.set_ticks([])
        # cb.ax.set_title(r'$|\psi_i|^2$')#,rotation=0,labelpad=12)
        return

    def add_inset(lattice, axs, lims, m, wave_max, anc = 'NW',label=None):
        subax = axs.inset_axes(lims)
        plot_estate(lattice,subax,m, anc, wave_max)
        if label != None:
            subax.text(ti.pos(0,0,5)[0],ti.pos(0,lattice.N-1,3)[1],label)
        return

    def energy_plot(lattice, r=None):
        fig_w = 3.4
        fig_h = 4
        fig_rat = fig_h/fig_w
        fig, axs = plt.subplots(figsize=(fig_w,fig_h))
       
        lattice.find_corners()
        lattice.find_edges()
        x = np.arange(len(lattice.energies))
        axs.plot(x[~lattice.corners],lattice.energies[~lattice.corners],'ko',markersize=0.5)
        axs.plot(x[lattice.corners],lattice.energies[lattice.corners],'ro',markersize=0.5)
        axs.plot(x[lattice.edges],lattice.energies[lattice.edges],'bo',markersize=0.5)

        axs.set_xlabel(r"$n$")
        axs.set_ylabel(r"$E$")
        y_max = np.amax(lattice.energies)
        axs.set_ylim((-y_max-0.1,y_max+0.1))

        lattice.find_corners()
        lattice.find_edges()
        state_locs = np.logical_or(lattice.corners,lattice.edges)
        states = np.argwhere(state_locs)

        psi = np.transpose(lattice.waves)[states] #wavefunction
        proba = (np.abs(psi))**2
        wave_max = np.amax(proba)

        length = ti.pos(lattice.N-1,0,0)[1] - ti.pos(0,lattice.N-1,3)[1] + 8
        width = ti.pos(lattice.N-1,lattice.N-1,2)[0]- ti.pos(0,0,5)[0] + 8
        ltow = length/width
        ratio = ltow/fig_rat
        inset_0_w = 0.28
        inset_1_w= 0.43
        pad = 0.02
        add_inset(lattice,axs,(pad,1-pad-inset_0_w*ratio,inset_0_w,inset_0_w*ratio),states[0], wave_max, label='(a)')
        add_inset(lattice,axs,(pad,1-2*pad-2*inset_0_w*ratio,inset_0_w,inset_0_w*ratio),states[2],wave_max, label='(b)')
        # add_inset(lattice,axs,(2*pad+inset_0_w,1-pad-inset_0_w*ratio,inset_0_w,inset_0_w*ratio),states[2], wave_max)
        add_inset(lattice,axs,(1-pad-inset_1_w,pad,inset_1_w,inset_1_w*ratio),states[4], wave_max, label='(c)')

        cax = axs.inset_axes((2*pad+inset_0_w,1-pad-inset_0_w*ratio,0.02,inset_0_w*ratio))
        cb = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1),cmap=cm.get_cmap('inferno_r')), cax=cax)
        cb.set_ticks([])
        cax.set_ylabel(r'$|\psi_i|^2$',rotation=0)
        cax.yaxis.set_label_coords(4.5,1)
            
        # file_name = f"{newpath}/energyplot{zoom}"
        # file_name = file_name.replace('.','')
        fig.savefig(f'output/real_space_plot.png',dpi=500,bbox_inches='tight')
        return

    energy_plot(lattice)

    return

if __name__ == "__main__":
    main()