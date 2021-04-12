import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from math import sqrt
from math import ceil
from math import floor
import os
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy import sparse
from scipy.sparse.linalg import eigsh
from numpy import random
import joblib
import scipy.linalg
from matplotlib import rc
from tqdm.auto import trange

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

def format_func(value, tick_number):
    if value <= 1:
        return f'{np.round(value,3)}'
    else:
        v = np.round(2 - value, 3)
        part1 = r'$\frac{1}{'
        part2 = r'}$'
        return fr'{part1}{v}{part2}'

def layout(mode):
    if len(mode)<=3:
        columns = len(mode)
        rows = 1
    elif 3 < len(mode) <= 6:
        columns = int(ceil(len(mode)/2))
        rows = 2
    else:
        columns = 3
        rows = 2
    return rows, columns

#find the number modes, up until a certain set value
def no_states(energies, nostates):
    abs = np.unique(np.round(np.abs(energies),4))
    if len(abs)<25:
        return len(abs)
    else:
        return nostates

def find_mode(eigenvalues, m): # finds the index mode which energy is closest to E
    abs = np.unique(np.round(np.abs(eigenvalues),4))
    absenergies = np.round(np.abs(eigenvalues),4)
    index = [i for i, e in enumerate(absenergies) if e == abs[m]]
    return(index)

def find_modeofenergy(eigenvalues, E=float): # finds the index mode which energy is closest to E
    abs = np.unique(np.round(np.abs(eigenvalues-E),4))
    absenergies = np.round(np.abs(eigenvalues-E),4)
    index = [i for i, e in enumerate(absenergies) if e == abs[0]]
    return(index)

def pos(i,j,s): # Gives the x,y coordinates of site i,j,s
    A = np.array([sqrt(3)/2 , -1/2])
    B = np.array([-sqrt(3)/2, -1/2])
    C = np.array([0,1])
    cell =  (i*C + j*A)*3
    if s==0: return(cell+C)
    if s==1: return(cell+A+C)
    if s==2: return(cell+A)
    if s==3: return(cell+A+B)
    if s==4: return(cell+B)
    if s==5: return(cell+B+C)

class Lattice:
    def __init__(self, PBC_i=False, PBC_j=False, Corners=False, a = 1, b = 1, l = 0, t = 1, M=0, N=10):
        self.PBC_i = PBC_i
        self.PBC_j = PBC_j
        self.Corners = Corners
        self.N = N
        self.b = b
        self.t = t
        self.l = l
        self.a = a
        self.M = M
        self.h = None
        self.energies = None
        self.energies_low = None
        self.waves = None
        self.periodic_hamiltonian = False
        self.fivesites = False
        self.foursites = False
        self.threesites = False

    def lat(self,i,j,s): return(6*self.N*i+6*j+s)

    def initialize_hamiltonian(self):
        vv = 1e7
        h = np.zeros((6*(self.N)**2,6*(self.N)**2), dtype = complex)

        for i in range(self.N):
            for j in range(self.N):

                h[self.lat(i,j,0), self.lat(i,j,1)] = -self.b*self.t
                h[self.lat(i,j,0), self.lat(i,j,5)] = -self.b*self.t
                h[self.lat(i,j,0), self.lat((i+1)%self.N,j,3)] = -self.a*self.t

                h[self.lat(i,j,2), self.lat(i,j,1)] = -self.b*self.t
                h[self.lat(i,j,2), self.lat(i,j,3)] = -self.b*self.t
                h[self.lat(i,j,2), self.lat(i,(j+1)%self.N,5)] = -self.a*self.t

                h[self.lat(i,j,4), self.lat(i,j,3)] = -self.b*self.t
                h[self.lat(i,j,4), self.lat(i,j,5)] = -self.b*self.t
                h[self.lat(i,j,4), self.lat((i-1)%self.N,(j-1)%self.N,1)] = -self.a*self.t

                h[self.lat(i,j,0), self.lat(i,j,4)] = -1j*self.b*self.l
                h[self.lat(i,j,1), self.lat(i,j,5)] = -1j*self.b*self.l
                h[self.lat(i,j,2), self.lat(i,j,0)] = -1j*self.b*self.l
                h[self.lat(i,j,3), self.lat(i,j,1)] = -1j*self.b*self.l
                h[self.lat(i,j,4), self.lat(i,j,2)] = -1j*self.b*self.l
                h[self.lat(i,j,5), self.lat(i,j,3)] = -1j*self.b*self.l

                if self.N !=1:
                    h[self.lat(i,j,0), self.lat((i+1)%self.N,j,4)] = -1j*self.a*self.l
                    h[self.lat(i,j,0), self.lat((i+1)%self.N,(j+1)%self.N,4)] = -1j*self.a*self.l

                    h[self.lat(i,j,1), self.lat((i+1)%self.N,(j+1)%self.N,5)] = -1j*self.a*self.l
                    h[self.lat(i,j,1), self.lat(i,(j+1)%self.N,5)] = -1j*self.a*self.l

                    h[self.lat(i,j,2), self.lat(i,(j+1)%self.N,0)] = -1j*self.a*self.l
                    h[self.lat(i,j,2), self.lat((i-1)%self.N,j,0)] = -1j*self.a*self.l

                    h[self.lat(i,j,3), self.lat((i-1)%self.N,j,1)] = -1j*self.a*self.l
                    h[self.lat(i,j,3), self.lat((i-1)%self.N,(j-1)%self.N,1)] = -1j*self.a*self.l

                    h[self.lat(i,j,4), self.lat((i-1)%self.N,(j-1)%self.N,2)] = -1j*self.a*self.l
                    h[self.lat(i,j,4), self.lat(i,(j-1)%self.N,2)] = -1j*self.a*self.l

                    h[self.lat(i,j,5), self.lat(i,(j-1)%self.N,3)] = -1j*self.a*self.l
                    h[self.lat(i,j,5), self.lat((i+1)%self.N,j,3)] = -1j*self.a*self.l

                for s in [0,2,4]:
                    h[self.lat(i,j,s), self.lat(i,j,s)] = +self.M/2
                for s in [1,3,5]:
                    h[self.lat(i,j,s), self.lat(i,j,s)] = -self.M/2

        if self.PBC_i == False:
            for j in range(self.N):
               h[self.lat(self.N-1,j,0), self.lat(0,j,3)] = 0
               h[self.lat(0,j,4), self.lat(self.N-1,(j-1)%self.N,1)] = 0

               h[self.lat(0,(j+1)%self.N,3), self.lat(self.N-1,j,1)] = 0
               h[self.lat(self.N-1,j,0), self.lat(0,j,4)] = 0
               h[self.lat(0,(j+1)%self.N,4), self.lat(self.N-1,j,2)] = 0
               h[self.lat(self.N-1,j,5), self.lat(0,j,3)] = 0

               h[self.lat(self.N-1,j,1), self.lat(0,(j+1)%self.N,5)] = 0
               h[self.lat(self.N-1,j,0), self.lat(0,(j+1)%self.N,4)] = 0
               h[self.lat(0,j,3), self.lat(self.N-1,j,1)] = 0
               h[self.lat(0,j,2), self.lat(self.N-1,j,0)] = 0

        if self.PBC_j == False:
            for i in range(self.N):
                h[self.lat(i,self.N-1,2), self.lat(i,0,5)] = 0
                h[self.lat(i,0,4), self.lat((i-1)%self.N,self.N-1,1)] = 0

                h[self.lat((i+1)%self.N,0,3), self.lat(i,self.N-1,1)] = 0
                h[self.lat(i,self.N-1,1), self.lat(i,0,5)] = 0
                h[self.lat((i+1)%self.N,0,4), self.lat(i,self.N-1,2)] = 0
                h[self.lat(i,self.N-1,2), self.lat(i,0,0)] = 0

                h[self.lat(i,self.N-1,1), self.lat((i+1)%self.N,0,5)] = 0
                h[self.lat(i,0,5), self.lat(i,self.N-1,3)] = 0
                h[self.lat(i,self.N-1,0), self.lat((i+1)%self.N,0,4)] = 0
                h[self.lat(i,0,4), self.lat(i,self.N-1,2)] = 0

        #dimer geometry
        if self.PBC_j == False and self.Corners == True:
            for i in range(0,self.N):
                for s in [0,3,4,5]:
                    h[self.lat(i,0,s),:] = 0
                    h[:,self.lat(i,0,s)] = 0
                    h[self.lat(i,0,s),self.lat(i,0,s)] = vv

                for s in [0,1,2,3]:
                    h[self.lat(i,self.N-1,s),:] = 0
                    h[:, self.lat(i,self.N-1,s)] = 0
                    h[self.lat(i,self.N-1,s),self.lat(i,self.N-1,s)] = vv


        if self.PBC_i==False and self.Corners == True:
            for j in range(0,self.N):
                for s in [2,3,4,5]:
                    h[self.lat(0,j,s),:]=0
                    h[:,self.lat(0,j,s)]=0
                    h[self.lat(0,j,s),self.lat(0,j,s)] = vv

                for s in [0,1,2,5]:
                    h[self.lat(self.N-1,j,s),:]=0
                    h[:,self.lat(self.N-1,j,s)]=0
                    h[self.lat(self.N-1,j,s),self.lat(self.N-1,j,s)] = vv

        #5 site corners
        if self.fivesites == True:
            h[self.lat(0,0,4),:]=0
            h[:,self.lat(0,0,4)]=0
            h[self.lat(0,0,4),self.lat(0,0,4)]=vv

            h[self.lat(self.N-1,self.N-1,1),:]=0
            h[:,self.lat(self.N-1,self.N-1,1)]=0
            h[self.lat(self.N-1,self.N-1,1),self.lat(self.N-1,self.N-1,1)]=vv

        # #4 site corners
        if self.foursites == True:
            h[self.lat(0,self.N-1,2),:]=0
            h[:,self.lat(0,self.N-1,2)]=0
            h[self.lat(0,self.N-1,2),self.lat(0,self.N-1,2)]=vv

            h[self.lat(0,self.N-1,3),:]=0
            h[:,self.lat(0,self.N-1,3)]=0
            h[self.lat(0,self.N-1,3),self.lat(0,self.N-1,3)]=vv

            h[self.lat(self.N-1,0,5),:]=0
            h[:,self.lat(self.N-1,0,5)]=0
            h[self.lat(self.N-1,0,5),self.lat(self.N-1,0,5)]=vv

            h[self.lat(self.N-1,0,0),:]=0
            h[:,self.lat(self.N-1,0,0)]=0
            h[self.lat(self.N-1,0,0),self.lat(self.N-1,0,0)]=vv

        # #3 site corners
        if self.threesites == True:
            h[self.lat(0,0,4),:]=0
            h[:,self.lat(0,0,4)]=0
            h[self.lat(0,0,4),self.lat(0,0,4)]=vv

            h[self.lat(self.N-1,self.N-1,1),:]=0
            h[:,self.lat(self.N-1,self.N-1,1)]=0
            h[self.lat(self.N-1,self.N-1,1),self.lat(self.N-1,self.N-1,1)]=vv
            
            h[self.lat(0,0,5),:]=0
            h[:,self.lat(0,0,5)]=0
            h[self.lat(0,0,5),self.lat(0,0,5)]=vv

            h[self.lat(self.N-1,self.N-1,2),:]=0
            h[:,self.lat(self.N-1,self.N-1,2)]=0
            h[self.lat(self.N-1,self.N-1,2),self.lat(self.N-1,self.N-1,2)]=vv

            h[self.lat(0,0,3),:]=0
            h[:,self.lat(0,0,3)]=0
            h[self.lat(0,0,3),self.lat(0,0,3)]=vv

            h[self.lat(self.N-1,self.N-1,0),:]=0
            h[:,self.lat(self.N-1,self.N-1,0)]=0
            h[self.lat(self.N-1,self.N-1,0),self.lat(self.N-1,self.N-1,0)]=vv

        h = np.conjugate(h.transpose()) + h
        self.h = h
        return

    def initialize_periodic_hamiltonian(self, k):
        phi = np.pi/2
        kx = k[0]
        ky = k[1]

        self.periodic_hamiltonian = np.zeros((6,6), dtype=complex)
        for m in range(0,6):
            self.periodic_hamiltonian[m,(m+1)%6] = self.t*self.b
        
        self.periodic_hamiltonian[2,0] = self.l*np.exp(-1j*phi)*(self.b+self.a*(np.exp(3*1j*kx)+np.exp(1.5*1j*(kx+ky*np.sqrt(3)))))
        self.periodic_hamiltonian[3,0] = self.t*self.a*np.exp(1.5*1j*(kx+ky*np.sqrt(3)))
        self.periodic_hamiltonian[4,0] = self.l*np.exp(1j*phi)*(self.b+self.a*(np.exp(1.5*1j*(kx+ky*np.sqrt(3)))+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
        
        self.periodic_hamiltonian[3,1] = self.l*np.exp(-1j*phi)*(self.b+self.a*(np.exp(1.5*1j*(kx+ky*np.sqrt(3)))+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
        self.periodic_hamiltonian[4,1] = self.t*self.a*np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))
        self.periodic_hamiltonian[5,1] = self.l*np.exp(1j*phi)*(self.b+self.a*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
        
        self.periodic_hamiltonian[4,2] = self.l*np.exp(-1j*phi)*(self.b+self.a*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
        self.periodic_hamiltonian[5,2] = self.t*self.a*np.exp(-3*1j*kx)

        self.periodic_hamiltonian[5,3] = self.l*np.exp(-1j*phi)*(self.b+self.a*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx+ky*np.sqrt(3)))))

        self.periodic_hamiltonian = self.periodic_hamiltonian + np.conjugate(self.periodic_hamiltonian.transpose())

        return


    def make_names(self, name="", output="output"):
        if self.a < 1:
            alph = np.round(self.a,3)
            p = alph
        elif self.a == 1 and self.b == 1:
            alph = 1
            p = alph
        else:
            v= np.round(self.b,3)
            part1 = r'$\frac{1}{'
            part2 = r'}$'
            alph = fr'{part1}{v}{part2}'
            p = 2 - v

        if self.l < 1:
            lamb = np.round(self.l,3)
            q = lamb
        elif self.l == 1 and self.t == 1:
            lamb = 1
            q = 1
        else:
            v= np.round(self.t,3)
            part1 = r'$\frac{1}{'
            part2 = r'}$'
            lamb = fr'{part1}{v}{part2}'
            q = 2 - v

        if self.PBC_i == False and self.PBC_j == False:
            condition = "OBC"
        elif (self.PBC_i == False and self.PBC_j == True) or (self.PBC_i == True and self.PBC_j == False):
            condition = "Ribbon"
        elif self.PBC_i == True and self.PBC_j == True:
            condition = "PBC"
        else:
            condition = ""

        if self.Corners == True and condition != "PBC":
            corners = "with Corners"
            corn = "_corners"
        else:
            corners = ""
            corn = ""

        if name == "Energy Eigenstates" or name == "Density of States" or name == "Energy Eigenvalues of the Hamiltonian":
            title = rf"{condition} {corners} {name}: $\alpha =$ {alph}, $\lambda =$ {lamb}"
        elif name == "Energy vs Alpha":
            title = rf"{condition} {corners} $E$ vs. $\alpha$: $\lambda =$ {lamb}"
            p = q
        elif name == "Energy vs Lambda":
            title = rf"{condition} {corners} $E$ vs. $\lambda$: $\alpha =$ {alph}"
        else:
            title = ""

        newpath = f'{output}/{condition}{corn}'

        if not os.path.exists(newpath):
            os.makedirs(newpath)

        return [newpath, title, p, q]

    def eigensystem(self):
        energies, waves = scipy.linalg.eigh(self.h)
        for i in range(0, len(energies)):
            if energies[i]>1000:
                energies[i] = np.nan
        self.waves = waves[:,~np.isnan(energies)]
        energies = energies[~np.isnan(energies)]
        self.energies = energies
        return

    def eigenvalues(self):
        self.energies = np.linalg.eigvalsh(self.h)
        return

    def densityofstates(self, r=None):
        fig = plt.figure()
        if r != None:
            plt.hist(self.energies,200, range=r)
        else:
            plt.hist(self.energies,200)
        [newpath, name, p, q] = self.make_names("Density of States")
        fig.suptitle(name)
        plt.xlabel("E")
        plt.ylabel("No. of states")

        if r == None:
            zoom = ""
        else:
            zoom = "_zoom"
        fig.savefig(f"{newpath}/dos_l{q}_a{p}_N{self.N}{zoom}.png",dpi=500)
        plt.close(fig)
        return
    
    def energy_plot(self, r=None):
        fig = plt.figure()
        a = len(self.energies)
        edge_energies = np.zeros(a,dtype=bool)
        prob = np.multiply(np.conjugate(self.waves),self.waves)
        #5 site corners
        if self.fivesites == True:
            for i in range(a):
                p = prob[self.lat(0,0,0),i]+prob[self.lat(0,0,1),i]+prob[self.lat(0,0,2),i]+prob[self.lat(0,0,3),i]+prob[self.lat(0,0,5),i]
                p = p+prob[self.lat(self.N-1,self.N-1,0),i]+prob[self.lat(self.N-1,self.N-1,2),i]+prob[self.lat(self.N-1,self.N-1,3),i]+prob[self.lat(self.N-1,self.N-1,4),i]+prob[self.lat(self.N-1,self.N-1,5),i]
                if p > 0.3:
                    edge_energies[i] = True
                else:
                    edge_energies[i] = False

        #4 site corners
        if self.foursites == True:
            for i in range(a):
                p = prob[self.lat(0,self.N-1,0),i]+prob[self.lat(0,self.N-1,1),i]+prob[self.lat(0,self.N-1,4),i]+prob[self.lat(0,self.N-1,5),i]
                p = p+prob[self.lat(self.N-1,0,1),i]+prob[self.lat(self.N-1,0,2),i]+prob[self.lat(self.N-1,0,3),i]+prob[self.lat(self.N-1,0,4),i]
                if p > 0.3:
                    edge_energies[i] = True
                else:
                    edge_energies[i] = False

        #3 site corners
        if self.threesites == True:
            for i in range(a):
                p = prob[self.lat(0,0,0),i]+prob[self.lat(0,0,1),i]+prob[self.lat(0,0,2),i]
                p = p+prob[self.lat(self.N-1,self.N-1,3),i]+prob[self.lat(self.N-1,self.N-1,4),i]+prob[self.lat(self.N-1,self.N-1,5),i]
                if p > 0.3:
                    edge_energies[i] = True
                else:
                    edge_energies[i] = False

        if r != None:
            min_en = int(min(range(len(self.energies)), key=lambda i: abs(self.energies[i]+r))+1)
            max_en = int((6*(self.N**2)-min_en))
            plt.plot(self.energies[min_en:max_en],'ko',markersize=0.5)
        else:
            x = np.arange(a)
            plt.plot(x[~edge_energies],self.energies[~edge_energies],'ko',markersize=0.5)
            plt.plot(x[edge_energies],self.energies[edge_energies],'ro',markersize=0.5)
        [newpath, name, p ,q] = self.make_names("Energy Eigenvalues of the Hamiltonian")
        # fig.suptitle(name)
        plt.xlabel(r"$n$")
        plt.ylabel(r"$E$")

        if r == None:
            zoom = ""
        else:
            zoom = "_zoom"
        file_name = f"{newpath}/energyplot_l{q}_a{p}_N{self.N}{zoom}"
        file_name = file_name.replace('.','')
        fig.savefig(f'{file_name}.png',dpi=500,bbox_inches='tight')
        plt.close(fig)
        return

    def plot_eigenstates(self, max_states):
        if len(np.unique(np.round(np.abs(self.energies),4))) < max_states:
            max_states = len(np.unique(np.round(np.abs(self.energies),4)))
        for m in range(0,no_states(self.energies, max_states)):
            self.plot_mode(m)
        return

    def plot_mode(self, m, shift=0):
        mode = find_mode(self.energies,m)
        rows, columns = layout(mode)
        fig, ax_array = plt.subplots(rows, columns, squeeze=False)
        count = 0

        for l, ax_row in enumerate(ax_array):
            for k,axes in enumerate(ax_row):
                psi = np.transpose(self.waves)[mode[count]] #wavefunction
                proba = (np.abs(psi))**2
                proba = proba/np.max(proba)

                axes.set_title(rf"$E$: {np.round(self.energies[mode[count]],4)+shift}", fontsize=10)
                count +=1

                cmap = matplotlib.cm.get_cmap('inferno_r')
                normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
                colors = [cmap(normalize(value)) for value in proba]

                #plot the probability distribution:
                x  = np.zeros(6*(self.N)**2)
                y = np.zeros(6*(self.N)**2)
                for i in range(self.N):
                    for j in range(self.N):
                        for l in range(6):
                            if self.h[self.lat(i,j,l),self.lat(i,j,l)] < 99:
                                x[self.lat(i,j,l)] = pos(i,j,l)[0]
                                y[self.lat(i,j,l)] = pos(i,j,l)[1]
                                circle = Circle(pos(i,j,l),0.5,color=colors[self.lat(i,j,l)],alpha=1,ec=None,zorder=1)
                                axes.add_artist(circle)

                axes.set_ylim(pos(0,self.N-1,3)[-1]-4,pos(self.N-1,0,0)[-1]+4)
                axes.set_xlim(pos(0,0,5)[0]-4,pos(self.N-1,self.N-1,1)[0]+4)
                axes.set_yticks([])
                axes.set_xticks([])
                axes.set_aspect('equal')

                plt.scatter(x,y,s=0, c=proba, cmap= 'inferno_r',vmin=min(proba), vmax=max(proba), facecolors='none')
                cb = plt.colorbar(ax=axes,fraction=0.1, shrink=0.74)
                cb.set_ticks([])
                cb.ax.set_ylabel(r'$|\psi_i|^2$',rotation=0,labelpad=12)

            [newpath, name, p, q] = self.make_names("Energy Eigenstates")
            # if len(mode)>6:
            #     plt.suptitle(f"First 6 {name}, Total States: {len(mode)}", fontsize = 12)
            # else:
            #     plt.suptitle(name)

            file = f"{newpath}/mode{m}_l{q}_a{p}_N{self.N}.png"
            fig.tight_layout()
            fig.savefig(file,dpi=500)
            plt.close(fig)
        return
    
    def plot_estate(self, m):
        en = np.arange(len(self.energies))
        mode = [i for i, e in enumerate(en) if e == m]
        rows, columns = layout(mode)
        fig, ax_array = plt.subplots(rows, columns, squeeze=False)
        count = 0

        for l, ax_row in enumerate(ax_array):
            for k,axes in enumerate(ax_row):
                psi = np.transpose(self.waves)[mode[count]] #wavefunction
                proba = (np.abs(psi))**2
                proba = proba/np.max(proba)

                axes.set_title(rf"$E$: {np.round(self.energies[mode[count]],4)}", fontsize=10)
                count +=1

                cmap = matplotlib.cm.get_cmap('inferno_r')
                normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
                colors = [cmap(normalize(value)) for value in proba]

                #plot the probability distribution:
                x  = np.zeros(6*(self.N)**2)
                y = np.zeros(6*(self.N)**2)
                for i in range(self.N):
                    for j in range(self.N):
                        for l in range(6):
                            if self.h[self.lat(i,j,l),self.lat(i,j,l)] < 99:
                                x[self.lat(i,j,l)] = pos(i,j,l)[0]
                                y[self.lat(i,j,l)] = pos(i,j,l)[1]
                                circle = Circle(pos(i,j,l),0.5,color=colors[self.lat(i,j,l)],alpha=1,ec=None,zorder=1)
                                axes.add_artist(circle)

                axes.set_ylim(pos(0,self.N-1,3)[-1]-4,pos(self.N-1,0,0)[-1]+4)
                axes.set_xlim(pos(0,0,5)[0]-4,pos(self.N-1,self.N-1,1)[0]+4)
                axes.set_yticks([])
                axes.set_xticks([])
                axes.set_aspect('equal',adjustable='box')

                plt.scatter(x,y,s=0, c=proba, cmap= 'inferno_r',vmin=min(proba), vmax=max(proba), facecolors='none')
                cb = plt.colorbar(ax=axes)#,fraction=0.1, shrink=0.74)
                cb.set_ticks([])
                cb.ax.set_ylabel(r'$|\psi_i|^2$',rotation=0,labelpad=12)

            [newpath, name, p, q] = self.make_names("Energy Eigenstates")
            # if len(mode)>6:
            #     plt.suptitle(f"First 6 {name}, Total States: {len(mode)}", fontsize = 12)
            # else:
            #     plt.suptitle(name)

            file = f"{newpath}/estate{m}_l{q}_a{p}_N{self.N}"
            file = file.replace('.','')
            fig.tight_layout()
            fig.savefig(file,dpi=500,bbox_inches='tight')
            plt.close(fig)
        return

    def single_state(self):
        self.initialize_hamiltonian()
        self.eigensystem()
        self.energy_plot()
        self.plot_cornerstates()
        return

    def plot_cornerstates(self):
        a = len(self.energies)
        edge_energies = np.zeros(a,dtype=bool)
        prob = np.multiply(np.conjugate(self.waves),self.waves)
        #5 site corners
        if self.fivesites == True:
            for i in range(a):
                p = prob[self.lat(0,0,0),i]+prob[self.lat(0,0,1),i]+prob[self.lat(0,0,2),i]+prob[self.lat(0,0,3),i]+prob[self.lat(0,0,5),i]
                p = p+prob[self.lat(self.N-1,self.N-1,0),i]+prob[self.lat(self.N-1,self.N-1,2),i]+prob[self.lat(self.N-1,self.N-1,3),i]+prob[self.lat(self.N-1,self.N-1,4),i]+prob[self.lat(self.N-1,self.N-1,5),i]
                if p > 0.3:
                    edge_energies[i] = True
                else:
                    edge_energies[i] = False

        #4 site corners
        if self.foursites == True:
            for i in range(a):
                p = prob[self.lat(0,self.N-1,0),i]+prob[self.lat(0,self.N-1,1),i]+prob[self.lat(0,self.N-1,4),i]+prob[self.lat(0,self.N-1,5),i]
                p = p+prob[self.lat(self.N-1,0,1),i]+prob[self.lat(self.N-1,0,2),i]+prob[self.lat(self.N-1,0,3),i]+prob[self.lat(self.N-1,0,4),i]
                if p > 0.3:
                    edge_energies[i] = True
                else:
                    edge_energies[i] = False

        #3 site corners
        if self.threesites == True:
            for i in range(a):
                p = prob[self.lat(0,0,0),i]+prob[self.lat(0,0,1),i]+prob[self.lat(0,0,2),i]
                p = p+prob[self.lat(self.N-1,self.N-1,3),i]+prob[self.lat(self.N-1,self.N-1,4),i]+prob[self.lat(self.N-1,self.N-1,5),i]
                if p > 0.3:
                    edge_energies[i] = True
                else:
                    edge_energies[i] = False
        states = np.argwhere(edge_energies)
        # eners = self.energies[states].transpose()
        # same_ener = np.round(np.diff(eners,prepend=0),2)==0
        # same_ener = same_ener.transpose()
        # states = states[~same_ener]
        for i in range(len(states)):
            self.plot_estate(states[i])
        return

    def find_energysize(self):
        self.initialize_hamiltonian()
        self.eigenvalues()
        return len(self.energies)

    def colour(self):
        if self.PBC_i == True and self.PBC_j == True:
            colour = 'r'
        else:
            colour = 'b'
        return colour

    # def phase_diagram(self, s=40, t=40):

    #     large_alpha = self.large_alpha
    #     large_hal = self.large_hal

    #     minval = np.zeros((2*s-1,2*t-1))
    #     vals = np.zeros((2*s-1, 2*t-1, 2))

    #     for k in range(0,t):
    #         hal = round(k/t, 3)

    #         for n in range(0,s):

    #             print(f"{s*k + n}/{t*s}", end='\r')

    #             alpha = round(n**2/s**2,3)

    #             self.alpha = alpha
    #             self.hal = hal

    #             self.large_hal = False
    #             self.large_alpha = False

    #             self.initialize_hamiltonian()
    #             self.sparse_eigenvalues()
    #             minval[n,k] = np.amin(np.abs(self.energies_low))
    #             vals[n,k,:] = [alpha, hal]

    #             self.large_alpha = True

    #             self.initialize_hamiltonian()
    #             self.sparse_eigenvalues()
    #             minval[-n,k] = np.amin(np.abs(self.energies_low))
    #             vals[-n,k,:] = [2-alpha, hal]

    #             self.large_hal = True

    #             self.initialize_hamiltonian()
    #             self.sparse_eigenvalues()
    #             minval[-n,-k] = np.amin(np.abs(self.energies_low))
    #             vals[-n,-k,:] = [2-alpha, 2-hal]

    #             self.large_alpha = False

    #             self.initialize_hamiltonian()
    #             self.sparse_eigenvalues()
    #             minval[n,-k] = np.amin(np.abs(self.energies_low))
    #             vals[n,-k,:] = [alpha, 2-hal]

    #     fig = plt.figure()
    #     plt.pcolormesh(vals[:,:,1], vals[:,:,0], minval, cmap = "inferno_r")

    #     plt.xlabel("Lambda")
    #     plt.ylabel("Alpha")
    #     plt.colorbar()

    #     [newpath, name] = self.make_names("Energy vs Lambda")

    #     plt.title(f"Phase Diagram, M = {self.M}")
    #     if self.M < 0:
    #         fig.savefig(f"{newpath}/phasediagram_negM{-self.M}.pdf")
    #     else:
    #         fig.savefig(f"{newpath}/phasediagram_M{self.M}.pdf")
    #     plt.close(fig)

    #     mask = minval < 0.01
    #     fig = plt.figure()
    #     plt.pcolormesh(vals[:,:,1], vals[:,:,0], mask, cmap = "inferno_r")

    #     plt.xlabel("Lambda")
    #     plt.ylabel("Alpha")
    #     plt.colorbar()

    #     [newpath, name] = self.make_names("Energy vs Lambda")

    #     plt.title("Phase Diagram 2")
    #     fig.savefig(f"{newpath}/phasediagram2.pdf")
    #     plt.close(fig)

    #     self.large_hal = large_hal
    #     self.large_alpha = large_alpha
    #     return

    def energy_spectrum(self, indep, t=100, min_val=0, max_val=1):
        #to use this function: create a lattice w all other param values
        #... and feed this fn the min and max vals you want to plot over
        a = self.find_energysize()
        bigenergies = np.zeros((a, t))
        edgeenergies = np.zeros((a, t))
        edge_states = np.zeros((a, t),dtype=bool)
        vals = np.round(np.linspace(min_val,max_val,num=t),3)
        a0 = self.a
        b0 = self.b
        l0 = self.l
        t0 = self.t
        
        for k in trange(0,t):
            if indep == 'l':
                self.l = vals[k]
            elif indep == 'a':
                self.a = vals[k]
            elif indep == 'b':
                self.b = vals[k]
            elif indep == 't':
                self.t = vals[k]
            else:
                print("That's not a parameter!")
                return

            self.initialize_hamiltonian()
            self.eigensystem()
            bigenergies[:,k] = self.energies
            prob = np.multiply(np.conjugate(self.waves),self.waves)
            
            #5 site corners
            if self.fivesites == True:
                for i in range(a):
                    p = prob[self.lat(0,0,0),i]+prob[self.lat(0,0,1),i]+prob[self.lat(0,0,2),i]+prob[self.lat(0,0,3),i]+prob[self.lat(0,0,5),i]
                    p = p+prob[self.lat(self.N-1,self.N-1,0),i]+prob[self.lat(self.N-1,self.N-1,2),i]+prob[self.lat(self.N-1,self.N-1,3),i]+prob[self.lat(self.N-1,self.N-1,4),i]+prob[self.lat(self.N-1,self.N-1,5),i]
                    if p > 0.3:
                        edgeenergies[i,k] = bigenergies[i,k]
                        bigenergies[i,k] = np.NaN
                    else:
                        edgeenergies[i,k] = np.NaN

            #4 site corners
            if self.foursites == True:
                for i in range(a):
                    p = prob[self.lat(0,self.N-1,0),i]+prob[self.lat(0,self.N-1,1),i]+prob[self.lat(0,self.N-1,4),i]+prob[self.lat(0,self.N-1,5),i]
                    p = p+prob[self.lat(self.N-1,0,1),i]+prob[self.lat(self.N-1,0,2),i]+prob[self.lat(self.N-1,0,3),i]+prob[self.lat(self.N-1,0,4),i]
                    if p > 0.3:
                        edgeenergies[i,k] = bigenergies[i,k]
                        bigenergies[i,k] = np.NaN
                    else:
                        edgeenergies[i,k] = np.NaN

            #3 site corners
            if self.threesites == True:
                for i in range(a):
                    p = prob[self.lat(0,0,0),i]+prob[self.lat(0,0,1),i]+prob[self.lat(0,0,2),i]
                    p = p+prob[self.lat(self.N-1,self.N-1,3),i]+prob[self.lat(self.N-1,self.N-1,4),i]+prob[self.lat(self.N-1,self.N-1,5),i]
                    if p > 0.3:
                        edgeenergies[i,k] = bigenergies[i,k]
                        bigenergies[i,k] = np.NaN
                    else:
                        edgeenergies[i,k] = np.NaN

            for i in range(0, len(bigenergies[:,k])):
                if bigenergies[i,k]>1000:
                    bigenergies[i,k] = np.nan
                if edgeenergies[i,k]>1000:
                    edgeenergies[i,k] = np.nan

        bigenergies = np.round(bigenergies, 4)
        other_energies = bigenergies
        # other_energies[edge_states == 1] = np.NaN
        new_array = [tuple(row) for row in bigenergies]
        uniques = np.unique(new_array, axis=0)

        if indep == 't' or indep == 'b':
            vals = 2 - vals

        if indep == 'l' or indep == 't':
            var = r'$\lambda$'
            thing = "Energy vs Lambda"
            name_var = 'a'
        elif indep == 'a' or indep == 'b':
            var = r'$\alpha$'
            thing = "Energy vs Alpha"
            name_var = 'l'
        fig, ax = plt.subplots()
        for m in range(a):
            ax.plot(vals, bigenergies[m,:], color='k', alpha=0.7, linewidth=0.2)
            ax.plot(vals, edgeenergies[m,:], color='r', alpha=0.7, linewidth=0.2)

        ax.set_xlabel(var)
        ax.set_ylabel(r"$E$")
        # ax.set_xlim(min_val,max_val)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

        [newpath, name, p, _] = self.make_names(thing)
        
        ax.set_title(name)
        file_path = f"{newpath}/M{self.M}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = f"{file_path}/{indep}_{name_var}{p}_N{self.N}.png"
        fig.savefig(file_name, dpi=500)
        plt.close(fig)

        # joblib.dump(vals, f"{newpath}/M{self.M}/{indep}_{name_var}{p}_N{self.N}_xvals")
        # joblib.dump(uniques, f"{newpath}/M{self.M}/{indep}_{name_var}{p}_N{self.N}_evals")

        self.a = a0
        self.b = b0
        self.l = l0
        self.t = t0
        return

    # def plot_groundstate(self):
    #     mode = find_mode(self.energies,0)
    #     count = 0
    #     for f in range(0,ceil(len(mode)/6)):
    #         if count > floor(len(mode)/6):
    #             if (len(mode)-count)<=3:
    #                 columns = len(mode)
    #                 rows = 1
    #             elif 3 < (len(mode)-count) <= 6:
    #                 columns = int(ceil(len(mode)/2))
    #                 rows = 2
    #             else:
    #                 rows = 2
    #                 columns = 3
    #                 fig, ax_array = plt.subplots(rows, columns, squeeze=False)
    #         else:
    #             rows = 2
    #             columns = 3
    #             fig, ax_array = plt.subplots(rows, columns, squeeze=False)

    #         for l, ax_row in enumerate(ax_array):
    #             for k,axes in enumerate(ax_row):
    #                 psi = np.transpose(self.waves)[mode[count]] #wavefunction
    #                 proba = (np.abs(psi))**2
    #                 proba = proba/np.max(proba)


    #                 axes.set_title(f"E: {np.round(self.energies[mode[count]],4)}, Mode:0.{count}", fontsize=10)
    #                 count +=1

    #                 cmap = matplotlib.cm.get_cmap('inferno_r')
    #                 normalize = matplotlib.colors.Normalize(vmin=min(proba), vmax=max(proba))
    #                 colors = [cmap(normalize(value)) for value in proba]

    #                 #plot the probability distribution:
    #                 x  = np.zeros(6*(self.N)**2)
    #                 y = np.zeros(6*(self.N)**2)
    #                 for i in range(self.N):
    #                     for j in range(self.N):
    #                         for l in range(6):
    #                             if self.h[self.lat(i,j,l),self.lat(i,j,l)] < 99:
    #                                 x[self.lat(i,j,l)] = pos(i,j,l)[0]
    #                                 y[self.lat(i,j,l)] = pos(i,j,l)[1]
    #                                 circle = Circle(pos(i,j,l),0.5,color=colors[self.lat(i,j,l)],alpha=1,ec=None,zorder=1)
    #                                 axes.add_artist(circle)
    #                 axes.set_ylim(pos(0,self.N-1,3)[-1]-4,pos(self.N-1,0,0)[-1]+4)
    #                 axes.set_xlim(pos(0,0,5)[0]-4,pos(self.N-1,self.N-1,1)[0]+4)
    #                 axes.set_yticklabels([])
    #                 axes.set_xticklabels([])
    #                 axes.set_aspect('equal')

    #                 plt.scatter(x,y,s=0, c=proba, cmap= 'inferno_r',vmin=min(proba), vmax=max(proba), facecolors='none')
    #                 plt.colorbar(ax=axes, use_gridspec=True)

    #             [newpath, name]= self.make_names("Energy Eigenstates")
    #             if not os.path.exists(f"{newpath}/groundstate"):
    #                 os.makedirs(f"{newpath}/groundstate")
    #             if len(mode)>6:
    #                 plt.suptitle(f"Ground State {f}/{ceil(len(mode)/6)}: {name}, Total States: {len(mode)}", fontsize = 10)
    #             else:
    #                 plt.suptitle(name)

    #             if self.large_hal == True:
    #                 p = np.round(2-self.hal,4)
    #             else:
    #                 p = np.round(self.hal,4)
    #             if self.large_alpha == True:
    #                 q = np.round(2-self.alpha,4)
    #             else:
    #                 q = np.round(self.alpha,4)

    #             file = f"{newpath}/groundstate/estate_hal{p}_alpha{q}_gs{f}.pdf"
    #             fig.savefig(file)
    #             plt.close(fig)
    #     return

    # def energy_spectrum_full(self, indep, set_val, t=100):
    #     large_alpha = self.large_alpha
    #     large_hal = self.large_hal
    #     a = self.find_energysize()
    #     bigenergies = np.zeros((a, 2*t-2))
    #     vals = np.zeros(2*t-2)

    #     for k in range(0,t):
    #         value = round(k/t, 3)
    #         vals[k] = value

    #         if indep == 'Lambda':
    #             self.large_hal = False
    #             self.alpha = set_val
    #             self.hal = value
    #         elif indep == 'Alpha':
    #             self.large_alpha = False
    #             self.alpha = value
    #             self.hal = set_val
    #         else:
    #             self.alpha = 0
    #             self.hal = 0

    #         print(f"{k}/{t}", end='\r')

    #         self.initialize_hamiltonian()
    #         self.eigenvalues()
    #         bigenergies[:,k] = self.energies

    #         # for i in range(0, len(bigenergies[:,k])):
    #         #     if bigenergies[i,k]>1000:
    #         #         bigenergies[i,k] = np.nan

    #         vals[1-k] = 2-value
    #         if indep == 'Lambda':
    #             self.large_hal = True
    #             self.alpha = set_val
    #             self.hal = value
    #         elif indep == 'Alpha':
    #             self.large_alpha = False
    #             self.alpha = value
    #             self.hal = set_val
    #         else:
    #             self.alpha = 0
    #             self.hal = 0

    #         self.initialize_hamiltonian()
    #         self.eigenvalues()
    #         bigenergies[:,-k] = self.energies

    #         # for i in range(0, len(bigenergies[:,1-k])):
    #         #     if bigenergies[i,1-k]>1000:
    #         #         bigenergies[i,1-k] = np.nan

    #     bigenergiesmask = bigenergies > 10
    #     bigenergies[bigenergiesmask] = np.nan
    #     bigenergies = np.round(bigenergies, 4)
    #     new_array = [tuple(row) for row in bigenergies]
    #     uniques = np.unique(new_array, axis=0)
    #     # uniques = uniques[~np.all(uniques == 0, axis=0)]

    #     fig = plt.figure()
    #     for m in range(0,uniques.shape[0]):
    #         plt.plot(vals, uniques[m,:], self.colour(), alpha=0.7, linewidth=0.1)

    #     plt.xlabel(indep)
    #     plt.ylabel("E/self.b*self.t")


    #     if indep == "Lambda":
    #         [newpath, name] = self.make_names("Energy vs Lambda")
    #     else:
    #         [newpath, name] = self.make_names("Energy vs Alpha")

    #     if not os.path.exists(f"{newpath}/M={self.M}"):
    #         os.makedirs(f"{newpath}/M={self.M}")

    #     plt.title(f"{name}, M = {self.M}")
    #     fig.savefig(f"{newpath}/M={self.M}/{name}.pdf")
    #     plt.close(fig)

    #     self.large_hal = large_hal
    #     self.large_alpha = large_alpha
    #     return

    def min_energy(self):
        self.initialize_hamiltonian()
        return np.amin(np.abs(self.energies),axis=None)

    def min_periodic_energy(self, kpoint):
        self.initialize_periodic_hamiltonian(kpoint)
        return np.amin(np.abs(np.linalg.eigvalsh(self.periodic_hamiltonian)),axis=None)