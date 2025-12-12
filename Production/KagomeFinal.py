# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 16:51:59 2025

@author: Sylvestre
"""



import numpy as np
import numpy.random as rnd
import numpy.linalg
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.random import normal
import scipy
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import plotly.graph_objects as go
import plotly.graph_objs as go





class Configuration:
    #"""generate a configuration of spins"""
    
    def __init__(self, a,theta,Nx,Ny, J,seed_control=False,seed=1500):
        #'we create a kagome lattice from parameter a and theta with Nx*Ny*3 spins , '
        self.a,self.theta,self.Nx,self.Ny,self.J=a,theta,Nx,Ny,J,
        self.N=Nx*Ny*3

        # a_1 and a_2 are the lattice translation generating vectors 
        a_1=a*np.array([1,0])
        a_2=a*np.array([-1*np.cos(theta),np.sin(theta)])
        
        # we construct the unit cell
        unit_cell=np.array([[0,0],0.5*a_2,0.5*a_1])
        
        # we construct the grid points from the translations of the lattice
        self.x_l,self.y_l=np.meshgrid(range(Nx),range(Ny))
        A_1=np.tensordot(self.y_l,[a_1,a_1,a_1],axes=0)
        A_2=np.tensordot(self.x_l,[a_2,a_2,a_2],axes=0)

        # the lattice points coordinates are created from adding a unit cell at each grid point
        self.lattice=np.tensordot(np.ones((Nx,Ny)),unit_cell,axes=0)+A_1+A_2

        # we create a dim object containing the dimensions of the lattice 
        dim=self.lattice.shape

        #seed control for when we want to test with the same configuration
        if seed_control==True:
            np.random.seed(seed) 

        # then we create a random spin configuration  self.config as a unit 3d vector on each grid point
        self.config=2*np.random.rand(dim[0],dim[1],dim[2],3)-1*np.ones((dim[0],dim[1],dim[2],3))

        # we make sure they are unit vectors
        Norm=np.repeat(LA.norm(self.config,axis=3)[:, :,:, np.newaxis], 3, axis=3)
        self.config/=Norm
        
        # we want to save some value while the simulation runs for post processing
        self.Energy=[self.total_energy()]
        self.acceptation_rates=[]

        self.Capa=[]

    def get_mean_field(self):
        """this part implements the sommation over spins nearest spins"""
        self.mean_field=np.zeros(self.config.shape)
        self.mean_field[:,:,0,:]=self.config[:,:,1,:]+self.config[:,:,2,:]
        self.mean_field[:,:,1,:]=self.config[:,:,2,:]+self.config[:,:,0,:]
        self.mean_field[:,:,2,:]=self.config[:,:,0,:]+self.config[:,:,1,:]




        self.mean_field[:,:,0,:]+= np.roll(self.config[:,:,1,:],(0,1),axis=(0,1)).copy()+np.roll(self.config[:,:,2,:],(1,0),axis=(0,1)).copy()
        self.mean_field[:,:,1,:]+=np.roll(self.config[:,:,2,:],(1,-1),axis=(0,1)).copy()+np.roll(self.config[:,:,0,:],(0,-1),axis=(0,1)).copy()
        self.mean_field[:,:,2,:]+=np.roll(self.config[:,:,0,:],(-1,0),axis=(0,1)).copy()+np.roll(self.config[:,:,1,:],(-1,1),axis=(0,1)).copy()



        return self.mean_field
  

    
    def total_energy(self):
        
        M1=self.get_mean_field()
        M2=self.config

        #np.einsum uses Einstein summation rule to calculate faster
        
        E=0.5*self.J*np.sum(np.einsum('ijkl,ijkl->ijk', M1, M2))
        
        return E




    
    def delta_energy(self):
        ''' this function is used to compute the energy difference after a change '''
        
        M1=self.get_mean_field()
        M2=self.flipped_spin-self.config
        
        C=np.einsum('ijkl,ijkl->ijk', M1, M2)

        #we compute the difference in energy contribution between the modified and unaltered spin
        Delta_E=(self.J)*C
        return Delta_E

    

    def overrelaxation_2(self,dose=0.1):
        #In this part we implement Nf iterrations of overrelaxation
      


        # we select the number Nf of sites that will be rotated
        Nf=int(round(dose*self.N))
        pick = np.zeros(self.N)
        pick[:Nf-1]=1
        np.random.shuffle(pick)
        pick=np.reshape(pick,(self.Nx,self.Ny,3))
        pick=pick.astype(bool)
        newpick=np.repeat(pick[:, :,:, np.newaxis], 3, axis=3)



        B=self.get_mean_field()
        Norm=np.repeat(LA.norm(B,axis=3)[:, :,:, np.newaxis], 3, axis=3)
        B/=Norm


        dotproduct=np.einsum('ijkl,ijkl->ijk', B, self.config)
        newdotproduct=np.repeat(dotproduct[:, :,:, np.newaxis], 3, axis=3)
        normal=self.config-newdotproduct*B



        newarr=self.config-2*normal

        self.config=np.where(newpick,newarr,self.config )
        







    def Monte_Carlo(self,Nf,T,measure_capa=False,overrelaxation=False,overrelaxation_dose=0.1):
        #'''In this part we implement the Nf monte carlo steps
        #here with the metropolis move'''

        Ener=[]
        beta=1/T

        for comp in range(Nf):

            

                    
            n = T * np.random.normal(0, 1, (self.Nx, self.Ny, 3, 3))
            self.flipped_spin=self.config+n
            Norm=np.repeat(LA.norm(self.flipped_spin,axis=3)[:, :,:, np.newaxis], 3, axis=3)
            self.flipped_spin/=Norm
            

            delta=self.delta_energy()


            proba=np.random.rand(self.Nx,self.Ny,3)
            expener=np.exp(-beta*delta)
            decision=np.where(proba< expener)


            newproba = np.repeat(proba[:, :,:, np.newaxis], 3, axis=3)
            newexpener= np.repeat(expener[:, :,:, np.newaxis], 3, axis=3)
            self.config=np.where(newproba<= newexpener,self.flipped_spin,self.config)


            flag=len(decision[0])

            if overrelaxation==True:

                self.overrelaxation_2(overrelaxation_dose)


            if comp%100==0 and measure_capa==True:
                Ener.append(self.total_energy())
                #print(self.total_energy())
                self.acceptation_rates.append(flag/self.N)
                #print(flag/self.N,T)
                
                
        if measure_capa==True:
            Capacite=np.var(Ener)/(T**2)
            self.Capa.append(Capacite)
            self.Energy=np.concatenate((self.Energy,Ener))
            self.Temperatures.append(T*np.ones(len(Ener)))
 


    

    def measure_Capa(self,Nf,T,Nx,Ny):
        beta=1/T
        Ener=[]
        for comp in range(Nf):


                    
            n = T * np.random.normal(0, 1, (self.Nx, self.Ny, 3, 3))

            self.flipped_spin=self.config+n
            Norm=np.repeat(LA.norm(self.flipped_spin,axis=3)[:, :,:, np.newaxis], 3, axis=3)
            
            self.flipped_spin/=Norm
            delta=self.delta_energy()

            proba=np.random.rand(self.Nx,self.Ny,3)
            expener=np.exp(-beta*delta)
            decision=np.where(proba<= expener)

            newproba = np.repeat(proba[:, :,:, np.newaxis], 3, axis=3)
            newexpener= np.repeat(expener[:, :,:, np.newaxis], 3, axis=3)
            #print(np.array_equal(np.where(newproba[:,:,:,0]<=newexpener[:,:,:,0]), np.where(newproba[:,:,:,1]<=newexpener[:,:,:,1])))
            self.config=np.where(newproba<= newexpener,self.flipped_spin,self.config)

            
            flag=len(decision[0])


            if comp%100==0:
                Ener.append(self.total_energy())
                #print(self.total_energy())
                self.acceptation_rates.append(flag/self.N)
                #print(flag/self.N,T)

        Capacite=np.var(Ener)/(T**2)
        self.Capa.append(Capacite/(3*Nx*Ny))
        self.Energy=np.concatenate((self.Energy,Ener))
        

        return Ener




    def verify_norm(self):
        """ we have sometimes a problem with the  spin vectors not 
        being normalised after too many rotations due to errors so we normalise them after each 
        flip and we verify their maximum norms"""
        #print('norm=',LA.norm(self.config,axis=3).max())
        #print('normf=',LA.norm(self.flipped_spin,axis=3).max())



    def display_config(self, arrow_scale=0.3):
    # positions (flatten)
        position = np.reshape(self.lattice, (3 * self.Nx * self.Ny, 2), order='C')
        arrows = np.reshape(self.config, (3 * self.Nx * self.Ny, 3), order='C')
        
        x = position[:, 0]
        y = position[:, 1]
        z = np.zeros_like(x)
        Sx, Sy, Sz = arrows[:, 0], arrows[:, 1], arrows[:, 2]
    
        # one color per sublattice
        colors = ['red', 'green', 'blue']
        spin_indices = np.tile([0, 1, 2], self.Nx * self.Ny)
    
        fig = go.Figure()
    
        # ---- Un trace par sous-lattice ----
        for s in range(3):
            xs, ys, zs = [], [], []
    
            for xi, yi, zi, Sxi, Syi, Szi, si in zip(x, y, z, Sx, Sy, Sz, spin_indices):
                if si == s:
                    xs += [xi, xi + arrow_scale * Sxi, None]
                    ys += [yi, yi + arrow_scale * Syi, None]
                    zs += [zi, zi + arrow_scale * Szi, None]
    
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(width=5, color=colors[s]),  # <-- UNE SEULE COULEUR VALIDÉE
                name=f"Sous-lattice {s}"
            ))
    
        # ---- points du réseau ----
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=4, color="black"),
            name="sites"
        ))
    
        fig.update_layout(scene=dict(aspectmode="data"))
        fig.show(renderer="browser")

    def triangle_spins_sum(self):
        """
        Calcule la somme vectorielle des trois spins de chaque triangle (cellule).
        Retourne un tableau de forme (Nx, Ny, 3).
        """
        return self.config.sum(axis=2)
    
    
    
J = 1
L = 5
Nx = L
Ny = L
theta = 2*np.pi/3

T_high = 1
T_low = 0.001
n_T = 20
Temp = np.linspace(T_high, T_low, n_T)

numberMC = 10000

# --- Affichage initial (une seule fois) ---
A = Configuration(1, theta, Nx, Ny, J)

# --- Boucle sur les températures ---
for j, T in enumerate(Temp):
    A.Monte_Carlo(numberMC, T)
    A.verify_norm()
    print(j, "done")

# --- Affichage final (une seule fois, après la dernière température) ---
A.display_config()



triangle_sums = A.triangle_spins_sum()

for i in range(A.Nx):
    for j in range(A.Ny):
        print(f"Sum of spins in the triangle ({i},{j}) :", triangle_sums[i, j])
    




n_runs = 15  # nombre de runs indépendants pour chaque T

# tableaux pour stocker les moyennes et écarts-types
E_mean_all = []
E_std_all = []
Capa_mean_all = []
Capa_std_all = []




for j, T in enumerate(Temp):
    decoherencetime = 10000
    numberMC = 10000

    Ener_runs = []
    Capa_runs = []

    for run in range(n_runs):
        A = Configuration(1, theta, Nx, Ny, J)

        A.Monte_Carlo(decoherencetime, T)
        A.verify_norm()

        Ener = A.measure_Capa(numberMC, T, Nx, Ny)
        A.verify_norm()

        Ener_runs.append(np.mean(Ener))
        Capa_runs.append(A.Capa[-1])

    # ---- ICI et UNIQUEMENT ICI (fin des runs pour cette température) ----
    E_mean_all.append(np.mean(Ener_runs))
    E_std_all.append(np.std(Ener_runs))
    
    Capa_mean = np.mean(Capa_runs)
    Capa_std = np.std(Capa_runs)
    
    # filtering because of a point of anomaly
    Capa_mean = min(Capa_mean, 2)
    Capa_std = min(Capa_std, 2 - Capa_mean)


    Capa_mean_all.append(Capa_mean)
    Capa_std_all.append(Capa_std)

    print(f"T step {j+1}/{n_T} done")
    
    
    
    
    
    
    


# Plot Heat Capacity Vs T in browser with plotly

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=Temp,
    y=Capa_mean_all,
    error_y=dict(
        type='data',
        array=Capa_std_all,
        visible=True
    ),
    mode='markers+lines',
    name="Capacité thermique"
))

fig.update_layout(
    xaxis_type="log",
    xaxis_title="Température T",
    yaxis_title="Capacité thermique C(T)",
    title="Capacité thermique interactive",
    template="plotly_white"
)

fig.show(renderer="browser")




# Plot Energy Vs T in browser with plotly


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=Temp,
    y=E_mean_all,
    error_y=dict(
        type='data',
        array=E_std_all,
        visible=True
    ),
    mode='markers+lines',
    name="Heat Capacity"
))

fig.update_layout(
    xaxis_title="Temperature T",
    yaxis_title="Energy E",
    title="Interactive Energy",
    template="plotly_white"
)

fig.show(renderer="browser")






