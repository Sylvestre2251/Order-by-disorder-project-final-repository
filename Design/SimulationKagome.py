



import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from numpy import linalg as LA
#from numpy.random import normal
import scipy
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os
import plotly.graph_objects as go








class Simulation(object):
    #This is the class used to do the simulation
    
    def __init__(self,Nx,Ny, J,directory,seed_control=False,seed=1500,preload=False,oldconfig=0):
        #we create a kagome lattice from parameter a and theta with Nx*Ny*3 spins
        # for this we define a few global variables 
        self.a,self.theta,self.Nx,self.Ny,self.J=1,2*np.pi/3,Nx,Ny,J,
        self.N=Nx*Ny*3

        # a_1 and a_2 are the generating vectors for the kagome lattice
        a_1=self.a*np.array([1,0])
        a_2=self.a*np.array([-1*np.cos(self.theta),np.sin(self.theta)])
        
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

        #seed control for when we want to test with the same starting configuration
        if seed_control==True:
            np.random.seed(seed) 

        # then we create a random spin configuration  self.config as a 3d vector on each grid point
        self.config=2*np.random.rand(dim[0],dim[1],dim[2],3)-1*np.ones((dim[0],dim[1],dim[2],3))

        # we normalize to get unit vectors at each grid points
        Norm=np.repeat(LA.norm(self.config,axis=3)[:, :,:, np.newaxis], 3, axis=3)
        self.config/=Norm
        
        # if we want to start from a previous simulation config we can pass it in the parameters
        if preload==True:
            self.config=oldconfig

        # we want to save some value while the simulation runs for post processing
        # especially The energy , the acceptation rate, the temperature and heat capacity
        self.Energy=[self.total_energy()]
        self.acceptation_rates=[]
        self.Temperatures=[]
        self.Capa=[]
        self.Q2=[]
        self.T2=[]


        #we create a directory to dump simulations results
        cwd = Path.cwd()
        parent = cwd.parent
        #current_dir = os.getcwd()
        
        relative_path = 'Results/'+str(directory)

        full_path = os.path.join(parent, relative_path)
        
        if not os.path.exists(full_path):
            os.mkdir(full_path)
        self.full_path=full_path
        

    def get_mean_field(self):
        #This part of the code adds the spins of the neighbors at lattice point
        #this creates a vector which is very usefull get energy contribution 
        # and do overelaxation fast

        # we create the (self.Nx,self.Ny, 3,3) tensor

        self.mean_field=np.zeros(self.config.shape)

        # we add the fields from the two other points on a triangle
        self.mean_field[:,:,0,:]=self.config[:,:,1,:]+self.config[:,:,2,:]
        self.mean_field[:,:,1,:]=self.config[:,:,2,:]+self.config[:,:,0,:]
        self.mean_field[:,:,2,:]=self.config[:,:,0,:]+self.config[:,:,1,:]


        #self.mean_field[:,:,0,:]+= np.roll(self.config[:,:,1,:],(0,-1),axis=(0,1)).copy()+np.roll(self.config[:,:,2,:],(-1,0),axis=(0,1)).copy()
        #self.mean_field[:,:,1,:]+=np.roll(self.config[:,:,2,:],(-1,1),axis=(0,1)).copy()+np.roll(self.config[:,:,0,:],(0,1),axis=(0,1)).copy()
        #self.mean_field[:,:,2,:]+=np.roll(self.config[:,:,0,:],(1,0),axis=(0,1)).copy()+np.roll(self.config[:,:,1,:],(1,-1),axis=(0,1)).copy()

        #then we add the spins of two neighboring points
        self.mean_field[:,:,0,:]+= np.roll(self.config[:,:,1,:],(0,1),axis=(0,1)).copy()+np.roll(self.config[:,:,2,:],(1,0),axis=(0,1)).copy()
        self.mean_field[:,:,1,:]+=np.roll(self.config[:,:,2,:],(1,-1),axis=(0,1)).copy()+np.roll(self.config[:,:,0,:],(0,-1),axis=(0,1)).copy()
        self.mean_field[:,:,2,:]+=np.roll(self.config[:,:,0,:],(-1,0),axis=(0,1)).copy()+np.roll(self.config[:,:,1,:],(-1,1),axis=(0,1)).copy()
        return self.mean_field
  


    def total_energy(self):
        # the total energy is just a dot product of the vector at each lattice site with 
        #the sum of neighbor's spins of course there is an overcounting factor 
        #that will be taken care of in post processing
        M3=self.get_mean_field()
        M4=self.config
        #the np einsum allows us to deal with the 4 dimensional array efficiently
        E=self.J*np.sum(np.einsum('ijkl,ijkl->ijk', M3, M4))

        return E





    def delta_energy(self):
        # this function is used to compute the energy difference for each 
        # lattice point this is very usefull to do the MC step 
        # at the same time on the whole lattice
        M1=self.get_mean_field()

        #we use the difference between the new and old spins
        M2=self.flipped_spin-self.config
        
        #the energy change at each lattice site is calculated with a dot product
        C=np.einsum('ijkl,ijkl->ijk', M1, M2)


        #we do not forget the J factor
        Delta_E=(self.J)*C
    
        return Delta_E


    def overrelaxation_2(self,dose=0.01):
        #In this part we implement Nf iterrations of overrelaxation
      


        # we select the number Nf of sites that will be rotated
        Nf=int(round(dose*self.N))
        pick = np.zeros(self.N)
        pick[:Nf-1]=1
        np.random.shuffle(pick)
        pick=np.reshape(pick,(self.Nx,self.Ny,3))
        newpick=np.repeat(pick[:, :,:, np.newaxis], 3, axis=3)
        
        newpick=newpick.astype(bool)
        #print(self.config)
        #print(self.total_energy())
        B=self.get_mean_field()
        Norm=np.repeat(LA.norm(B,axis=3)[:, :,:, np.newaxis], 3, axis=3)
        B/=Norm
        

        dotproduct=np.einsum('ijkl,ijkl->ijk', B, self.config)

        #print(dotproduct.max(),dotproduct.min())
        #dotproduct=np.sqrt(np.abs(dotproduct)) * np.sign(dotproduct)
        newdotproduct=np.repeat(dotproduct[:, :,:, np.newaxis], 3, axis=3)
        
        normal=self.config-newdotproduct*B
        #print((np.einsum('ijkl,ijkl->ijk', normal, B)).max(),(np.einsum('ijkl,ijkl->ijk', normal, B)).min(),(np.einsum('ijkl,ijkl->ijk', normal, self.config)).max(),(np.einsum('ijkl,ijkl->ijk', normal, self.config)).min())

        #newarr=(2*newdotproduct)*B-self.config
        newarr=self.config-2*normal
        #newarr=self.config-2*(self.config-newdotproduct*B)
        #print((np.einsum('ijkl,ijkl->ijk',self.config-normal, B)).max(),(np.einsum('ijkl,ijkl->ijk',self.config-normal, B)).min())
        #print(np.einsum('ijkl,ijkl->ijk',newarr-self.config, B).max(),np.einsum('ijkl,ijkl->ijk',newarr-self.config, B).min())

        #self.config=newarr
        self.config=np.where(newpick,newarr,self.config )


    def overrelaxation(self,dose=0.1):
        '''In this part we implement Nf iterrations of overrelaxation'''
      

        Nf=int(round(dose*self.N))

        pick=np.concatenate((np.ones(Nf),np.zeros(self.N-Nf)))
        np.random.shuffle(pick)
        pick=np.reshape(pick,(self.Nx,self.Ny,3))
        pick=pick.astype(bool)
        newpick=np.repeat(pick[:, :,:, np.newaxis], 3, axis=3)    


        B=self.get_mean_field()

        angle=np.pi
        Rot=np.zeros((self.Nx,self.Ny,3,4))
        Rot[:,:,:,0]=np.cos(angle/2)
        Rot[:,:,:,1:]=np.sin(angle/2)*B
        Norm=np.repeat(LA.norm(Rot,axis=3)[:, :,:, np.newaxis], 4, axis=3)
        Rot/=Norm
        Rotad=np.reshape(Rot,(self.N,4),order='C').copy()
        r = R.from_quat(Rotad,scalar_first=True)
        newarr=r.apply(  np.reshape(self.config,(self.N,3),order='C').copy())



        #M01=np.einsum('ijkl,ijkl->ijk', B,np.reshape(newarr,(self.Nx,self.Ny,3,3),order='C'))
        #M02=np.einsum('ijkl,ijkl->ijk', B, self.config)
        #print(np.einsum('ijkl,ijkl->ijk', B, normal).max())
        #print((M01-M02).max(),(M01-M02).min(),np.sum(np.abs(M01-M02)))

        self.config=np.where(newpick,np.reshape(newarr,(self.Nx,self.Ny,3,3),order='C') ,self.config )





    def Monte_Carlo(self,Nf,T,measure_capa=False,overrelaxation=False,overrelaxation_dose=0.1):
        #'''In this part we implement the Nf monte carlo steps
        #here with the metropolis move'''

        Ener=[]
        sisj1=[]
        sisj2=[]
        sisj3=[]
        beta=1/T

        for comp in range(Nf):

            

                    
            n = np.random.normal(0,np.sqrt(T),(self.Nx,self.Ny,3,3))
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


            if  measure_capa==True:
                Ener.append(self.total_energy())
                #print(self.total_energy())

                self.acceptation_rates.append(flag/self.N)
                #print(flag/self.N,T)

                if np.abs(Nf-comp)<=1000:
                    sisj1.append(np.sum(np.einsum('abcl,defl->abcdef',self.config,self.config)))
                    sisj2.append(np.sum(np.power(np.einsum('abcl,defl->abcdef',self.config,self.config),2)))
                    sisj3.append(np.sum(np.power(np.einsum('abcl,defl->abcdef',self.config,self.config),3)))
                #QQ.append((1/(self.N**2))*sisj2-1/3)
                #TT.append((1/(self.N**2))*(sisj3-(3/5)*sisj1))
                
                
        if measure_capa==True:
            Capacite=np.var(Ener)/(T**2)
            self.Capa.append(Capacite)
            self.Energy.append(Ener[len(Ener)-1])
            self.Temperatures.append(T)
            self.Q2.append((1/(self.N**2))*np.mean(sisj2)-1/3)
            self.T2.append((1/(self.N**2))*(np.mean(sisj3)-(3/5)*np.mean(sisj1)))

    def order_param(self):

        sisj1=np.sum(np.einsum('abcl,defl->abcdef',self.config,self.config))
        sisj2=np.sum(np.power(np.einsum('abcl,defl->abcdef',self.config,self.config),2))
        sisj3=np.sum(np.power(np.einsum('abcl,defl->abcdef',self.config,self.config),3))

        Q2=(1/(self.N**2))*sisj2-1/3
        T2=(1/(self.N**2))*(sisj3-(3/5)*sisj1)
        return Q2,T2





    def verify_norm(self):
        """ we have sometimes a problem with the  spin vectors not 
        being normalised after too many rotations due to errors so we normalise them after each 
        flip and we verify their maximum norms"""
        print('maximumspinsnorm=',LA.norm(self.config,axis=3).max())
        print('maximumrotatedvectornorm=',LA.norm(self.flipped_spin,axis=3).max())

    def saveconfig(self,name,T):
        # code for saving the spin configuration with parameters
        mydict={
        "Temperature": T,
        "Nx": self.Nx,
        "Ny": self.Ny,
        "J": self.J,
        "N":self.N,
        "lattice":np.array(self.lattice),
        "configuration":np.array(self.config),
        "Energy":np.array(self.Energy),
        "acceptation_rate":np.array(self.acceptation_rates)  ,
        "Capacité":np.array(self.Capa)  ,
        'Temperatures':self.Temperatures,
        "Q2":self.Q2  ,
        'T2':self.T2,
        }
        #np.save(title+'infos',[T,self.Nx,self.Ny,J,self.N])
        title= os.path.join(self.full_path, name)
        np.save(title, mydict)


    
    def display_config(self):
        """Interactive 3D visualization using Plotly"""
        position = np.reshape(self.lattice, (3 * self.Nx * self.Ny, 2), order='C')
        arrows = np.reshape(self.config, (3 * self.Nx * self.Ny, 3), order='C')

        x, y, z = position[:, 0], position[:, 1], np.zeros(3 * self.Nx * self.Ny)
        Sx, Sy, Sz = arrows[:, 0], arrows[:, 1], arrows[:, 2]

        X_lines, Y_lines, Z_lines = [], [], []
        for xi, yi, zi, Sxi, Syi, Szi in zip(x, y, z, Sx, Sy, Sz):
            X_lines += [xi, xi + Sxi, None]
            Y_lines += [yi, yi + Syi, None]
            Z_lines += [zi, zi + Szi, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=X_lines, y=Y_lines, z=Z_lines,
                                   mode="lines", line=dict(width=3, color="blue"), name="Spins"))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                   mode="markers", marker=dict(size=4, color="red"), name="Lattice points"))

        fig.update_layout(
            title="Kagome Lattice Spin Configuration",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='data'),
            width=1200, height=800
        )
        fig.show(renderer="browser")

    def spin_realign(self):



        position=np.reshape(self.lattice,(3*self.Nx*self.Ny,2),order='C')
        arrows=np.reshape(self.config,(3*self.Nx*self.Ny,3),order='C')

        Avg=[]
        for i in range(self.Nx):
            for j in range(self.Ny):
                n=np.cross(self.config[i,j,0,:],self.config[i,j,1,:])
                n2=np.cross(self.config[i,j,2,:],n)
                Avg.append(n)

        Avgn=np.mean(Avg,axis=0)
        #Avgn=np.mean(arrows,axis=0)


        def f(q,Avg=Avgn):

            q/=np.linalg.norm(q)
            r=R.from_quat(q)
            newarrows=np.sum(np.abs(r.apply(arrows)[:,2]))
            #newavg = r.apply(Avg)
            return newarrows

        res=scipy.optimize.minimize(f,np.array([1,1,1,1]), method='nelder-mead')
        r=R.from_quat(res.x/(np.linalg.norm(res.x)))
        arrows=r.apply(arrows,inverse=False)

        
        x, y, z = position[:, 0], position[:, 1], np.zeros(3 * self.Nx * self.Ny)
        Sx, Sy, Sz = arrows[:, 0], arrows[:, 1], arrows[:, 2]

        X_lines, Y_lines, Z_lines = [], [], []
        for xi, yi, zi, Sxi, Syi, Szi in zip(x, y, z, Sx, Sy, Sz):
            X_lines += [xi, xi + Sxi, None]
            Y_lines += [yi, yi + Syi, None]
            Z_lines += [zi, zi + Szi, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=X_lines, y=Y_lines, z=Z_lines,
                                   mode="lines", line=dict(width=3, color="blue"), name="Spins"))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                   mode="markers", marker=dict(size=4, color="red"), name="Lattice points"))

        fig.update_layout(
            title="Kagome Lattice Spin Configuration",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='data'),
            width=1200, height=800
        )
        fig.show(renderer="browser")

