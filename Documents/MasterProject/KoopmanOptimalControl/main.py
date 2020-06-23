#%%
import importlib

import sympy as sy
import numpy as np
import plotFunctions
importlib.reload(plotFunctions)
from plotFunctions import *
import tensorflow as tf
import keras as kr

x, t = sy.symbols('x t')
#Single Well
V_s=x**2
#Asymmetric Double Well
V_2=0.25*x**4-0.5*x**2+0.25/3*x**3-0.25*x
#4 Well
V_4=2*(x**8 + 0.8*sy.exp(-80*x**2)+0.2*sy.exp(-80*(x-0.5)**2)+0.5*sy.exp(-40*(x+0.5)**2));
#Unforced Duffing
V_d=0.25 * x ** 4 - 0.5 * x ** 2
x_list_list=[np.linspace(-1,1,1000),np.linspace(-2,2,1000),np.linspace(-1,1,1000),np.linspace(-2,2,1000)]
titles=["Single Well Potential Function","Assymetric Double-Well Potential Function","Assymetric 4-well Potential Function","Unforced Duffing Oscillator"]
for i,V in enumerate([V_s,V_2,V_4,V_d]):
    fig,ax=plt.subplots()
    plot_potential(sy.lambdify(x,V),x_list_list[i],ax=ax,fig=fig,title=titles[i])
    plt.savefig("Potential/"+str(V)+".png")
#%%
import plotFunctions
importlib.reload(plotFunctions)
from plotFunctions import *
plt.close("all")
pot_types=["duffing","4well","2well","singlewell"]
traj_types=["restarts","langevin"]
temps=[1,5]
lengths=[10,100000]
restarts=[8000,1]
drag=[0.0,0.2]

for pot in pot_types:
        plot_trajectory_fromfile("Trajectories/",pot,'restarts',"1","5", "2000","0.25")
        plt.savefig("Trajectories/Figures/{}_damp0.25_init.png".format(pot))
        plot_trajectory_fromfile("Trajectories/", pot, 'restarts', "1","5", "2000","0.0")
        plt.savefig("Trajectories/Figures/{}_damp0.0_init.png".format(pot))
plot_trajectory_fromfile("Trajectories/", "vanderpol", "vanderpol", "1", "5", "2000","0.0")
plt.savefig("Trajectories/Figures/vanderpol_init.png")

#%%
import plotFunctions
importlib.reload(plotFunctions)
from plotFunctions import *
plot_trajectory_fromfile("Trajectories/", "duffing", "restarts", "1", "10", "8000","0.25")

#%%
from hde import HDE
TrajectoryFile="Trajectories/duffing_restarts_temp1_points10x8000_damping0.25"
traj_list, trng_list = pickle.load(open(TrajectoryFile, 'rb'))
#model_srv=HDE(2,n_components=7,lag_time=1,reversible=False,n_epochs=5,learning_rate=0.001,dropout_rate=0)
#model_srv.fit(traj_list)
#%%
x,y=create_dataset(traj_list,lag_time= 50) ###ARGS TAU
dledmd= EDMD_DL()
dledmd.build(2, 20,100,num_layers=5)###NEIG ARGS
dledmd.train(5,x,y,x,y,10,log_interval=1)

#%%
import KRONIC
importlib.reload(KRONIC)
from KRONIC import *
dt=trng_list[0][1]-trng_list[0][0]
controller_srv=KRONIC_srv(model_srv)
controller_srv.setup_controller([[0],[1]],7,V_s,nfuncs=9,drag=0.25,dt=dt,tau=1)
#controller_srv.get_Bu(np.reshape(np.array([[0],[1]]),(1,-1)),np.transpose(model_srv.transform(np.reshape(np.array([[0],[0]]),(1,-1)))[:,:9]))
sol,tsol,usol=controller_srv.integrate_controlled(xref=np.array([[0],[0]]),x0=np.array([[-0.3],[0]]),length=5,dt=dt/1)
plt.plot(tsol,sol)
plt.figure()
plt.plot(tsol,usol)
#%%
import KRONIC
importlib.reload(KRONIC)
from KRONIC import *
dt=trng_list[0][1]-trng_list[0][0]
controller_edmd=KRONIC_edmd(dledmd)
controller_edmd.setup_controller([[0],[1]],1,V_d,nfuncs=20,drag=0.25,dt=dt,tau=50)
sol,tsol,usol=controller_edmd.integrate_controlled(xref=np.array([[-1],[0]]),x0=np.array([[-0.6],[0]]),length=10,dt=dt/1)
#controller_edmd.get_Bu(np.transpose(np.array([[0],[1]])),dledmd.real_eigenfunctions(np.squeeze(np.reshape(np.array([[0],[0]]),(1,-1)))))

import plotFunctions
importlib.reload(plotFunctions)
from plotFunctions import *

plot_controlled_trajectory(sol,tsol,usol,"Duffing","0.25","1")

#%%

controller_edmd.Q
#%%
import plotFunctions
importlib.reload(plotFunctions)
from plotFunctions import *
nfunc=5
plot_eigenfunctions_srv(model_srv,[-2,2,-2,2],100,nfunc=nfunc)
#plt.savefig("singlewell_restarts_temp1_points10x8000_damping0_srv_eigfunc")
grad = [K.function([model_srv.encoder.inputs], [K.gradients(model_srv.encoder.output[:,i], model_srv.encoder.inputs[0])]) for i in range(0,nfunc)]
plot_gradients_srv(grad,[-2,2,-2,2],30)
#plt.savefig("singlewell_restarts_temp1_points10x8000_damping0_srv_gradeigfunc")

#%%
import plotFunctions
importlib.reload(plotFunctions)
from plotFunctions import *

plot_eigenvalues_srv(model_srv)
#%%
plot_eigenvalues_edmd(dledmd)

plot_reconstr_traj(dledmd,-1.2,0.7,20,"2well_restarts_temp1_points10x8000_damping0.25".partition("_")[0],float(TrajectoryFile.split('g')[-1]),tau=10)
#plot_eigenfunctions_edmd(dledmd,[-args.range_base,args.range_base,-1.5*args.range_base,1.5*args.range_base],500,nfunc=2)
plot_eigenfunctions_edmd(dledmd,[-2,2,-1.5*2,1.5*2],500,nfunc=7)
plot_gradients_edmd(dledmd,[-2,2,-1.5*2,1.5*2],30,7)
