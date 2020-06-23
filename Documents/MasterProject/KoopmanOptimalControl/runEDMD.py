import numpy as np
import plotFunctions
from plotFunctions import *
from edmd_dict_match import *
import argparse
import keras.backend as K
import tensorflow as tf
import pickle
import warnings
warnings.filterwarnings('ignore')
traj_locs="Trajectories/"
#%%


parser = argparse.ArgumentParser()
parser.add_argument('--trajectoryfile', type=str)
parser.add_argument('--n_eig', type=int)##20
parser.add_argument('--tau', type=int)##Range 1 (0.01) 5 (0.05) 10 (0.1) 50 (0.5) 100 (1)

args = parser.parse_args()
traj_file=args.trajectoryfile
tau=args.tau
nfunc=args.n_eig

#%%
traj_file="duffing_restarts_temp1_points10x1000_damping0.25"
nfunc=20
tau=1

#%%

drag=float(traj_file.split('g')[-1])
potential=traj_file.partition("_")[0]
x, t = sy.symbols('x t')
if str(potential) == '4well':
    V = 2 * (x ** 8 + 0.8 * sy.exp(-80 * x ** 2) + 0.2 * sy.exp(-80 * (x - 0.5) ** 2) + 0.5 * sy.exp(
        -40 * (x + 0.5) ** 2));
    xrange = [-1.2, 1.2]
    prange=[-1.1,1.1,-3,3]
    x0list = [[-0.8, 0], [-0.05, 0], [0.85, 0]]
    xreflist = [[0.25, 0], [-0.25, 0], [-0.6, 0]]
elif str(potential) == '2well':
    V = 0.25 * x ** 4 - 0.5 * x ** 2 + 0.25 / 3 * x ** 3 - 0.25 * x
    xrange = [-2, 2]
    prange=[-2,2,-3,3]
    x0list = [[1.8, 0], [-1.5, 0], [0, 0.5]]
    xreflist = [[1, 0], [-0.5, 0]]
elif str(potential) == 'singlewell':
    V = x ** 2
    xrange = [-1, 1]
    prange=[-1,1,-2,2]
    x0list = [[0.75, 0], [-0.5, 0.6], [0, 1]]
    xreflist = [[0, 0], [0.5, 0]]
elif str(potential) == 'duffing':
    V = 0.25 * x ** 4 - 0.5 * x ** 2
    xrange = [-2, 2]
    prange=[-2,2,-3,3]
    x0list = [[0, 0.1], [1.7, 0], [-1.5, 0.1]]
    xreflist = [[1, 0], [-0.5, 0]]
elif str(potential) == 'vanderpol':
    V = 0
    xrange = [-3, 3]
    prange=[-3,3,-6,6]
    x0list=[[1.5,0],[2.5,0],[-1,0.5]]
    xreflist=[[1,0],[0,0]]
else:
    print('Error: no good potential function.')

#%%
#openfile='../Trajectories/Consolidated_{}_{}_traj'.format( 'duffing', 'damped')

#%%
openfile=traj_locs+traj_file
traj_list, trng_list = pickle.load(open(openfile, 'rb'))
traj_list=traj_list[:]
trng_list=trng_list[:]
try:
    dt = trng_list[0][1] - trng_list[0][0]
except:
    dt = trng_list[1] - trng_list[0]
x,y=create_dataset(traj_list,lag_time= tau)
dledmd= EDMD_DL()
dledmd.build(2, nfunc,25,num_layers=5)
dledmd.train(100,x,y,x,y,25000,log_interval=1,opt_interval=15,opt_iterations=500)


#%%
import importlib
import plotFunctions
import edmd_dict_match
importlib.reload(edmd_dict_match)
importlib.reload(plotFunctions)
from plotFunctions import *
from edmd_dict_match import *
plot_eigenvalues_edmd(dledmd)
#plt.savefig("Eigenvalues/{}_edmd_eigval_tau{}.png".format(traj_file,tau))
for x0 in x0list:
    plot_reconstr_traj(dledmd,x0,20,V,xrange,drag,tau,dt)
    #plt.savefig("Reconstruction/{}_edmd_recon_{}_{}_tau{}.png".format(traj_file,x0[0],x0[1],tau))
plot_eigenfunctions_edmd(dledmd,prange,500,nfunc=7)
#avefig("Eigenfunctions/{}_edmd_eigfuc_tau{}.png".format(traj_file,tau))
plot_gradients_edmd(dledmd,prange,30,7)
#plt.savefig("EigenfunctionsGradients/{}_edmd_grad_tau{}.png".format(traj_file,tau))

#%%
import lqr_control
importlib.reload(lqr_control)
from lqr_control import *


ctrllr=CONTROLLER(dledmd)
ctrllr.setup_controller(tau,dt,np.array([[0],[1]]),x)
ctrllr.solve_ADMM(10,500,dledmd.real_eigenfunctions(np.array([[-0.8],[0]]).T).T,tolerance=0.1)





#%%
print("StartControl")
import KRONIC
importlib.reload(KRONIC)
from KRONIC import *
controller_edmd=KRONIC_edmd(dledmd)
for x0 in x0list:
    for xref in xreflist:
        for Qmag in [1,10,100]:
            try:
                controller_edmd.setup_controller([[0], [1]], 1/Qmag, V, nfuncs=nfunc, drag=drag, dt=dt, tau=tau)
                sol,tsol,usol=controller_edmd.integrate_controlled(potential,xref=np.array(xref).reshape(-1,1),x0=np.array(x0).reshape(-1,1),length=10,dt=dt)
                plot_controlled_trajectory(sol,tsol,usol,potential,str(drag),str(tau))
                #plt.savefig("ControlledTraj/" + traj_file + "_edmd_x0{}_xref{}_Qmagn{}_tau{}.png".format(x0[0], xref[0], Qmag, tau))
                print("success Q{} xre{} x0{}".format(Qmag,xref,x0))
            except:
                try:
                    controller_edmd.setup_controller([[0], [1]], 1/Qmag, V, nfuncs=nfunc-1, drag=drag, dt=dt, tau=tau)
                    sol, tsol, usol = controller_edmd.integrate_controlled(potential,xref=np.array(xref).reshape(-1, 1),x0=np.array(x0).reshape(-1, 1), length=10, dt=dt)
                    plot_controlled_trajectory(sol, tsol, usol, potential, str(drag), str(tau))
                    #plt.savefig("ControlledTraj/" + traj_file + "_edmd_x0{}_xref{}_Qmagn{}_tau{}.png".format(x0[0], xref[0], Qmag,tau))
                    print("success Q{} xre{} x0{}".format(Qmag, xref, x0))
                except:
                    print("fail Q{} xre{} x0{}".format(Qmag, xref, x0))
                    continue


