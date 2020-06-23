import numpy as np
import plotFunctions
from plotFunctions import *
from hde import HDE
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
parser.add_argument('--n_eig', type=int)
parser.add_argument('--tau', type=int)
parser.add_argument('--reversible', type=bool)
args = parser.parse_args()

traj_file=args.trajectoryfile
nfunc=args.n_eig
tau=args.tau
rev=args.reversible
#%%
traj_file="2well_langevin_temp70_points10000x1_damping0.25"
nfunc=7
tau=100
rev=False


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



openfile=traj_locs+traj_file
traj_list, trng_list = pickle.load(open(openfile, 'rb'))
try:
    dt = trng_list[0][1] - trng_list[0][0]
except:
    dt = trng_list[1] - trng_list[0]
model_srv=HDE(2,n_components=nfunc,lag_time=tau,reversible=rev,n_epochs=100,learning_rate=0.001,dropout_rate=0,hidden_layer_depth=5,batch_size=50000,)
model_srv.fit(traj_list)
#%%
import importlib
plot_eigenvalues_srv(model_srv)
plt.savefig("Eigenvalues/{}_srv_eigfunc_tau{}.png".format(traj_file,tau))
plot_eigenfunctions_srv(model_srv,prange,500,nfunc=nfunc)
plt.savefig("Eigenfunctions/{}_srv_eigfunc_tau{}.png".format(traj_file,tau))
grad = [K.function([model_srv.encoder.inputs], [K.gradients(model_srv.encoder.output[:,i], model_srv.encoder.inputs[0])]) for i in range(0,nfunc)]
plot_gradients_srv(grad,prange,30,nfunc)
plt.savefig("EigenfunctionsGradients/{}_srv_gradeigfunc_tau{}.png".format(traj_file,tau))

#%%
import KRONIC
importlib.reload(KRONIC)
from KRONIC import *

print("StartControl")
import KRONIC
importlib.reload(KRONIC)
from KRONIC import *
controller_srv=KRONIC_srv(model_srv)
x0=x0list[0]
xref=xreflist[0]
for Qmag in [1,100]:
    try:
        controller_srv.setup_controller([[0], [1]], 1/Qmag, V, nfuncs=5, drag=drag, dt=dt, tau=tau,)
        sol,tsol,usol=controller_srv.integrate_controlled(potential,xref=np.array(xref).reshape(-1,1),x0=np.array(x0).reshape(-1,1),length=10,dt=dt)
        plot_controlled_trajectory(sol,tsol,usol,potential,str(drag),str(tau))
        plt.savefig("ControlledTraj/" + traj_file + "_x0{}_xref{}_Qmagn{}_tau{}.png".format(x0[0], xref[0], Qmag, tau))
        print("success Q{} xre{} x0{}".format(Qmag,xref,x0))
    except:
        print("fail Q{} xre{} x0{}".format(Qmag, xref, x0))

