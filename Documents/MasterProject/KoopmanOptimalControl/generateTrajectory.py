import argparse
import numpy as np
import sdeint
from scipy import integrate
import sympy as sy
import sys
import pickle
import warnings
import random
from plotFunctions import *


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--saveloc', type=str)
parser.add_argument('--potential', type=str)
parser.add_argument('--trajectoryType', type=str)
parser.add_argument('--n_starts', type=int)
parser.add_argument('--timestep', type=float)
parser.add_argument('--drag', type=float)
parser.add_argument('--temperature', type=int)
parser.add_argument('--traj_len', type=int)
args = parser.parse_args()

traj_file=args.potential + '_' + args.trajectoryType +  '_temp' + str(args.temperature) + '_points' + str(args.traj_len)+"x"+str(args.n_starts)+"_damping"+str(args.drag)
savefile = args.saveloc + traj_file
print(args.potential)

def obtain_trajectory(potential,xrange, num_traj,traj_generation='damped', drag=0.05,
                          dt=0.01, traj_len=300, temperature=5, debug=False,x0=None):
    acc = -sy.diff(potential)
    acc = sy.lambdify(x, acc)
    traj_list = []
    trng_list = []
    if traj_generation == 'restarts':
        def fun(t, x):
            return np.array([[x[1]], [acc(x[0]) - drag * x[1]]])
        for ith_traj in range(0, num_traj):
            if ith_traj%100==0:
                print("Done",ith_traj,"trajectories")
            if x0==None:
                x0=[random.uniform(xrange[0],xrange[1]),random.uniform(-1,1)]
            traj, trng = integrate_trajectory(fun, x0, integration_method='ode', tend=traj_len, dt=dt)
            traj_list.append(traj)
            trng_list.append(trng)
            x0=None
        if num_traj == 0:
            return traj, trng
        else:
            return traj_list, trng_list
    elif traj_generation == 'langevin':
        k = 8 * 10 ** (-3);
        def f(x, t):
            # import pdb; pdb.set_trace()st
            return np.array([x[1], acc(x[0]) - drag * x[1]])
            # return A.dot(x)
        def G(x, t):
            random = np.sqrt(2 * k * temperature * drag)
            B = np.diag([0, random])
            return B
        for ith_traj in range(0, num_traj):
            if ith_traj%100==0:
                print("Done",ith_traj,"trajectories")
            x0=[random.uniform(xrange[0],xrange[1]),0]
            traj, trng = integrate_trajectory(f, x0, G_func=G, integration_method='sde', tend=traj_len, dt=dt)
            traj_list.append(traj)
            trng_list.append(trng)
        if num_traj == 0:
            return traj, trng
        else:
            return traj_list, trng_list
    elif traj_generation == 'vanderpol':
        def fun(t, x):
            return np.array([[x[1]],[-x[0]+(1-x[0]**2)*x[1]]])
        for ith_traj in range(0, num_traj):
            if ith_traj%100==0:
                print("Done",ith_traj,"trajectories")
            x0=[random.uniform(xrange[0],xrange[1]),random.uniform(-1,1)]
            traj, trng = integrate_trajectory(fun, x0, integration_method='ode', tend=traj_len, dt=dt)
            traj_list.append(traj)
            trng_list.append(trng)
        if num_traj == 0:
            return traj, trng
        else:
            return traj_list, trng_list

    else:
        print('Error: Trajectory Generation is not defined')
        return -1


def integrate_trajectory(f, x0, G_func=None, integration_method='ode', tend=200, dt=0.01):
    tspan = np.arange(0, tend + dt, dt)
    if integration_method is 'ode':
        solver = integrate.ode(f).set_integrator('dop853', method='bdf', atol=0.05, rtol=0.05)
        solver.set_initial_value(x0, 0)
        y = []
        while solver.successful() and solver.t < tend:
            y.append(solver.integrate(solver.t + dt))
        sol = np.reshape(y, (-1, len(y[0])))
        sol = sol.real
        if len(tspan) == len(sol):
            return sol, tspan
        elif len(tspan) > len(sol):
            return sol, tspan[:-1]
    elif integration_method is 'sde':
        if G_func is not None:
            print('Solution of f to x0 is:', f(x0, 0))
            print('Solution of G to x0 at t=0 is:', G_func(x0, 0))
            sol = sdeint.itoint(f, G_func, x0, tspan)
            if len(tspan) == len(sol):
                return sol, tspan
            elif len(tspan) > len(sol):
                return sol, tspan[:-1]
        else:
            print('Error: Did not input an uncertain process for SDE')


x, t = sy.symbols('x t')
if str(args.potential) == '4well':
    V = 2 * (x ** 8 + 0.8 * sy.exp(-80 * x ** 2) + 0.2 * sy.exp(-80 * (x - 0.5) ** 2) + 0.5 * sy.exp(
        -40 * (x + 0.5) ** 2));
    xrange=[-1.2,1.2]
elif str(args.potential) == '2well':
    V = 0.25 * x ** 4 - 0.5 * x ** 2 + 0.25 / 3 * x ** 3 - 0.25 * x
    xrange = [-2, 2]
elif str(args.potential) == 'singlewell':
    V = x ** 2
    xrange = [-2, 2]
elif str(args.potential)== 'duffing':
    V=0.25 * x ** 4 - 0.5 * x ** 2
    xrange = [-2, 2]
elif str(args.potential)== 'vanderpol':
    V=0
    xrange=[-3,3]
else:
    print('Error: no good potential function.')

traj, trng = obtain_trajectory(V, xrange, args.n_starts, traj_generation=args.trajectoryType, drag=args.drag,
                                   dt=args.timestep, traj_len=args.traj_len, temperature=args.temperature, debug=False)
pickle.dump([traj, trng], open(savefile, 'wb'))

plot_trajectory_fromfile(args.saveloc,args.potential,args.trajectoryType,args.temperature,args.traj_len,args.n_starts,args.drag)
plt.savefig(args.saveloc+"Figures/"+traj_file+".png")