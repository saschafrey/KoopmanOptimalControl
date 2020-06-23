from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from cycler import cycler
import pickle
from edmd_dict_match import *
import sympy as sy
import keras.backend as K
import tensorflow as tf
default_cycler = (cycler(color=['#800000', '#1D5F83', '#767676', '#000000','#70AD47','#7030A0']))
plt.rc('axes', prop_cycle=default_cycler)

def plot_trajectory_fromfile(saveloc,potential,type,temp,length,number,damping):
    TrajectoryFile="{}{}_{}_temp{}_points{}x{}_damping{}".format(saveloc,potential,type,temp,length,number,damping)
    traj_list, trng_list = pickle.load(open(TrajectoryFile, 'rb'))
    if type == "langevin":
        traj_conc=np.squeeze(traj_list)
        generator_str="langevin dynamics at temperature {}".format(temp)
    else:
        traj_conc = np.concatenate(traj_list)
        generator_str= "{} trajectories of length t={} at random initial states".format(number,length)

    try:
        dt=trng_list[0][1]-trng_list[0][0]
    except:
        dt = trng_list[1] - trng_list[0]
    trng_conc = np.arange(0, len(traj_conc) * dt, dt)
    fig, ax = plt.subplots(2, 1, figsize=(7, 8))
    index=int(20/dt)
    ax[0].plot(trng_conc[:index], traj_conc[:index, 0])
    ax[0].plot(trng_conc[:index], traj_conc[:index, 1], linewidth=0.2)
    ax[0].set_ylabel('State')
    ax[0].set_xlabel('Time')
    ax[1].set_ylabel('Velocity')
    ax[1].set_xlabel('Position')
    ax[0].legend(['Position', 'Velocity'], loc='lower right')
    ax[1].scatter(traj_conc[:, 0], traj_conc[:, 1], 0.01)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    ax[0].set_title(
        'Trajectory and phase-plot for {} potential\n'
        'generated with {} \n Damping factor set to {}'.format(potential,generator_str,damping))

    return 0

def plot_potential(V,xlist,ax :plt.Axes,fig,title="Potential Function",xlabel="x [-]",ylabel="V(x) [-]"):
    Vlist=[V(x) for x in xlist]
    ax.plot(xlist,Vlist)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_eigenfunctions_srv(srv_model,ranges,bins,nfunc=1):
    xy_list,xx,yy,grids=create_grid(ranges,bins)
    state=srv_model.transform(xy_list)
    if nfunc==1:
        fig,ax=plt.subplots()
        plot_eigenfunction(state[:,0],bins,fig,xx,yy,ax=ax)
    else:
        fig,ax=plt.subplots(nfunc,1)
        for i in range(0,nfunc):
            plot_eigenfunction(state[:, i], bins, fig, xx, yy, ax=ax[i])
            if not i == nfunc - 1:
                ax[i].axes.get_xaxis().set_ticks([])
        ax[0].set_title("First {} eigenfunctions discovered using SRV".format(nfunc))
    return 0

def plot_eigenfunctions_edmd(edmd_model: EDMD_DL,ranges,bins,nfunc=1):
    xy_list,xx,yy,grids=create_grid(ranges,bins)
    state=edmd_model.real_eigenfunctions(xy_list)
    #state=edmd_model.eigenfunctions(xy_list).real
    if nfunc==1:
        fig,ax=plt.subplots()
        plot_eigenfunction(state[:,0],bins,fig,xx,yy,ax=ax)
    else:
        fig,ax=plt.subplots(nfunc,1)
        for i in range(0,nfunc):
            plot_eigenfunction(state[:, i], bins, fig, xx, yy, ax=ax[i])
            if not i == nfunc - 1:
                ax[i].axes.get_xaxis().set_ticks([])
        ax[0].set_title("First {} eigenfunctions discovered using EDMD-DL".format(nfunc))
    return 0

def plot_gradients_srv(func,ranges,bins,nfunc):
    xy_list, xx, yy, grids = create_grid(ranges, bins)
    #sess = tf.InteractiveSession()
    #sess.run(tf.initialize_all_variables())
    #evaluated_gradients = np.squeeze(sess.run(gradients, feed_dict={srv_model.encoder.input: xy_list}))
    fig,ax=plt.subplots(nfunc,1)
    for i,axi in enumerate(ax):
        plot_eigenfunction_quiver(-np.squeeze(func[i](xy_list)), bins, fig,xx,yy, levels=None, ax=axi, title=None,
                              color_label=None, c_lim=None, xlabel='x', ylabel='v', include_cbar=True)
        if not i==nfunc-1:
            ax[i].axes.get_xaxis().set_ticks([])
    ax[0].set_title("First {} gradients of eigenfunctions discovered using SRV".format(nfunc))

def plot_gradients_edmd(edmd_model: EDMD_DL,ranges,bins,nfunc):
    xy_list, xx, yy, grids = create_grid(ranges, bins)
    #sess = tf.InteractiveSession()
    #sess.run(tf.initialize_all_variables())
    #evaluated_gradients = np.squeeze(sess.run(gradients, feed_dict={srv_model.encoder.input: xy_list}))
    evaluated_gradients=edmd_model.real_grad_eigenfunctions(xy_list)
    fig,ax=plt.subplots(nfunc,1)
    for i,axi in enumerate(ax):
        plot_eigenfunction_quiver(np.transpose(evaluated_gradients[:,:,i]), bins, fig,xx,yy, levels=None, ax=axi, title=None,
                              color_label=None, c_lim=None, xlabel='x', ylabel='v', include_cbar=True)
        if not i==nfunc-1:
            ax[i].axes.get_xaxis().set_ticks([])
    ax[0].set_title("First {} gradients of eigenfunctions discovered using EDMD-DL".format(nfunc))

def plot_eigenvalues_srv(srv_model):
    eigvals=srv_model.eigenvalues_
    fig,ax=plt.subplots()
    ax.scatter(eigvals.real,eigvals.imag,marker="x")
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title("Eigenvalues returned by SRV on Argand Diagram")

def plot_eigenvalues_edmd(edmd_model: EDMD_DL):
    eigvals=edmd_model.eigenvalues()
    fig,ax=plt.subplots()
    ax.scatter(eigvals.real,eigvals.imag,marker="x")
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title("Eigenvalues returned by EDMD-DL on Argand Diagram")

def plot_reconstr_traj(edmd_model: EDMD_DL,x0: list,length,V,xrange,drag,tau,dt):
    if V==0:
        t_real, trng_real = obtain_local_trajectory(V, xrange, 1, 'vanderpol', drag, dt * tau, length,
                                                    x0=x0)
    else:
        t_real,trng_real=obtain_local_trajectory(V,xrange,1,'restarts',drag,dt*tau,length,x0=x0)

    fig,ax=plt.subplots(1,2)
    ax[0].plot(edmd_model.predict(np.array(x0),int(length/(dt*tau))))
    ax[1].plot(np.squeeze(t_real))
    ax[0].set_title("Reconstructed Trajectory \n With Koopman functions")
    ax[1].set_title("Real Integrated Trajectory")
    ax[0].set_xlabel("Points")
    ax[1].set_xlabel("Points")
    ax[0].set_ylabel("State")
    ax[1].legend(["Position","Velocity"])
    return 0




def create_grid(ranges, nbin):
    grids = []
    for i in range(int(len(ranges) / 2)):
        grids.append(np.linspace(ranges[2 * i], ranges[2 * i + 1], nbin))
    xx, yy = np.meshgrid(grids[0], grids[1])
    xy_list = np.concatenate([xx.reshape(nbin ** 2, 1), yy.reshape(nbin ** 2, 1)], axis=-1)
    return xy_list, xx, yy, grids

def plot_eigenfunction(state, bins, fig,xx,yy, levels=None, ax=None, title=None,
                       color_label=None, c_lim=None, xlabel='x', ylabel='v', include_cbar=True):

    # im = ax.contour(xx, yy, signal.medfilt(state.reshape(bins, bins),kernel_size=15), levels=levels, cmap='viridis_r' )
    im = ax.contourf(xx, yy, state.reshape(bins, bins), levels=levels, cmap='viridis_r')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if not title is None:
        ax.set_title(title, fontsize=12)
    #ax.axes.get_yaxis().set_ticks([])
    if include_cbar:
        cbar = fig.colorbar(im, ax=ax)
        if not color_label is None: cbar.set_label(color_label, fontsize=12)
    return ax

def plot_eigenfunction_quiver(grad, bins, fig,XX,YY, levels=None, ax=None, title=None,
                                      color_label = None, c_lim=None, xlabel='x', ylabel='v', include_cbar=True):
    im = ax.quiver(XX,YY,grad[:,0],grad[:,1],np.abs(np.sum(grad,axis=1)))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    #ax.axes.get_yaxis().set_ticks([])
    if include_cbar:
        cbar = fig.colorbar(im, ax=ax)
        if not color_label is None: cbar.set_label(color_label, fontsize=12)
    return ax

def plot_controlled_trajectory(sol,tsol,usol,potential,damp,tau):
    fig,ax=plt.subplots(1,2,figsize=(10,6))
    ax[0].plot(tsol,sol)
    ax[0].set_title("Controlled Trajectory")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("State x")
    ax[0].legend(["Position","Velocity"])
    ax[1].plot(tsol,usol)
    ax[1].set_title("Control Inputs")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Control Input Bu")
    ax[1].legend(["Ux1","Ux2"])
    fig.tight_layout(pad=0.5,h_pad=1.5,rect=[0,0,1,0.9])
    fig.suptitle("Controlled Trajectory for {} potential with \n damping {} and tau {}".format(potential,damp,tau))





from scipy import integrate
import sdeint
import random

def obtain_local_trajectory(potential,xrange, num_traj,traj_generation='damped', drag=0.05,
                          dt=0.01, traj_len=300, temperature=5, debug=False,x0=None):
    x, t = sy.symbols('x t')
    acc = -sy.diff(potential)
    acc = sy.lambdify(x, acc)
    traj_list = []
    trng_list = []
    if traj_generation == 'restarts':
        def fun(t, x):
            return np.array([[x[1]], [acc(x[0]) - drag * x[1]]])
        for ith_traj in range(0, num_traj):
            if x0==None:
                x0=[random.uniform(xrange[0],xrange[1]),random.uniform(-1,1)]
            traj, trng = integrate_local_trajectory(fun, x0, integration_method='ode', tend=traj_len, dt=dt)
            traj_list.append(traj)
            trng_list.append(trng)
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
            x0=[random.uniform(xrange[0],xrange[1]),0]
            traj, trng = integrate_local_trajectory(f, x0, G_func=G, integration_method='sde', tend=traj_len, dt=dt)
            print(traj)
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
            x0=[random.uniform(xrange[0],xrange[1]),random.uniform(-1,1)]
            traj, trng = integrate_local_trajectory(fun, x0, integration_method='ode', tend=traj_len, dt=dt)
            traj_list.append(traj)
            trng_list.append(trng)
        if num_traj == 0:
            return traj, trng
        else:
            return traj_list, trng_list

    else:
        print('Error: Trajectory Generation is not defined')
        return -1


def integrate_local_trajectory(f, x0, G_func=None, integration_method='ode', tend=200, dt=0.01):
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