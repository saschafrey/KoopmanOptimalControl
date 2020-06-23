from edmd_dict_match import *
from util import *
import controlpy
from matplotlib import pyplot as plt
import scipy.io
import sympy as sy
from hde import HDE
import numpy as np
import keras.backend as K
import cmath
from scipy import integrate

class KRONIC_srv:
    def __init__(self, model: HDE):
        self.model=model


    def setup_controller(self,B,r: float,potential,nfuncs=3,drag=0.25,dt=0.1,tau=1):
        self.drag=drag
        self.r=r
        x, t = sy.symbols('x t')
        self.potential=potential
        acc = -sy.diff(potential)
        self.acc = sy.lambdify(x, acc)
        if nfuncs> self.model.n_components:
            print("Error: trying to use more functions than have been trained")
            return -1
        self.n=np.shape(B)[0]
        self.k=nfuncs
        self.r=np.shape(B)[1]
        self.B=B
        self.Q=np.eye(nfuncs)
        print(self.model.encoder.inputs[0])
        self.gradient_phi= [K.function([self.model.encoder.inputs], [K.gradients(self.model.encoder.output[:,i], self.model.encoder.inputs[0])]) for i in range(0,nfuncs)]
        self.Lambda=np.diag(np.log(self.model.eigenvalues_[:nfuncs])/(tau*dt))##SHOULD ALL BE REAL


    def get_Bu(self,x,phiref):
        x=x.transpose()
        grad_phi=np.squeeze(np.array([self.gradient_phi[i](x) for i in range(0,self.k)]))
        B_phi=np.dot(grad_phi,self.B)
        Klqr,P,_=controlpy.synthesis.controller_lqr_discrete_time(self.Lambda,B_phi,self.Q,np.eye(self.r))
        return -np.dot(self.B,np.dot(Klqr,np.transpose(self.model.transform(x)[:,:self.k])-phiref))

    def integrate_controlled(self, potential, xref=np.array([[0], [0]]), x0 = np.array([[-0.3], [0]]), length = 10, dt = 0.1):
        phi_ref=np.transpose(self.model.transform(np.reshape(xref,(1,-1)))[:,:self.k])
        if potential is "vanderpol":
            def f(t, x):
                bu = self.get_Bu(np.reshape(x, (-1, 1)), phi_ref)
                return np.squeeze(np.array([[x[1]+bu[0]],[-x[0]+(1-x[0]**2)*x[1]+bu[1]]]))
        else:
            def f(t,x):
                bu=self.get_Bu(np.reshape(x,(-1,1)),phi_ref)
                ret=np.squeeze(np.array([[x[1]+bu[0]],[self.acc(x[0])-self.drag*x[1]+bu[1]]]))
                return ret

        sol,tsol=self.integrate_trajectory(f,x0,tend=length,dt=dt)
        np.shape(sol)
        usol=np.squeeze(np.array([self.get_Bu(np.array([sol[i,:]]).reshape((-1,1)),phi_ref) for i in range(0,len(sol))]))
        return sol,tsol,usol

    def integrate_trajectory(self,f, x0, tend=10, dt=0.1):
        tspan = np.arange(0, tend + dt, dt)
        solver = integrate.ode(f).set_integrator('dop853', method='bdf', atol=0.01, rtol=0.01)
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

class KRONIC_edmd:
    def __init__(self, model: EDMD_DL):
        self.model=model

    def setup_controller(self,B,r: float,potential,nfuncs=3,drag=0.0,dt=0.1,tau=2):
        self.drag=drag
        self.r=r
        x, t = sy.symbols('x t')
        self.potential=potential
        acc = -sy.diff(potential)
        self.acc = sy.lambdify(x, acc)
        n,k=self.model.get_dimensions()
        self.n=n
        if nfuncs > k:
            print("Error: trying to use more functions than have been trained")
            return -1
        self.k=nfuncs
        self.r=np.shape(B)[1]
        self.B=B
        Q_bar=np.eye(self.n)
        Q=np.zeros((nfuncs,nfuncs))
        Q[1:self.n+1,1:self.n+1]=Q_bar
        V = self.model.real_eigenvectors()
        self.Q = np.matmul(np.matmul(np.linalg.inv(V[:self.k,:self.k]), Q), np.linalg.inv(np.transpose(V[:self.k,:self.k]))).real
        lamda = np.log(self.model.eigenvalues()) / (dt * tau)
        lamda=lamda[:nfuncs]
        A = np.diag(lamda)
        skip_flag = 0
        # making lambda "real"
        for i, lam in enumerate(lamda):
            if skip_flag == 1:
                skip_flag = 0
                continue
            if ~np.isreal(lam):
                mod,phase=cmath.polar(lam)
                A[i:(i + 2), i:(i + 2)] = np.array([[mod*np.cos(phase), mod*np.sin(phase)], [-mod*np.sin(phase), mod*np.cos(phase)]])
                skip_flag = 1
        self.Lambda=A.real
        return 0

    def get_Bu(self,x,phiref):
        grad_phi=np.transpose(np.squeeze(self.model.real_grad_eigenfunctions(x.transpose())))[:self.k,:]
        B_phi=np.dot(grad_phi,self.B)
        Klqr,P,_=controlpy.synthesis.controller_lqr_discrete_time(self.Lambda,B_phi,self.Q,self.r)
        return -np.dot(self.B,np.dot(Klqr,np.transpose(self.model.real_eigenfunctions(x.transpose()))[:self.k,:]-phiref[:self.k])).real

    def integrate_controlled(self,potential,xref=np.array([[0],[0]]),x0=np.array([[-0.3],[0]]),length=10,dt=0.1):
        phi_ref=np.transpose(self.model.real_eigenfunctions(xref.transpose())[:,:self.k])
        if potential is "vanderpol":
            def f(t, x):
                bu = self.get_Bu(np.reshape(x, (-1, 1)), phi_ref)
                return np.squeeze(np.array([[x[1]+bu[0]],[-x[0]+(1-x[0]**2)*x[1]+bu[1]]]))
        else:
            def f(t,x):
                bu=self.get_Bu(np.reshape(x,(-1,1)),phi_ref)
                return np.squeeze(np.array([[x[1]+bu[0]],[self.acc(x[0])-self.drag*x[1]+bu[1]]]))
        print("test function call", f(0,x0))
        sol,tsol=self.integrate_trajectory(f,x0,tend=length,dt=dt)
        usol = np.squeeze(np.array([self.get_Bu(np.reshape(sol[i],(-1,1)),phi_ref) for i in range(0, len(sol))]))
        return sol,tsol,usol

    def integrate_trajectory(self,f, x0, tend=10, dt=0.1):
        tspan = np.arange(0, tend + dt, dt)
        solver = integrate.ode(f).set_integrator('dop853', method='bdf', atol=0.01, rtol=0.01)
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

