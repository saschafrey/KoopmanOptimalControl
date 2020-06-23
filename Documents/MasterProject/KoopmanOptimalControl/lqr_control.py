from edmd_dict_match import *
from util import *
import cvxpy as cvo
from matplotlib import pyplot as plt
import mosek
import cvxopt
import scipy.io

class CONTROLLER:
    def __init__(self,dledmd):
        self.dledmd_model=dledmd #type: EDMD_DL
        self.xdim,self.kdim=self.dledmd_model.get_dimensions()
        self.__controller_setup=-1
        #Can put flags or other things here.


#* Continuous eigenvalues
    def setup_controller(self,tau,dt,Bx,data):
        #making eigenvalues representative of cont system
        lamda=np.log(self.dledmd_model.eigenvalues())/(dt*tau)
        print(lamda)
        A=np.diag(lamda)
        skip_flag=0
        #making lambda "real"
        for i,lam in enumerate(lamda):
            if skip_flag==1:
                skip_flag=0
                continue
            if ~np.isreal(lam):
                print(i,lam)
                print(A[i:(i+2),i:(i+2)])
                A[i:(i+2),i:(i+2)]=np.array([[lam.real,lam.imag],[-lam.imag,lam.real]])
                skip_flag=1
        #Obtain the matrices Bk for k in Nk
        Bk=np.zeros((self.kdim,self.kdim,self.kdim))
        J=np.zeros((len(data),self.kdim))
        H=np.zeros_like(J)
        grad=self.dledmd_model.real_grad_eigenfunctions(data)
        H=self.dledmd_model.real_eigenfunctions(data)
        J = np.tensordot(grad, np.array(Bx), axes=[[0], [0]])
        for i in range(self.kdim):
            Ji=np.transpose(J)*H[:,i]
            Bk[:,:,i] = np.matmul(Ji,scipy.linalg.pinv(np.transpose(H)))
        #Define the state cost matrix Q and transform it such that only the state parts of the dictionaries are minimised.
        Q=self.build_Q(1)
        V=self.dledmd_model.real_eigenvectors()
        Q_adj=np.matmul(np.matmul(np.linalg.inv(V),Q),np.linalg.inv(np.transpose(V)))
        self.Qadj=Q_adj
        self.Bk=Bk
        self.A=A
        print(self.Bk)
        ##Begin the ADMM Algorithm given by algorithm 1 in the paper.
        self.__controller_setup=0
        scipy.io.savemat('A.mat', mdict={'Apy': A})
        scipy.io.savemat('Bk.mat', mdict={'Bkpy':Bk})
        scipy.io.savemat('Qadj.mat', mdict={'Qpy': Q_adj})
        return 0

    def solve_ADMM(self,rho,iterations,z_0,tolerance=0.1):
        scipy.io.savemat('z0.mat', mdict={'z0py': z_0})
        print(iterations)
        #rho is T in MATLB and rho in paper.
        W=np.zeros((self.kdim,self.kdim)) #W as defined in the control paper, lagrangian weights. Z in Matlab
        a=np.zeros((self.kdim,1)) #alpha from the paper, control weights in Ac=Az+sum(ak*Bk*z)
        P=np.eye(self.kdim)
        error=[]
        for k in range(0,iterations):
            print(k)
            prev_P=P
            prev_a=a
            prev_W=W
            ### Step 1 Optimisation problem, finding best alpha and Z given fixed P and W. Z is a relaxation to make
            ### the problem convex
            a_cvo=cvo.Variable((self.kdim,1)) #a_n in matlab
            Z_cvo=cvo.Variable((self.kdim,self.kdim),symmetric=True) #R_n in matlab
            A_cl=self.A.real
            print("Acl:", A_cl)
            for i in range(0,self.kdim):
                A_cl=A_cl+a_cvo[i]*self.Bk[:,:,i].real
            print("Acl:", A_cl)
            La= cvo.trace(cvo.matmul(W,Z_cvo))+(rho/2)*cvo.power(cvo.norm(Z_cvo,'fro'),2)
            print(La.is_convex())
            objective=cvo.Minimize(La)
            constraints= [cvo.bmat([[Z_cvo-cvo.matmul(cvo.transpose(A_cl),P)-cvo.matmul(P,A_cl)-self.Qadj.real, a_cvo],
                                    [cvo.transpose(a_cvo), np.atleast_2d(1)]])>>0]
            #constraints=[cvo.matmul(cvo.transpose(A_cl),P)+cvo.matmul(P,A_cl)+self.Qadj.real+cvo.matmul(a_cvo,cvo.transpose(a_cvo))<<Z_cvo]

            #print(constraints)
            prob = cvo.Problem(objective, constraints)
            prob.solve(solver=cvo.CVXOPT,verbose=True)  # Returns the optimal value.
            print("status 1:", prob.status)
            print(a_cvo.value)

            #print("optimal value", prob.value)
            #print("optimal var", np.shape(Z_cvo.value), np.shape(a_cvo.value))
            a=a_cvo.value
            ### Step 2 find optimal for P with fixed rest, particularly fixed alpha and W. Convex problem by nature.
            A_cl=self.A.real
            for i in range(0,self.kdim):
                A_cl=A_cl+a[i]*self.Bk[:,:,i] ##Redefining closed loop not as a var.
            A_cl=A_cl.real
            P_cvo=cvo.Variable((self.kdim,self.kdim),symmetric=True)
            long_expr=cvo.matmul(cvo.transpose(A_cl),P_cvo)+cvo.matmul(P_cvo,A_cl)+self.Qadj.real+cvo.matmul(a,cvo.transpose(a))
            Lp1=cvo.matmul(cvo.matmul(cvo.transpose(z_0.real),P_cvo),z_0.real)
            Lp2=cvo.trace(cvo.matmul(W,long_expr))
            Lp3=(rho/2)*cvo.norm(long_expr,'fro')
            Lp=Lp1+Lp2+Lp3
            prob2=cvo.Problem(cvo.Minimize(0),[P_cvo>>0])
            prob2.solve(solver=cvo.MOSEK)
            print("status 2:", prob2.status)
            #print("optimal value", prob2.value)
            #print("optimal var", np.shape(P_cvo.value))
            P=P_cvo.value
            ### Step 3: Update the weights.
            #print('Acl',A_cl,'P',P)
            #print(np.shape(np.matmul(np.transpose(A_cl),P)))
            W= W + rho*(np.matmul(np.transpose(A_cl),P)+np.matmul(P,A_cl)+self.Qadj.real+np.matmul(a,np.transpose(a)))
            curr_err=np.linalg.norm(prev_a-a,2)+np.linalg.norm(prev_P-P,'fro')+np.linalg.norm(prev_W-W,'fro')
            error.append(curr_err)
            if curr_err<tolerance:
                break
        #print(a)
        plt.plot(error)





        return a,prev_a






    def build_Q(self,Q_magn):
        Q_bar=np.eye(self.xdim)*Q_magn
        Q=np.zeros((self.kdim,self.kdim))
        Q[1:self.xdim+1,1:self.xdim+1]=Q_bar
        return Q