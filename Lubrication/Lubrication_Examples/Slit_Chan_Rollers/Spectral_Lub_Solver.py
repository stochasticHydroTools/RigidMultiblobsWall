'''
Class to handle Lubrication solve
'''
import numpy as np
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
from functools import partial
import copy
import inspect
import time
import general_application_utils
import sys
from mobility import mobility as mob
import pyamg
import scipy.sparse as sp
from sksparse.cholmod import cholesky
import Lubrication_Class_Spectral_SC as Lub_cc #Lubrication_Class_Spectral
from stochastic_forcing import stochastic_forcing as stochastic
from quaternion_integrator.quaternion import Quaternion

from numba import jit, njit, prange


class Spectral_Lub_Solver(object):
    '''
    Class to handle Lubrication solve
    '''

    def __init__(self, bodies, a, eta, cutoff, periodic_length, debye_length=1e-4):
        '''
        Constructor. Take arguments like ...
        '''
        # Location as np.array.shape = 3
        self.bodies = bodies
        self.periodic_length = periodic_length
        self.tolerance = 1e-08
        self.eta = eta
        self.a = a
        self.kT = 0.0041419464
        self.dt = 1.0
        self.cutoff = cutoff
        self.cutoff_wall = 1.0e10  # 1.45
        self.num_rejections_wall = 0
        self.num_rejections_jump = 0
        self.debye_length = debye_length
        self.delta = 1e-3

        self.reflect_forces = np.zeros(6*len(bodies))

        self.LC = Lub_cc.Lubrication(debye_length)
        self.R_MB = None
        self.R_Sup = None
        self.Delta_R = None

        self.Delta_R_cut = None
        self.Delta_R_cut_wall = None
        

        self.solver = None

    def project_to_periodic_image(self, r, L):
        '''
        Project a vector r to the minimal image representation
        centered around (0,0,0) and of size L=(Lx, Ly, Lz). If 
        any dimension of L is equal or smaller than zero the 
        box is assumed to be infinite in that direction.
        '''
        if L is None:
            exit()

        if L is not None:
            for i in range(3):
                if(L[i] > 0):
                    r[i] = r[i] - int(r[i] / L[i] + 0.5 *
                                      (int(r[i] > 0) - int(r[i] < 0))) * L[i]
        # print r
        return r

    def put_r_vecs_in_periodic_box(self, r_vecs_np, L):
        r_vecs = np.copy(r_vecs_np)
        for r_vec in r_vecs:
            for i in range(3):
                if L[i] > 0:
                    while r_vec[i] < 0:
                        r_vec[i] += L[i]
                    while r_vec[i] > L[i]:
                        r_vec[i] -= L[i]
        return r_vecs

    def Init_DPStokesSolver(self, z_max, r_vecs_np=None, domain='DPBW'):
        '''
        Setup DPStokes Solver. Must pass in a maximum height
        '''
        from DoublyPeriodicStokes.python_interface.common_interface_wrapper import FCMJoint as FCMJoint
        start = time.time()
        if r_vecs_np is None:
            r_vecs_np = [b.location for b in self.bodies]
        r_vecs = list(self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length))
        num_particles = len(r_vecs)
        
        
        L = self.periodic_length
        
        device = 'gpu' #change to cpu for walls
        domType = domain#'DPBW' # for bottom wall
        has_torque = True
        #Simulation domain limits
        xmin = 0.0
        xmax = L[0]
        ymin = 0.0
        ymax = L[1]
        zmin = 0.0
        zmax = z_max
        # Initialize the solver with all the parameters
        self.solver = FCMJoint(device)
        #Initialize can be called several times in order to change the parameters
        self.solver.Initialize(numberParticles=num_particles, hydrodynamicRadius=self.a, kernType=0,
                        domType=domType, has_torque=has_torque,
                        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                        viscosity=self.eta, optInd=0, ref=False)
        
        print('solver successfully initialized')
        self.solver.SetPositions(np.array(r_vecs).flatten())
        forces_test = 0.0*np.array(r_vecs).flatten() + 1.0
        print('test')
        #MFtest = self.solver.Mdot(forces_test)
        

    def Set_R_Mats(self, r_vecs_np=None):
        '''
        Set lubrication matricees in sparse csc format for the class members
        '''
        
        if r_vecs_np is None:
            r_vecs_np = [b.location for b in self.bodies]
        r_vecs = list(self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length))
        
        self.solver.SetPositions(np.array(r_vecs).flatten())
        
        num_particles = len(r_vecs)
        r_tree = spatial.cKDTree(
            np.array(r_vecs), boxsize=self.periodic_length)

        start = time.time()
        neighbors = []
        for j in range(num_particles):
            s1 = r_vecs[j]
            idx = r_tree.query_ball_point(s1, r=self.cutoff*self.a)
            idx_trim = [i for i in idx if i > j]
            neighbors.append(idx_trim)
        end = time.time()
        #print('neighbor list time : '+ str((end - start)))

        small = 0.5*6.0*np.pi*self.eta*self.a*self.tolerance

        start = time.time()
        data = []
        rows = []
        cols = []
        self.LC.ResistCOO(r_vecs, neighbors, self.a, self.eta, self.cutoff,
                          self.cutoff_wall, self.periodic_length, False, data, rows, cols)
        end = time.time()
        #print('C++ R pair time : '+ str((end - start)))
        start = time.time()
        if data:
            R_MB_coo_cut = sp.coo_matrix(
                (np.array(data,np.float64), (np.array(rows,np.int32), np.array(cols,np.int32))), shape=(6*num_particles, 6*num_particles), copy=False)
            R_MB_cut = R_MB_coo_cut.tocsc()
        else:
            # sp.csc_matrix(shape=(6*num_particles, 6*num_particles))
            R_MB_cut = sp.diags(
                small*np.ones(6*num_particles), 0, format='csc')
            
        end = time.time()
        #print('R pair time : '+ str((end - start)))

        start = time.time()
        data = []
        rows = []
        cols = []
        self.LC.ResistCOO_wall(r_vecs, self.a, self.eta, self.cutoff_wall,
                               self.periodic_length, False, data, rows, cols)
        if data:
            R_MB_coo_cut_wall = sp.coo_matrix(
                (np.array(data,np.float64), (np.array(rows,np.int32), np.array(cols,np.int32))), shape=(6*num_particles, 6*num_particles), copy=False)
            R_MB_cut_wall = R_MB_coo_cut_wall.tocsc()
        else:
            # sp.csc_matrix(shape=(6*num_particles, 6*num_particles))
            R_MB_cut_wall = sp.diags(
                small*np.ones(6*num_particles), 0, format='csc')

        self.R_MB = R_MB_cut + R_MB_cut_wall
        end = time.time()
        #print('R wall time : '+ str((end - start)))

        data = []
        rows = []
        cols = []
        self.LC.ResistCOO(r_vecs, neighbors, self.a, self.eta, self.cutoff,
                          self.cutoff_wall, self.periodic_length, True, data, rows, cols)
        if data:
            R_sup_coo_cut = sp.coo_matrix(
                (np.array(data,np.float64), (np.array(rows,np.int32), np.array(cols,np.int32))), shape=(6*num_particles, 6*num_particles), copy=False)
            R_sup_cut = R_sup_coo_cut.tocsc()
        else:
            # sp.csc_matrix(shape=(6*num_particles, 6*num_particles))
            R_sup_cut = sp.diags(
                small*np.ones(6*num_particles), 0, format='csc')

        data = []
        rows = []
        cols = []
        self.LC.ResistCOO_wall(r_vecs, self.a, self.eta, self.cutoff_wall,
                               self.periodic_length, True, data, rows, cols)
        if data:
            R_sup_coo_cut_wall = sp.coo_matrix(
                (np.array(data,np.float64), (np.array(rows,np.int32), np.array(cols,np.int32))), shape=(6*num_particles, 6*num_particles), copy=False)
            R_sup_cut_wall = R_sup_coo_cut_wall.tocsc()
        else:
            # sp.csc_matrix(shape=(6*num_particles, 6*num_particles))
            R_sup_cut_wall = sp.diags(
                small*np.ones(6*num_particles), 0, format='csc')

        self.R_Sup = R_sup_cut + R_sup_cut_wall

        self.Delta_R = self.R_Sup - self.R_MB

        self.Delta_R_cut = R_sup_cut - R_MB_cut
        self.Delta_R_cut_wall = R_sup_cut_wall - R_MB_cut_wall
        #################################
        #print(R_sup_cut_wall.toarray())
        #print(R_MB_cut_wall.toarray())
        #print(self.Delta_R_cut_wall.toarray())

        

    def Compute_DeltaR(self, r_vecs_np):
        '''
        Return lubrication matricees in sparse csc format
        '''
        start = time.time()
        if r_vecs_np is None:
            r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        num_particles = len(r_vecs)
        r_tree = spatial.cKDTree(
            np.array(r_vecs), boxsize=self.periodic_length)

        small = 6.0*np.pi*self.eta*self.a*self.tolerance

        neighbors = []
        for j in range(num_particles):
            s1 = r_vecs[j]
            idx = r_tree.query_ball_point(s1, r=self.cutoff*self.a)
            idx_trim = [i for i in idx if i > j]
            neighbors.append(idx_trim)

        data = []
        rows = []
        cols = []
        self.LC.ResistCOO(r_vecs, neighbors, self.a, self.eta, self.cutoff,
                          self.cutoff_wall, self.periodic_length, False, data, rows, cols)
        if data:
            R_MB_coo_cut = sp.coo_matrix(
                (data, (rows, cols)), shape=(6*num_particles, 6*num_particles))
            R_MB_cut = R_MB_coo_cut.tocsc()
        else:
            # sp.csc_matrix(shape=(6*num_particles, 6*num_particles))
            R_MB_cut = sp.diags(
                small*np.ones(6*num_particles), 0, format='csc')

        data = []
        rows = []
        cols = []
        self.LC.ResistCOO_wall(r_vecs, self.a, self.eta, self.cutoff_wall,
                               self.periodic_length, False, data, rows, cols)
        if data:
            R_MB_coo_cut_wall = sp.coo_matrix(
                (data, (rows, cols)), shape=(6*num_particles, 6*num_particles))
            R_MB_cut_wall = R_MB_coo_cut_wall.tocsc()
        else:
            # sp.csc_matrix(shape=(6*num_particles, 6*num_particles))
            R_MB_cut_wall = sp.diags(
                small*np.ones(6*num_particles), 0, format='csc')

        R_MB = R_MB_cut + R_MB_cut_wall

        data = []
        rows = []
        cols = []
        self.LC.ResistCOO(r_vecs, neighbors, self.a, self.eta, self.cutoff,
                          self.cutoff_wall, self.periodic_length, True, data, rows, cols)
        if data:
            R_sup_coo_cut = sp.coo_matrix(
                (data, (rows, cols)), shape=(6*num_particles, 6*num_particles))
            R_sup_cut = R_sup_coo_cut.tocsc()
        else:
            # sp.csc_matrix(shape=(6*num_particles, 6*num_particles))
            R_sup_cut = sp.diags(
                small*np.ones(6*num_particles), 0, format='csc')

        data = []
        rows = []
        cols = []
        self.LC.ResistCOO_wall(r_vecs, self.a, self.eta, self.cutoff_wall,
                               self.periodic_length, True, data, rows, cols)
        if data:
            R_sup_coo_cut_wall = sp.coo_matrix(
                (data, (rows, cols)), shape=(6*num_particles, 6*num_particles))
            R_sup_cut_wall = R_sup_coo_cut_wall.tocsc()
        else:
            # sp.csc_matrix(shape=(6*num_particles, 6*num_particles))
            R_sup_cut_wall = sp.diags(
                small*np.ones(6*num_particles), 0, format='csc')

        R_Sup = R_sup_cut + R_sup_cut_wall

        Delta_R = R_Sup - R_MB

        end = time.time()
        # print 'mat create time : '+ str((end - start))

        return Delta_R

    def Wall_Mobility_Mult(self, X, r_vecs_np=None):
        '''
        Multiply a vector X  of forces and torques by the RPB mobility. X should be formatted according to [F_1 T_1 F_2 T_2 ...]^{T} 
        '''
        
        if r_vecs_np is None:
            r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        
        r_vecs = np.squeeze(r_vecs)
        
        if r_vecs_np is not None:
            self.solver.SetPositions(np.array(r_vecs).flatten())
            
        num_particles = r_vecs.size // 3 

        FT = X.reshape(num_particles, 6)
        F = FT[:, 0:3].flatten()
        T = FT[:, 3:6].flatten()

        start = time.time()
        U, W = self.solver.Mdot(forces=F, torques=T)
        end = time.time()
        #print('time for mult op: '+ str((end - start)))


        UW = np.concatenate((U.reshape(num_particles, 3),
                            W.reshape(num_particles, 3)), axis=1)
        Mob_U = UW.flatten()

        out = Mob_U
        

        return out
    
    def Form_Mobility(self, r_vecs):
        '''
        Computes the RPY mobility
        '''
        Dim = 6*(len(r_vecs))
        I = np.eye(Dim)
        Mob = np.zeros((Dim, Dim))
        for k in range(Dim):
            vel = self.Wall_Mobility_Mult(I[:, k], r_vecs_np=r_vecs)
            Mob[:, k] = vel
        return Mob
    
    
    def Wall_Mobility_Mult_Parts(self, X, r_vecs_np=None):
        '''
        Multiply a vector X  of forces and torques by the RPB mobility. X should be formatted according to [F_1 T_1 F_2 T_2 ...]^{T} 
        '''
        start = time.time()
        if r_vecs_np is None:
            r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        num_particles = r_vecs.shape[0]

        FT = X.reshape(num_particles, 6)
        F = FT[:, 0:3].flatten()
        T = FT[:, 3:6].flatten()

        self.solver.Mdot(forces=F, torques=T)

        end = time.time()
        # print 'time for mult op: '+ str((end - start))

        return M_FT, F, T

    def Lub_Mobility_RFD_RHS(self):
        '''
        Computes the RHS for a Lubrication Corrected Euler Maruyama scheme. Two RHS vectors are returned to be used in the function Lubrucation_solve
        '''
        r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)

        W = np.random.randn(6*r_vecs.shape[0], 1)
        Wrfd = np.reshape(W, (r_vecs.shape[0], 6))
        Wrfd = Wrfd[:, 0:3]

        Qp = r_vecs + (self.delta/2.0)*Wrfd
        self.put_r_vecs_in_periodic_box(Qp, self.periodic_length)
        Qm = r_vecs - (self.delta/2.0)*Wrfd
        self.put_r_vecs_in_periodic_box(Qm, self.periodic_length)

        DRp = self.Compute_DeltaR(list(Qp))
        DRm = self.Compute_DeltaR(list(Qm))

        UWrfd, res_list = self.Lubrucation_RFD_solve(X=W)

        MUW = self.Wall_Mobility_Mult(UWrfd)
        DRpMW = DRp.dot(MUW)
        DRmMW = DRm.dot(MUW)

        RHS_Xm = (1.0/self.delta)*(DRmMW-DRpMW)

        MpW = self.Wall_Mobility_Mult(UWrfd, r_vecs_np=Qp)
        MmW = self.Wall_Mobility_Mult(UWrfd, r_vecs_np=Qm)
        
        self.solver.SetPositions(np.array(r_vecs).flatten())

        RHS_X = (1.0/self.delta)*(MpW-MmW)

        return RHS_Xm, RHS_X

    def Lub_Mobility_Centered_RFD(self, W=None):
        '''
        Uses dense linear algebra and centered RFD to compute the divergence of the Lub. corrected mobility 
        '''
        r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        if W is None:
            # np.random.uniform(-1.0,1.0,6*r_vecs.shape[0]) #
            W = np.random.randn(6*r_vecs.shape[0])
        L = self.periodic_length
        Wrfd = np.reshape(W, (r_vecs.shape[0], 6))
        Wrfd = Wrfd[:, 0:3]

        Qp = r_vecs + (self.delta/2.0)*Wrfd
        self.put_r_vecs_in_periodic_box(Qp, self.periodic_length)
        Qm = r_vecs - (self.delta/2.0)*Wrfd
        self.put_r_vecs_in_periodic_box(Qm, self.periodic_length)

        MLub = self.Form_Lub_Mobility_Dense(Qp)
        Mp = np.dot(MLub, W)

        MLub = self.Form_Lub_Mobility_Dense(Qm)
        Mm = np.dot(MLub, W)

        self.solver.SetPositions(np.array(r_vecs).flatten())
  
        RFD = (1.0/self.delta)*(Mp-Mm)

        return RFD




    def Lub_Mobility_Root_RHS(self, print_res=False):
        '''
        Returns both DR^{1/2} and M_RPY^{1/2} to be used in the function Lubrucation_solve to compute the square root of the lubrication corrected mobility
        '''
        Dim = self.Delta_R.shape[1]

        W1 = np.transpose(np.random.randn(1, Dim))
        W2 = np.transpose(np.random.randn(1, Dim))

        start = time.time()
        small = 1e-5 #*6.0*np.pi*self.eta*self.a*self.tolerance
        Eig_Shift_DR_cut = self.Delta_R + \
            sp.diags(small*np.ones(Dim), 0, format='csc')
        factor = cholesky(Eig_Shift_DR_cut)
        DRL = factor.L()
        DRhalf = factor.apply_Pt(DRL.dot(W1))
        end = time.time()
        #print('time for DR root: '+ str((end - start)))


        start = time.time()
        WallCutHalf, it_lanczos = stochastic.stochastic_forcing_lanczos(factor=1.0,
                                                                        tolerance=self.tolerance,
                                                                        dim=Dim,
                                                                        mobility_mult=self.Wall_Mobility_Mult,
                                                                        L_mult=None,
                                                                        print_residual=print_res,
                                                                        z=W2)

        #print('USING UNTESTED CUDA LANCZOS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #WallCutHalf, it_lanczos = self.CUDA_Lanczos(W2,tol=self.tolerance)

        end = time.time()
        #print('time for M + MDRM root: '+ str(it_lanczos))
        print('time for M root: '+ str((end - start)) + ' num its: ' + str(it_lanczos))

        RHS_Xm = np.sqrt(2*self.kT / self.dt)*(DRhalf)
        RHS_X = np.sqrt(2*self.kT / self.dt)*(WallCutHalf)

        return RHS_Xm, RHS_X

    def IpMDR_Mult(self, X):
        '''
        Returns (I + M_RPY*DR)*X
        '''
        start = time.time()
        D_R = self.Delta_R.dot(X)
        end = time.time()
        # print 'time for DR mult : '+ str((end - start))

        start = time.time()
        M_Delta_R = self.Wall_Mobility_Mult(D_R)
        out = X + M_Delta_R
        end = time.time()
        # print 'time for M mult : '+ str((end - start))

        return out

    def IpMDR_PC(self, X, R_fact=None, isolated=[]):
        '''
        Returns (R_fact)^{-1}*X except for particles in the isolated set, which just return X
        '''
        start = time.time()

        RHS = self.R_MB.dot(X)
        for k in isolated:
            RHS[6*k:6*k+6] *= 0.0

        Y_F = R_fact(RHS)

        for k in isolated:
            Y_F[6*k:6*k+6] = X[6*k:6*k+6]

        end = time.time()
        # print 'time for PC op: '+ str((end - start))
        return Y_F

    def IpMDR_Swan_PC(self, X, R_fact=None):
        '''
        Returns (R_fact)^{-1}*X
        '''
        start = time.time()

        RHS = X
        Y_F = R_fact(RHS)

        end = time.time()
        # print 'time for PC op: '+ str((end - start))
        return Y_F

    def IpDRM_Mult(self, X):
        '''
        Returns (I + DR*M_RPY)*X
        '''

        M_X = self.Wall_Mobility_Mult(X)
        D_R_M = self.Delta_R.dot(M_X)
        out = X + D_R_M

        return out

    def IpDRM_PC(self, X, R_fact=None):
        '''
        Returns R_MB*(R_fact)^{-1}*X
        '''
        R_inv_X = R_fact(X)
        Y_F = self.R_MB.dot(R_inv_X)

        return Y_F

    def Lubrucation_RFD_solve(self, X, X0=None, print_residual=False, its_out=1000, PC_flag=True):
        '''
        Solve the transposed lubrication problem for bodies to get a random vector to be used in the Euler Maruyama mathod. 
        '''
        if(self.Delta_R is None):
            self.Set_R_Mats()

        num_particles = len(self.bodies)
        res_list = []

        RHS = X

        RHS_norm = np.linalg.norm(RHS)
        if RHS_norm > 0:
            RHS = RHS / RHS_norm

        if PC_flag:
            start = time.time()
            small = 6.0*np.pi*self.eta*self.a*self.tolerance
            Eig_Shift_R_Sup = self.R_Sup + \
                sp.diags(small*np.ones(6*num_particles), 0, format='csc')
            factor = cholesky(Eig_Shift_R_Sup)
            end = time.time()
            # print 'factor time : '+ str((end - start))
            PC_operator_partial = partial(self.IpDRM_PC, R_fact=factor)
            PC = spla.LinearOperator(
                (6*num_particles, 6*num_particles), matvec=PC_operator_partial, dtype='float64')
        else:
            PC = None

        A = spla.LinearOperator(
            (6*num_particles, 6*num_particles), matvec=self.IpDRM_Mult, dtype='float64')

        (U_gmres, info_precond) = pyamg.krylov.gmres(A, RHS, x0=X0, tol=self.tolerance, M=PC,
                                                     maxiter=min(its_out, A.shape[0]), restrt=min(100, A.shape[0]), residuals=res_list)
        if RHS_norm > 0:
            U_gmres = U_gmres * RHS_norm
        # print res_list
        return U_gmres, res_list

    def Lubrucation_solve(self, X, Xm, X0=None, print_residual=False, its_out=1000, PC_flag=True):
        '''
        Solve the lubrication problem for bodies using GMRES. Computes the solution U = [I + M_RPY*DR]^{-1} * (X + M*Xm).
        The PC ignores 'Isolated particles' which are those that are far from the wall and other particcles (making the PC less efective)
        '''
        if(self.Delta_R is None):
            self.Set_R_Mats()

        num_particles = len(self.bodies)
        res_list = []

        RHS = np.zeros((6*num_particles, 1))
        if Xm is not None:
            MXm = self.Wall_Mobility_Mult(Xm)
            MXm = MXm[:, np.newaxis]
            RHS += MXm
        if X is not None:
            RHS += X

        RHS_norm = np.linalg.norm(RHS)
        if RHS_norm > 0:
            RHS = RHS / RHS_norm

        if PC_flag:
            #############
            r_vecs_np = [b.location for b in self.bodies]
            r_vecs = list(self.put_r_vecs_in_periodic_box(
                r_vecs_np, self.periodic_length))
            r_tree = spatial.cKDTree(
                np.array(r_vecs), boxsize=self.periodic_length)

            isolated = []
            for j in range(num_particles):
                s1 = r_vecs[j]
                if s1[2] < self.cutoff*self.a:
                    continue
                idx = r_tree.query_ball_point(s1, r=self.cutoff*self.a)
                idx_trim = [i for i in idx if i > j]
                if not idx_trim:
                    isolated.append(j)
            ##############

            start = time.time()
            small = 6.0*np.pi*self.eta*self.a*self.tolerance
            Eig_Shift_R_Sup = self.R_Sup + \
                sp.diags(small*np.ones(6*num_particles), 0, format='csc')
            factor = cholesky(Eig_Shift_R_Sup)
            end = time.time()
            print('factor time : ' + str((end - start)))
            PC_operator_partial = partial(
                self.IpMDR_PC, R_fact=factor, isolated=isolated)

            ################## SWAN #####################
            #start = time.time()
            #gamma = 6.0*np.pi*self.eta*self.a
            #gamma_r = 8.0*np.pi*self.eta*(self.a**3)
            #onev = np.ones(6*num_particles)
            #diag_m = onev
            # for k in range(num_particles):
            #diag_m[6*k:6*k+3] *= (1.0/gamma)
            #diag_m[6*k+3:6*k+6] *= (1.0/gamma_r)
            #Shift_R_Sup = sp.diags(diag_m,0,format='csc')*self.Delta_R + sp.diags(onev,0,format='csc')
            #factor = cholesky(Shift_R_Sup)
            #end = time.time()
            #PC_operator_partial = partial(self.IpMDR_Swan_PC, R_fact=factor)
            ################## SWAN #####################

            PC = spla.LinearOperator(
                (6*num_particles, 6*num_particles), matvec=PC_operator_partial, dtype='float64')
        else:
            PC = None

        A = spla.LinearOperator(
            (6*num_particles, 6*num_particles), matvec=self.IpMDR_Mult, dtype='float64')

        res_list = []
        if X0 is not None:
            X0 = X0/RHS_norm

        (U_gmres, info_precond) = pyamg.krylov.fgmres(A, RHS, x0=X0, tol=self.tolerance, M=PC,
                                                     maxiter=min(its_out, A.shape[0]), restrt=min(100, A.shape[0]), residuals=res_list)
        #(U_gmres, info_precond) = pyamg.krylov.bicgstab(A,RHS, x0=X0, tol=self.tolerance, M=PC, maxiter=min(its_out,A.shape[0]), residuals=res_list)
        # (U_gmres, info_precond) = utils.gmres(A, RHS, x0=X0, tol=self.tolerance, M=PC, maxiter=min(its_out,A.shape[0]), restart = min(100,A.shape[0])) #, callback=counter)
        if RHS_norm > 0:
            U_gmres = U_gmres * RHS_norm
        # print res_list
        return U_gmres, res_list

    def Form_Lub_Mobility(self):
        '''
        Computes the Lubrication corrected mobility from self.bodies using ittereative linear algebra
        '''
        Dim = self.Delta_R.shape[1]
        I = np.eye(Dim)
        Mob = np.zeros((Dim, Dim))
        for k in range(Dim):
            vel, res = self.Lubrucation_solve(X=None, Xm=I[:, k])
            Mob[:, k] = vel
        return Mob


    def Form_Lub_Mobility_Dense(self, r_vecs):
        '''
        Computes the Lubrication corrected mobility from r_vecs using dense linear algebra
        '''
        M = self.Form_Mobility(r_vecs)
        Minv = np.linalg.pinv(M)
        DR = self.Compute_DeltaR(list(r_vecs))
        R = Minv + DR
        Mlub = np.linalg.pinv(R)
        return Mlub

    def Stochastic_Velocity_From_FT(self, FT):
        '''
        Performs one step of Euler Maruyama using lubrication corrections and ittereative linear algebra
        '''
        if self.kT > 0:
            Root_Xm, Root_X = self.Lub_Mobility_Root_RHS()
            Root_X = Root_X[:, np.newaxis]

            Drift_Xm, Drift_X = self.Lub_Mobility_RFD_RHS()
            Drift_Xm = Drift_Xm[:, np.newaxis]
            Drift_X = Drift_X[:, np.newaxis]

            FT = FT[:, np.newaxis]

            RHS_Xm = Root_Xm + self.kT*Drift_Xm + FT
            RHS_X = Root_X + self.kT*Drift_X
        else:
            RHS_Xm = FT[:, np.newaxis]
            RHS_X = None

        vel, res = self.Lubrucation_solve(X=RHS_X, Xm=RHS_Xm)
        return vel

    def Stochastic_Velocity_From_FT_centered_RFD(self, FT):
        '''
        Performs one step of Euler Maruyama using lubrication corrections and dense linear algebra for the RFD
        '''
        Root_Xm, Root_X = self.Lub_Mobility_Root_RHS()
        Root_X = Root_X[:, np.newaxis]

        center = self.Lub_Mobility_Centered_RFD()
        # print "centered RFD"
        # print  self.kT*center

        FT = FT[:, np.newaxis]

        RHS_Xm = Root_Xm + FT
        RHS_X = Root_X

        vel, res = self.Lubrucation_solve(X=RHS_X, Xm=RHS_Xm)
        vel += self.kT*center

        return vel

    def Stochastic_Velocity_From_FT_no_lub(self, FT):
        '''
        Performs one step of Euler Maruyama WITHOUT using lubrication corrections (just RPY)
        '''
        r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        W = np.random.randn(6*r_vecs.shape[0], 1)
        W2 = np.random.randn(6*r_vecs.shape[0], 1)

        Wrfd = np.reshape(W, (r_vecs.shape[0], 6))
        Wrfd = Wrfd[:, 0:3]

        Qp = r_vecs + (self.delta/2.0)*Wrfd
        Qm = r_vecs - (self.delta/2.0)*Wrfd

        MpW = self.Wall_Mobility_Mult(W, r_vecs_np=Qp)
        MmW = self.Wall_Mobility_Mult(W, r_vecs_np=Qm)
        
        self.solver.SetPositions(np.array(r_vecs).flatten())

        drift = (self.kT/self.delta)*(MpW-MmW)

        def MobV(v):
            Mv = self.Wall_Mobility_Mult(v)
            return Mv

        Mhalf, it_lanczos = stochastic.stochastic_forcing_lanczos(factor=np.sqrt(2*self.kT / self.dt),
                                                                  tolerance=self.tolerance,
                                                                  dim=6 *
                                                                  r_vecs.shape[0],
                                                                  mobility_mult=MobV,
                                                                  L_mult=None,
                                                                  print_residual=False)

        MF = MobV(FT)
        vel = MF + drift + Mhalf
        return vel

    def Update_Bodies(self, FT):
        '''
        Updates the positions and orientations of the bodies using stochastic velocity from Euler Maruyama (computed from `Stochastic_Velocity_From_FT')
        '''
        velocities = self.Stochastic_Velocity_From_FT(FT)
        for k, b in enumerate(self.bodies):
            b.location_new = b.location + velocities[6*k:6*k+3] * self.dt
            quaternion_dt = Quaternion.from_rotation(
                (velocities[6*k+3:6*k+6]) * self.dt)
            b.orientation_new = quaternion_dt * b.orientation

        reject_wall = 0
        reject_jump = 0

        reject_wall, reject_jump = self.Check_Update_With_Jump()
        self.num_rejections_wall += reject_wall
        self.num_rejections_jump += reject_jump

        if (reject_wall+reject_jump) == 0:
            L = self.periodic_length
            for b in self.bodies:
                b.location = b.location_new
                b.orientation = b.orientation_new

        self.Set_R_Mats()  # VERY IMPORTANT
        
        r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(r_vecs_np, self.periodic_length)
        self.solver.SetPositions(np.array(r_vecs).flatten())
        return reject_wall, reject_jump

    def Update_Bodies_Trap(self, FT_calc, Omega=None, Out_Torque=False, Cut_Torque=None):
        L = self.periodic_length
        # Save initial configuration
        for k, b in enumerate(self.bodies):
            np.copyto(b.location_old, b.location)
            b.orientation_old = copy.copy(b.orientation)

        # compute forces for predictor step
        r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        
        self.solver.SetPositions(np.array(r_vecs).flatten())
        start = time.time()
        FT = FT_calc(self.bodies, r_vecs)
        end = time.time()
        print('F calc time : ' + str((end - start)))
        FT = FT.flatten()
        FT = FT[:, np.newaxis]

        # If Omega Specified, add torques
        VO_guess = None
        if Omega is not None:
            FTrs = np.reshape(FT, (len(self.bodies), 6))
            F = FTrs[:, 0:3]
            start = time.time()
            T_omega, VO_guess = self.Torque_from_Omega(Omega, F)
            if Cut_Torque is not None:
                Tn = np.linalg.norm(T_omega, axis=1)
                NewNorm = np.minimum(Tn, Cut_Torque)/Tn
                T_omega = NewNorm[:, None]*T_omega
            end = time.time()
            print('Omega time : ' + str((end - start)))
            FTrs[:, 3::] += T_omega
            FT = np.reshape(FTrs, (6*len(self.bodies), 1))

        # compute relevant matrix root for pred. and corr. steps
        start = time.time()
        Root_Xm, Root_X = self.Lub_Mobility_Root_RHS()
        end = time.time()
        print('root time : ' + str((end - start)))
        X = Root_X[:, np.newaxis]
        MXm = self.Wall_Mobility_Mult(Root_Xm)
        MXm = MXm[:, np.newaxis]
        Mhalf = X + MXm

        # compute predictor velocities
        start = time.time()
        vel_p, res_p = self.Lubrucation_solve(X=Mhalf, Xm=FT, X0=VO_guess)
        end = time.time()
        print('solve 1 : ' + str((end - start)) + ' num its = ' + str(len(res_p)))
        
        # compute rfd for M
        W = np.random.randn(6*r_vecs.shape[0], 1)
        Wrfd = np.reshape(W, (r_vecs.shape[0], 6))
        Wrfd = Wrfd[:, 0:3]

        Qp = r_vecs + (self.delta/2.0)*Wrfd
        Qm = r_vecs - (self.delta/2.0)*Wrfd

        MpW = self.Wall_Mobility_Mult(W, r_vecs_np=Qp)
        MmW = self.Wall_Mobility_Mult(W, r_vecs_np=Qm)

        D_M = 2.0*(self.kT/self.delta)*(MpW-MmW)
        D_M = D_M[:, np.newaxis]

        # maybe important (gets reset in Set_R_Mats)
        self.solver.SetPositions(np.array(r_vecs).flatten())
        
        # update to corrector positions
        for k, b in enumerate(self.bodies):
            b.location = b.location_old + vel_p[6*k:6*k+3] * self.dt
            quaternion_dt = Quaternion.from_rotation(
                (vel_p[6*k+3:6*k+6]) * self.dt)
            b.orientation = quaternion_dt * b.orientation_old

        r_vecs_np = [b.location for b in self.bodies]
        r_vecs_c = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        self.Set_R_Mats(r_vecs_np=r_vecs_c)  # VERY IMPORTANT, this also sets solver positions


        # compute RHSs for the corr. step
        RHS_X_C = D_M + Mhalf

        start = time.time()
        FT_C = FT_calc(self.bodies, r_vecs_c)
        end = time.time()
        print('F calc time : ' + str((end - start)))
        FT_C = FT_C.flatten()
        FT_C = FT_C[:, np.newaxis]

        # If Omega Specified, add torques
        VO_guessc = vel_p
        second_order = False
        if Omega is not None:
            FTrsc = np.reshape(FT_C, (len(self.bodies), 6))
            Fc = FTrsc[:, 0:3]
            if second_order:
                # you could use the previous Torque_from_Omega solve as an initial guess here and it should work very well
                Tc, VO_guessc = self.Torque_from_Omega(Omega, Fc)
                if Cut_Torque is not None:
                    Tn = np.linalg.norm(Tc, axis=1)
                    NewNorm = np.minimum(Tn, Cut_Torque)/Tn
                    Tc = NewNorm[:, None]*Tc
            else:
                Tc = T_omega
            FTrsc[:, 3::] += Tc
            FT_C = np.reshape(FTrsc, (6*len(self.bodies), 1))

        RHS_Xm_C = FT_C
        # compute for corrected velocity and update positions

        start = time.time()
        vel_c, res_c = self.Lubrucation_solve(
            X=RHS_X_C, Xm=RHS_Xm_C, X0=VO_guessc)
        end = time.time()
        print('solve 2 : ' + str((end - start)) + ' num its = ' + str(len(res_c)))

        vel_trap = 0.5 * (vel_c + vel_p)

        for k, b in enumerate(self.bodies):
            b.location_new = b.location_old + vel_trap[6*k:6*k+3] * self.dt
            quaternion_dt = Quaternion.from_rotation(
                (vel_trap[6*k+3:6*k+6]) * self.dt)
            b.orientation_new = quaternion_dt * b.orientation_old

        reject_wall = 0
        reject_jump = 0
        reject_wall, reject_jump = self.Check_Update_With_Jump_Trap()
        self.num_rejections_wall += reject_wall
        self.num_rejections_jump += reject_jump

        if (reject_wall+reject_jump) == 0:
            for b in self.bodies:
                np.copyto(b.location, b.location_new)
                b.orientation = copy.copy(b.orientation_new)
        else:
            for b in self.bodies:
                np.copyto(b.location, b.location_old)
                b.orientation = copy.copy(b.orientation_old)

        self.Set_R_Mats()  # VERY IMPORTANT
        if Out_Torque:
            return reject_wall, reject_jump, T_omega.flatten()
        else:
            return reject_wall, reject_jump

################################################################################
    def Update_Bodies_Trap_Det(self, FT_calc, Omega=None, Out_Torque=False, Cut_Torque=None):
        L = self.periodic_length
        # Save initial configuration
        for k, b in enumerate(self.bodies):
            np.copyto(b.location_old, b.location)
            b.orientation_old = copy.copy(b.orientation)

        # compute forces for predictor step
        r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        start = time.time()
        FT = FT_calc(self.bodies, r_vecs)
        end = time.time()
        print('F calc time : ' + str((end - start)))
        FT = FT.flatten()
        FT = FT[:, np.newaxis]

        # If Omega Specified, add torques
        VO_guess = None
        if Omega is not None:
            FTrs = np.reshape(FT, (len(self.bodies), 6))
            F = FTrs[:, 0:3]
            start = time.time()
            T_omega, VO_guess = self.Torque_from_Omega(Omega, F, IG=VO_guess)
            if Cut_Torque is not None:
                Tn = np.linalg.norm(T_omega, axis=1)
                NewNorm = np.minimum(Tn, Cut_Torque)/Tn
                T_omega = NewNorm[:, None]*T_omega
            end = time.time()
            print('Omega time : ' + str((end - start)))
            FTrs[:, 3::] += T_omega
            FT = np.reshape(FTrs, (6*len(self.bodies), 1))

        # compute predictor velocities and update positions
        start = time.time()
        vel_p, res_p = self.Lubrucation_solve(X=None, Xm=FT, X0=VO_guess)
        end = time.time()
        print('solve 1 : ' + str((end - start)))

        for k, b in enumerate(self.bodies):
            b.location = b.location_old + vel_p[6*k:6*k+3] * self.dt
            quaternion_dt = Quaternion.from_rotation(
                (vel_p[6*k+3:6*k+6]) * self.dt)
            b.orientation = quaternion_dt * b.orientation_old

        r_vecs_np = [b.location for b in self.bodies]
        r_vecs_c = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        self.Set_R_Mats(r_vecs_np=r_vecs_c)  # VERY IMPORTANT

        start = time.time()
        FT_C = FT_calc(self.bodies, r_vecs_c)
        end = time.time()
        print('F calc time : ' + str((end - start)))
        FT_C = FT_C.flatten()
        FT_C = FT_C[:, np.newaxis]

        # If Omega Specified, add torques
        VO_guessc = vel_p
        second_order = True
        if Omega is not None:
            FTrsc = np.reshape(FT_C, (len(self.bodies), 6))
            Fc = FTrsc[:, 0:3]
            if second_order:
                # you could use the previous Torque_from_Omega solve as an initial guess here and it should work very well
                Tc, VO_guessc = self.Torque_from_Omega(Omega, Fc, IG=VO_guess)
                if Cut_Torque is not None:
                    Tn = np.linalg.norm(Tc, axis=1)
                    NewNorm = np.minimum(Tn, Cut_Torque)/Tn
                    Tc = NewNorm[:, None]*Tc
            else:
                Tc = T_omega
            FTrsc[:, 3::] += Tc
            FT_C = np.reshape(FTrsc, (6*len(self.bodies), 1))

        RHS_Xm_C = FT_C
        # compute for corrected velocity and update positions

        start = time.time()
        vel_c, res_c = self.Lubrucation_solve(
            X=None, Xm=RHS_Xm_C, X0=VO_guessc)
        end = time.time()
        print('solve 2 : ' + str((end - start)))

        vel_trap = 0.5 * (vel_c + vel_p)

        for k, b in enumerate(self.bodies):
            b.location_new = b.location_old + vel_trap[6*k:6*k+3] * self.dt
            quaternion_dt = Quaternion.from_rotation(
                (vel_trap[6*k+3:6*k+6]) * self.dt)
            b.orientation_new = quaternion_dt * b.orientation_old

        reject_wall = 0
        reject_jump = 0
        reject_wall, reject_jump = self.Check_Update_With_Jump_Trap()
        self.num_rejections_wall += reject_wall
        self.num_rejections_jump += reject_jump

        if (reject_wall+reject_jump) == 0:
            for b in self.bodies:
                np.copyto(b.location, b.location_new)
                b.orientation = copy.copy(b.orientation_new)
        else:
            for b in self.bodies:
                np.copyto(b.location, b.location_old)
                b.orientation = copy.copy(b.orientation_old)

        self.Set_R_Mats()  # VERY IMPORTANT
        if Out_Torque:
            return reject_wall, reject_jump, T_omega.flatten()
        else:
            return reject_wall, reject_jump

##########################################################################################
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def fast_update_rvec(r_vecs,F,D,kT,dt,Nb):
        r_out = 0*r_vecs
        for k in prange(Nb):
            for part in range(3):
                r_out[k,part] = r_vecs[k,part] + (dt*D/kT)*F[k,part] + np.sqrt(2.0*D*dt)*np.random.randn()
        return r_out


    def Update_Bodies_No_Hydro_2D(self, FT_calc, D):
        '''
        Calculate Brownian Dynamics without hydro in 2D given a diffusion coeficient D
        '''
        for k, b in enumerate(self.bodies):
            np.copyto(b.location_old, b.location)


        L = self.periodic_length
        r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        FT = FT_calc(self.bodies, r_vecs)
        FT = FT.flatten()
        FT = FT[:, np.newaxis]
        FTrs = np.reshape(FT, (len(self.bodies), 6))
        F = FTrs[:, 0:3]

        #r_new = self.fast_update_rvec(r_vecs,F,D,self.kT,self.dt,len(self.bodies))

        for k, b in enumerate(self.bodies):
            #b.location_new = r_new[k]
            #b.location_new[2] = 0.0
            for part in range(3):
                b.location_new[part] = b.location_old[part] + (self.dt*D/self.kT)*F[k,part] + np.sqrt(2.0*D*self.dt)*np.random.randn()


        reject_wall = 0
        reject_jump = 0
        reject_wall, reject_jump = self.Check_Update_With_Jump_Trap()
        self.num_rejections_wall += reject_wall
        self.num_rejections_jump += reject_jump

        if (reject_wall+reject_jump) == 0:
            for b in self.bodies:
                np.copyto(b.location, b.location_new)
                b.orientation = copy.copy(b.orientation_new)
        else:
            for b in self.bodies:
                np.copyto(b.location, b.location_old)
                b.orientation = copy.copy(b.orientation_old)

        return reject_wall, reject_jump


    def Torque_from_Omega(self, Om, F, r_vecs_np=None):
        '''
        Given an angular velocity Om, a torque is computed based on the forces (F) so that the angular velocity of the particles 
        is approximatly constrained to be = Om*y_hat.
        '''
        if r_vecs_np is None:
            r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        
        num_particles = len(r_vecs)
        
        if r_vecs_np is not None:
            self.solver.SetPositions(np.array(r_vecs).flatten())

        def Mrr(torque):
            U,W = self.solver.Mdot(forces=0*torque, torques=torque)
            return W

        def Mtr(torque):
            U,W = self.solver.Mdot(forces=0*torque, torques=torque)
            return U

        def Mtt(force):
            U,W = self.solver.Mdot(forces=force, torques=0*force)
            return U

        def V_T_Mat_Mult(VT):
            VT = np.reshape(VT, (len(self.bodies), 6))

            V = np.copy(VT)
            V[:, 3::] *= 0
            V = -1.0*V.flatten()
            T = VT[:, 3::].flatten()

            out = np.reshape(self.IpMDR_Mult(V), (len(self.bodies), 6))
            out[:, 0:3] += np.reshape(Mtr(T), (len(self.bodies), 3))
            out[:, 3::] += np.reshape(Mrr(T), (len(self.bodies), 3))
            return out

        Om0 = np.zeros(6*len(self.bodies))
        for i in range(len(self.bodies)):
            Om0[6*i+4] = Om
        F0 = np.concatenate((F, np.zeros(F.shape)), axis=1)
        F0 = F0.flatten()
        RHS = self.IpMDR_Mult(
            Om0) - self.Wall_Mobility_Mult(F0, r_vecs_np=r_vecs)

        A = spla.LinearOperator(
            (6*len(self.bodies), 6*len(self.bodies)), matvec=V_T_Mat_Mult, dtype='float64')

        ############### PC 1 ############
        # Get the tt and rt blocks of DR
        ttInd = np.zeros((len(self.bodies), 6))
        ttInd[:, 0:3] = 1
        ttInd = np.nonzero(ttInd.flatten())[0]

        rrInd = np.zeros((len(self.bodies), 6))
        rrInd[:, 3::] = 1
        rrInd = np.nonzero(rrInd.flatten())[0]

        DRtt = self.Delta_R[:, ttInd][ttInd, :]
        DRrt = self.Delta_R[:, ttInd][rrInd, :]

        c1 = 6.0 * np.pi * self.eta * self.a
        c2 = 8.0 * np.pi * self.eta * self.a**3

        Vmat = sp.diags(c1*np.ones(3*len(self.bodies)), 0, format='csc') + DRtt
        Vfact = cholesky(Vmat)

        def PC_mult(ab):
            AB = np.reshape(ab, (len(self.bodies), 6))
            a = -c1*AB[:, 0:3].flatten()
            v = Vfact(a)
            t = c2*AB[:, 3::].flatten() + DRrt.dot(v)
            V = np.reshape(v, (len(self.bodies), 3))
            T = np.reshape(t, (len(self.bodies), 3))
            return np.concatenate((V, T), axis=1).flatten()

        PC = spla.LinearOperator(
            (6*len(self.bodies), 6*len(self.bodies)), matvec=PC_mult, dtype='float64')
        ############### PC 1 ############

        # Scale RHS to norm 1
        RHS_norm = np.linalg.norm(RHS)
        if RHS_norm > 0:
            RHS = RHS / RHS_norm

        # use 8*pi*eta*a^3*Omega as initial guess for torque
        Om_g = np.zeros((len(self.bodies), 3))
        Om_g[:, 1] += Om
        T_g = c2*Om_g
        V_g = 0*T_g
        X0_vt = np.concatenate((V_g, T_g), axis=1).flatten()
        X0_vt *= (1.0/RHS_norm)

        # Solve linear system
        res_list = []
        (VT_gmres, info_precond) = pyamg.krylov.gmres(A, RHS, M=PC, x0=X0_vt,
                                                      tol=self.tolerance, maxiter=100, restrt=min(100, A.shape[0]), residuals=res_list)

        print(res_list)
        # Scale solution with RHS norm
        if RHS_norm > 0:
            VT_gmres = VT_gmres * RHS_norm

        VT = np.reshape(VT_gmres, (len(self.bodies), 6))
        Torque = VT[:, 3::]

        VO_guess = np.concatenate((VT[:, 0:3], Om_g), axis=1)
        VO_guess = VO_guess.flatten()

        return Torque, VO_guess

    def Update_Bodies_no_lub(self, FT_calc):
        '''
        Euler Maruyama method with no lubrication corrections (just RPY)
        '''
        r_vecs_np = [b.location for b in self.bodies]
        r_vecs = self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length)
        FT = FT_calc(self.bodies, r_vecs)
        FT = FT.flatten()
        FT = FT[:, np.newaxis]

        velocities = self.Stochastic_Velocity_From_FT_no_lub(FT)
        for k, b in enumerate(self.bodies):
            b.location_new = b.location + velocities[6*k:6*k+3] * self.dt
            quaternion_dt = Quaternion.from_rotation(
                (velocities[6*k+3:6*k+6]) * self.dt)
            b.orientation_new = quaternion_dt * b.orientation

        reject_wall = 0
        reject_jump = 0
        reject_wall, reject_jump = self.Check_Update_With_Jump()
        self.num_rejections_wall += reject_wall
        self.num_rejections_jump += reject_jump

        if (reject_wall+reject_jump) == 0:
            L = self.periodic_length
            for b in self.bodies:
                b.location = b.location_new
                b.orientation = b.orientation_new

        return reject_wall, reject_jump

    def Check_Update_With_Jump_Trap(self):
        '''
        Make sure particle dont move too much durung a step. if they do, mark that step for rejection.
        '''
        r_vecs = [b.location_new for b in self.bodies]
        r_vecs_old = [b.location_old for b in self.bodies]
        num_particles = len(r_vecs)
        reject_wall = 0
        reject_jump = 0
        for j in range(num_particles):
            s1 = r_vecs[j]
            if s1[2] < 0:
                print("rejected time step, wall")
                reject_wall = 1
                return reject_wall, reject_jump

            s1_old = r_vecs_old[j]
            r = s1-s1_old
            r = self.project_to_periodic_image(r, self.periodic_length)
            disp = np.linalg.norm(r)
            if disp > 2*self.a:
                print("rejected time step large jump: ", disp, s1, s1_old)
                reject_jump = 1
                return reject_wall, reject_jump

        return reject_wall, reject_jump

    def Check_Update_With_Jump(self):
        '''
        Make sure particle dont move too much durung a step. if they do, mark that step for rejection.
        '''
        r_vecs = [b.location_new for b in self.bodies]
        r_vecs_old = [b.location for b in self.bodies]
        num_particles = len(r_vecs)
        reject_wall = 0
        reject_jump = 0
        for j in range(num_particles):
            s1 = r_vecs[j]
            if s1[2] < 0:
                print("rejected time step, wall")
                reject_wall = 1
                return reject_wall, reject_jump

            s1_old = r_vecs_old[j]
            r = s1-s1_old
            r = self.project_to_periodic_image(r, self.periodic_length)
            disp = np.linalg.norm(r)
            if disp > 2*self.a:
                print("rejected time step large jump: ", disp, s1, s1_old)
                reject_jump = 1
                return reject_wall, reject_jump

        return reject_wall, reject_jump
