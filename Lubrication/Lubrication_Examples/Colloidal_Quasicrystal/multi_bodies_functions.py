'''
In this module the user can define functions that modified the
code multi_blobs.py. For example, functions to define the
blobs-blobs interactions, the forces and torques on the rigid
bodies or the slip on the blobs.
'''
import numpy as np
import sys
import imp
import os.path
import copy
from functools import partial
import scipy.spatial as spatial

import scipy.sparse.linalg as spla

import general_application_utils
from quaternion_integrator.quaternion import Quaternion

import numba as nmba
from numba import njit, jit, prange
    

def project_to_periodic_image(r, L):
  '''
  Project a vector r to the minimal image representation
  centered around (0,0,0) and of size L=(Lx, Ly, Lz). If 
  any dimension of L is equal or smaller than zero the 
  box is assumed to be infinite in that direction.
  '''
  if L is not None:
    for i in range(3):
      if(L[i] > 0):
        r[i] = r[i] - int(r[i] / L[i] + 0.5 * (int(r[i]>0) - int(r[i]<0))) * L[i]
  return r

def default_zero_r_vectors(r_vectors, *args, **kwargs):
  return np.zeros((r_vectors.size / 3, 3))


def default_zero_blobs(body, *args, **kwargs):
  ''' 
  Return a zero array of shape (body.Nblobs, 3)
  '''
  return np.zeros((body.Nblobs, 3))


def default_zero_bodies(bodies, *args, **kwargs):
  ''' 
  Return a zero array of shape (2*len(bodies), 3)
  '''
  return np.zeros((2*len(bodies), 3))
  

def set_slip_by_ID(body, slip):
  '''
  This function assign a slip function to each body.
  If the body has an associated slip file the function
  "active_body_slip" is assigned (see function below).
  Otherwise the slip is set to zero.

  This function can be override to assign other slip
  functions based on the body ID, (ID of a structure
  is the name of the clones file (without .clones)).
  See the example in
  "examples/pair_active_rods/".
  '''
  if slip is not None:
    active_body_slip_partial = partial(active_body_slip, slip = slip)
    body.function_slip = active_body_slip_partial
  else:
    body.function_slip = default_zero_blobs
  return


def active_body_slip(body, slip):
  '''
  This function set the slip read from the *.slip file to the
  blobs. The slip on the file is given in the body reference 
  configuration (quaternion = (1,0,0,0)) therefore this
  function rotates the slip to the current body orientation.
  
  This function can be used, for example, to model active rods
  that propel along their axes. 
  '''
  # Get rotation matrix
  rotation_matrix = body.orientation.rotation_matrix()

  # Rotate  slip on each blob
  slip_rotated = np.empty((body.Nblobs, 3))
  for i in range(body.Nblobs):
    slip_rotated[i] = np.dot(rotation_matrix, slip[i])
  return slip_rotated


def bodies_external_force(bodies, r_vectors, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2*len(bodies), 3)
  
  In this is example we just set it to zero.
  '''
  
  ext_FT = np.zeros((len(bodies), 3))
  
  return ext_FT



def bodies_external_torque(bodies, r_vectors, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2*len(bodies), 3)
  
  In this is example we just set it to zero.
  '''
  
  ext_FT = np.zeros((len(bodies), 3))
  return ext_FT
  

def blob_external_force(r_vectors, *args, **kwargs):
  '''
  This function compute the external force acting on a
  single blob. It returns an array with shape (3).
  
  In this example we add gravity and a repulsion with the wall;
  the interaction with the wall is derived from the potential

  U(z) = U0 + U0 * (a-z)/b   if z<a
  U(z) = U0 * exp(-(z-a)/b)  iz z>=a

  with 
  e = repulsion_strength_wall
  a = blob_radius
  h = distance to the wall
  b = debye_length_wall
  '''
  f = np.zeros(3)

  # Get parameters from arguments
  blob_mass = kwargs.get('blob_mass')
  blob_radius = kwargs.get('blob_radius')
  g = kwargs.get('g')
  repulsion_strength_firm = kwargs.get('repulsion_strength_firm') 
  debye_length_firm = kwargs.get('debye_length_firm')
  firm_delta = kwargs.get('firm_delta')
  z_max = kwargs.get('z_max')
  # Add gravity
  f += -g * blob_mass * np.array([0., 0., 1.0])
  
  # Add wall interaction
  h = r_vectors[2]
  if h > blob_radius*(1.0-firm_delta):
      f[2] += (repulsion_strength_firm / debye_length_firm) * \
          np.exp(-(h-blob_radius*(1.0-firm_delta))/debye_length_firm)
  else:
      f[2] += (repulsion_strength_firm / debye_length_firm)
      
  if z_max is not None:
      H_chan = (z_max-blob_radius*(1.0-firm_delta))
      if h < H_chan:
          f[2] -= (repulsion_strength_firm / debye_length_firm) * \
              np.exp(-(H_chan-h)/debye_length_firm)
      else:
          f[2] -= (repulsion_strength_firm / debye_length_firm)

  # Add wall interaction
  eps_soft = kwargs.get('repulsion_strength_wall')
  b_soft = kwargs.get('debye_length_wall')
  h = r_vectors[2]
  if h > (blob_radius):
      f[2] += (eps_soft / b_soft) * np.exp(-(h-blob_radius)/b_soft)
  else:
      f[2] += (eps_soft / b_soft)
      
  if z_max is not None:
      H_chan = (z_max-blob_radius)
      if h < H_chan:
          f[2] -= (eps_soft / b_soft) * np.exp(-(H_chan-h)/b_soft)
      else:
          f[2] -= (eps_soft / b_soft)

  return f
      
      
  #################
  # drop potential
  #################
  #Rc = kwargs.get('Srad')
  #drop_eps = kwargs.get('drop_eps')
  #Hc = kwargs.get('h_shift')
  
  #r_s = np.copy(r_vectors)
  #r_s[2] += Hc
  
  #rad = np.linalg.norm(r_s)

  #f += (drop_eps/rad)*((Rc-rad)**1)*r_s
  #################    
      
  return f


def calc_one_blob_forces(r_vectors, *args, **kwargs):
  '''
  Compute one-blob forces. It returns an array with shape (Nblobs, 3).
  '''
  Nblobs = int(r_vectors.size / 3)
  force_blobs = np.zeros((Nblobs, 3))
  r_vectors = np.reshape(r_vectors, (Nblobs, 3))
  
  # Loop over blobs
  for blob in range(Nblobs):
    force_blobs[blob] += blob_external_force(r_vectors[blob], *args, **kwargs)   

  return force_blobs




@njit(parallel=True, fastmath=True)
def blob_blob_force_numba(r_vectors, r_vectors_images, moments, E_mom, E_mom_image, zax, L, a, repulsion_strength_firm, debye_length_firm, firm_delta, repulsion_strength, debye_length, C, B_torque, C_z):
  '''
  This function compute the force between two blobs
  with vector between blob centers r.

  In this example the force is derived from the potential

  U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
  U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a

  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
  a = blob_radius
  '''

  N = r_vectors.size // 3
  N_im = r_vectors_images.size // 3
  r_vectors = r_vectors.reshape((N, 3))
  r_vectors_images = r_vectors_images.reshape((N_im, 3))
  force = np.zeros((N, 3))
  torque = np.zeros((N, 3))

  for i in prange(N):
    #torque[i,:] += 100.0*(1.0/3.0)*C*(2.0*a)*np.cross(zax[i,:],np.array([0.0,0.0,1.0]))
    #torque[i,:] += B_torque[i,:]
    for j in range(N_im):
      if i == j:
        continue
      dr = np.zeros(3)
      for k in range(3):
        dr[k] = r_vectors_images[j,k] - r_vectors[i,k]
        if L[k] > 0:
          dr[k] -= int(dr[k] / L[k] + 0.5 * (int(dr[k]>0) - int(dr[k]<0))) * L[k]

      # Compute force
      r_norm = np.sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2])
      r_hat = dr/r_norm
      dist = (1.0/r_norm)

      m_z_i_norm = np.linalg.norm(E_mom[i,:])
      m_z_j_norm = np.linalg.norm(E_mom_image[j,:])

      m_z_i = E_mom[i,:]/m_z_i_norm
      m_z_j = E_mom_image[j,:]/m_z_j_norm


      mi_z_d_rhat = np.dot(m_z_i,r_hat)
      mj_z_d_rhat = np.dot(m_z_j,r_hat)
      mi_z_d_mj_z = np.dot(m_z_i,m_z_j)



      F_mag_z = C_z*(m_z_i_norm*m_z_j_norm)*dist*dist*dist*dist
      T_mag_z = (1.0/3.0)*(m_z_i_norm*m_z_j_norm)*C_z*dist*dist*dist
      
      mi_z_X_mj_z = np.cross(m_z_i,m_z_j)
      mi_z_X_r = np.cross(m_z_i,r_hat)
      mj_z_X_r = np.cross(m_z_j,r_hat)

      # the torque is computed as m_j X B_i + r X F_ij
      # which equals m_i X B_j
      # Interactions between uniformly magnetized spheres, William A. Booth, 2016
      # Quantifying hydrodynamic collective states of magnetic colloidal spinners and rollers, I, Aranson

      force_mag_z = F_mag_z * ( (mi_z_d_rhat) * m_z_j + (mj_z_d_rhat) * m_z_i - (5.0 * (mj_z_d_rhat*mi_z_d_rhat) - mi_z_d_mj_z) * r_hat )
      force[i,:] -= force_mag_z
      ######### You used to not have this and the sign was +
      torque[i,:] += T_mag_z * (3 * mj_z_d_rhat * mi_z_X_r -  mi_z_X_mj_z) # used to be +
     
      if j < N:
        m_i_norm = np.linalg.norm(moments[i,:])
        m_j_norm = np.linalg.norm(moments[j,:])

        m_i = moments[i,:]/m_i_norm
        m_j = moments[j,:]/m_j_norm

        mi_d_rhat = np.dot(m_i,r_hat)
        mj_d_rhat = np.dot(m_j,r_hat)
        mi_d_mj = np.dot(m_i,m_j)

        mi_X_mj = np.cross(m_i,m_j)
        mi_X_r = np.cross(m_i,r_hat)
        mj_X_r = np.cross(m_j,r_hat)

        F_mag = C*(m_i_norm*m_j_norm)*dist*dist*dist*dist
        T_mag = (1.0/3.0)*C*(m_i_norm*m_j_norm)*dist*dist*dist
      
        force_mag = F_mag * ( (mi_d_rhat) * m_j + (mj_d_rhat) * m_i - (5.0 * (mj_d_rhat*mi_d_rhat) - mi_d_mj) * r_hat )
        force[i,:] -= force_mag
        torque[i,:] += T_mag * (3 * mj_d_rhat * mi_X_r -  mi_X_mj) # used to be +

        for k in range(3): 
          #print(force_torque)
          #################
          offset = 2.0*a-2.0*firm_delta*a
          if r_norm > (offset):
            force[i,k] += -((repulsion_strength_firm / debye_length_firm) * np.exp(-(r_norm-(offset)) / debye_length_firm) / np.maximum(r_norm, 1.0e-12)) * dr[k]
          else:
            force[i,k] += -((repulsion_strength_firm / debye_length_firm) / np.maximum(r_norm, 1.0e-12)) * dr[k]

          offset = 2.0*a
          if r_norm > (offset):
            force[i,k] += -((repulsion_strength / debye_length) * np.exp(-(r_norm-(offset)) / debye_length) / np.maximum(r_norm, 1.0e-12)) * dr[k]
          else:
            force[i,k] += -((repulsion_strength / debye_length) / np.maximum(r_norm, 1.0e-12)) * dr[k]


  # print('mean forces and torques')
  # print(np.mean(force))
  # print(np.mean(torque))

  for i in prange(N):
    torque[i,:] += B_torque[i,:]
  return force, torque



def body_body_force_torque(r, quaternion_i, quaternion_j, *args, **kwargs):
  '''
  This function compute the force between two bodies
  with vector between locations r.
  In this example the torque is zero and the force 
  is derived from a Yukawa potential
  
  U = eps * exp(-r_norm / b) / r_norm
  
  with
  eps = potential strength
  r_norm = distance between bodies' location
  b = Debye length
  '''
  force_torque = np.zeros((2, 3))
  
  return force_torque


def calc_body_body_forces_torques_python(bodies, r_vectors, *args, **kwargs):
  '''
  This function computes the body-body forces and torques and returns
  an array with shape (2*Nblobs, 3).
  '''
  Nbodies = len(bodies)
  force_torque_bodies = np.zeros((2*len(bodies), 3))
  
  # Double loop over bodies to compute forces
  for i in range(Nbodies-1):
    for j in range(i+1, Nbodies):
      # Compute vector from j to u
      r = bodies[j].location - bodies[i].location
      force_torque = body_body_force_torque(r, bodies[i].orientation, bodies[j].orientation, *args, **kwargs)
      # Add forces
      force_torque_bodies[2*i] += force_torque[0]
      force_torque_bodies[2*j] -= force_torque[0]
      # Add torques
      force_torque_bodies[2*i+1] += force_torque[1]
      force_torque_bodies[2*j+1] -= force_torque[1]

  return force_torque_bodies


def Magnetic_Mobility(Moments,r_vectors,L,a,chi,RB_0):
  ''' 
  Returns the product of the magnetic mobility at the blob level to the moments the blobs. 
  Mobility for particles in an unbounded domain
  This function uses numba.
  taken from 
  https://pubs.rsc.org/en/content/articlelanding/2019/sm/c9sm00890j#cit66
  '''
  # Variables
  N = int(r_vectors.size // 3)
  r_vectors = r_vectors.reshape(N, 3)
  Moments = Moments.reshape(N, 3)
  B_net = np.zeros((N, 3))
  ChiOverThree = chi / 3.0
  a3 = a*a*a
  inva = 1.0/a
  inva3 = 1.0/a3

  norm_fact_f = 1.0 / RB_0

  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0 
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
	  
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]              
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1         

            if i == j_image:
              Mxx = 1.0
              Mxy = 0
              Mxz = 0
              Myy = Mxx
              Myz = 0
              Mzz = Mxx           
            else:
              # Normalize distance with hydrodynamic radius
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              r3 = r*r2
              
              # TODO: We should not divide by zero 
              invr = 1.0 / r
              invr2 = invr * invr
              invr3 = invr * invr2

              if r > 2:
                c1 = 1.0
                c2 = -3.0 * invr2
                Mxx = (c1 + c2*rx*rx) * invr3*a3*ChiOverThree
                Mxy = (     c2*rx*ry) * invr3*a3*ChiOverThree
                Mxz = (     c2*rx*rz) * invr3*a3*ChiOverThree
                Myy = (c1 + c2*ry*ry) * invr3*a3*ChiOverThree
                Myz = (     c2*ry*rz) * invr3*a3*ChiOverThree
                Mzz = (c1 + c2*rz*rz) * invr3*a3*ChiOverThree 
              else:
                c1 = ChiOverThree*(1.0 - 0.5625*r*inva + 0.03125*r3*inva3) # 1/32 = 0.03125 #9/16 = 0.5625
                c2 = ChiOverThree*(0.09375*r3*inva3 - 0.5625*r*inva)*invr2      # 3/32 = 0.09375
                Mxx = c1 + c2 * rx*rx 
                Mxy =      c2 * rx*ry 
                Mxz =      c2 * rx*rz 
                Myy = c1 + c2 * ry*ry 
                Myz =      c2 * ry*rz 
                Mzz = c1 + c2 * rz*rz 
                
            Myx = Mxy
            Mzx = Mxz
            Mzy = Myz
	  
            # 2. Compute product M_ij * F_j           
            B_net[i,0] += (Mxx * Moments[j,0] + Mxy * Moments[j,1] + Mxz * Moments[j,2]) * norm_fact_f
            B_net[i,1] += (Myx * Moments[j,0] + Myy * Moments[j,1] + Myz * Moments[j,2]) * norm_fact_f
            B_net[i,2] += (Mzx * Moments[j,0] + Mzy * Moments[j,1] + Mzz * Moments[j,2]) * norm_fact_f

  return B_net.flatten()

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

def force_torque_calculator_sort_by_bodies(bodies, r_vectors, *args, **kwargs):
  '''
  Return the forces and torque in each body with
  format [f_1, t_1, f_2, t_2, ...] and shape (2*Nbodies, 3),
  where f_i and t_i are the force and torque on the body i.
  '''
  # Create auxiliar variables
  Nblobs = int(r_vectors.size / 3)
  force_torque_bodies = np.zeros((2*len(bodies), 3))
  force_blobs = np.zeros((Nblobs, 3))
  blob_mass = 1.0
  blob_radius = bodies[0].blob_radius

  # Compute one-blob forces (same function for all blobs)
  force_blobs += calc_one_blob_forces(r_vectors, blob_radius = blob_radius, blob_mass = blob_mass, *args, **kwargs)

  # Compute blob-blob forces (same function for all pair of blobs)
  #force_blobs += calc_blob_blob_forces(r_vectors, blob_radius = blob_radius, *args, **kwargs)
  ########################################################
  L = kwargs.get('periodic_length')
  time_s = kwargs.get('time_s')

  repulsion_strength_firm = kwargs.get('repulsion_strength_firm')
  debye_length_firm = kwargs.get('debye_length_firm')
  firm_delta = kwargs.get('firm_delta')

  repulsion_strength = kwargs.get('repulsion_strength')
  debye_length = kwargs.get('debye_length')


  mu_dipole = kwargs.get('mu_dipole')
  B_0 = kwargs.get('B_0')
  RB_0 = 48.22
  mu_induced = RB_0*B_0 
  # https://www.wolframalpha.com/input?i=convert+4*pi*%282.25+um%29%5E3+*+1.27+*%281+Militesla%29%2F%283*%284*pi*1e-7%29+Henry%2Fm%29+in+attoJoules%2FMilitessla
  #### used to be
  # https://www.wolframalpha.com/input?i=4*pi*%282.25+um%29%5E3+*+1.06+*%281+mT%29%2F%283*%284*pi*1e-7%29+Henry%2Fm%29+in+attoJoules%2Fmilitessla


  Diam = 2.0*blob_radius
  C = 0.3 #* (mu_full*mu_full/(Diam*Diam*Diam*Diam)) #Used to be (2.0*np.pi/5.0)
  # https://www.wolframalpha.com/input?i=%283%2F%284*pi%29%29*+%281+attoJoule+%2Fmillitesla+%29%5E2+*+%284*pi*1e-7+Henry%2Fm%29+%2F+%281+um%29%5E4+to+pN
  

  B_z = kwargs.get('B_z')

  m_z = RB_0*B_z
  C_z = 0.3 #

  
  print('C = '+str(C))
  print('C_z = '+str(C_z))


  
  zax = np.zeros((Nblobs, 3))
  B_torque = np.zeros((Nblobs, 3))

  Omega = kwargs.get('B_field_freq')

  m_rot = np.array([np.cos(2*np.pi*(Omega)*time_s), np.sin(2*np.pi*(2*Omega)*time_s), 0.0])
  
  chi_exp = 1.27 # used to be 1.06
  Mdot_Mag = partial(Magnetic_Mobility,r_vectors=r_vectors,L=L,a=blob_radius,chi=chi_exp,RB_0=RB_0)
  Mdot_E = partial(Magnetic_Mobility,r_vectors=r_vectors,L=L,a=blob_radius,chi=1.0,RB_0=((1.0/chi_exp)*RB_0))


  mom_perm = np.zeros((Nblobs, 3))
  B_applied = np.zeros((Nblobs, 3))
  B_E = np.zeros((Nblobs, 3))
  for k, b in enumerate(bodies):
    R = b.orientation.rotation_matrix()
    mom_perm[k,:] = mu_dipole*R[:,0].T
    B_applied[k,:] = B_0*m_rot
    B_E[k,:] = B_z*np.array([0.0,0.0,1.0])

  #RHS_mag = B_applied.flatten() - Mdot_Mag(mom_perm.flatten()) + (1.0/RB_0)*mom_perm.flatten()
  RHS_E = B_E.flatten()

  system_size = Nblobs*3
  A = spla.LinearOperator((system_size, system_size), matvec = Mdot_Mag, dtype='float64')
  Ae = spla.LinearOperator((system_size, system_size), matvec = Mdot_E, dtype='float64') 
  #counter = gmres_counter() 
  induced_mom = RB_0*B_applied.flatten()
  # (induced_mom, info_precond_mag) = spla.gmres(A,                        
  #                                              RHS_mag, 
  #                                              x0=None, #(RB_0*B_applied.flatten())
  #                                              rtol=1.0e-7,
  #                                              restart=100, 
  #                                              maxiter=100) #, callback=counter

  E_mom = RB_0*RHS_E
  #counter = gmres_counter()
  # (E_mom, info_precond_mag) = spla.gmres(Ae,                        
  #                                        RHS_E, 
  #                                        x0=(RB_0*RHS_E), 
  #                                        rtol=1.0e-4,
  #                                        restart=100, 
  #                                        maxiter=100) #, callback=counter

  induced_mom = induced_mom.reshape((Nblobs,3))
  E_mom = E_mom.reshape((Nblobs,3))
  moments = np.zeros((Nblobs, 3))
  #print('+++++++++++++++++++++++++')
  for k, b in enumerate(bodies):
    R = b.orientation.rotation_matrix()
    moments[k,:] = mom_perm[k,:] + induced_mom[k,:]
    zax[k,:] = R[:,2].T
    B_torque[k,:] = B_0*np.cross(moments[k,:],m_rot)

  #print(moments)
  #print('+++++++++++++++++++++++++')


  
  r_vectors_img = 1.0*r_vectors
  r_vectors_img[:,2] *= -1.0
  r_vectors_images = r_vectors #np.concatenate((r_vectors,r_vectors_img),axis=0) #
  E_mom_image = E_mom #np.concatenate((E_mom,1.0*E_mom),axis=0) #
  #########################################################
  force_mag, torque_mag = blob_blob_force_numba(r_vectors, r_vectors_images, moments, E_mom, E_mom_image, zax, L, blob_radius, repulsion_strength_firm, debye_length_firm, firm_delta, repulsion_strength, debye_length, C, B_torque, C_z)
  force_blobs += force_mag
  force_torque_bodies[0::2,:] += force_blobs
  force_torque_bodies[1::2,:] += torque_mag
  #########################################################


  return force_torque_bodies


def preprocess(bodies, *args, **kwargs):
  '''
  This function is call at the start of the schemes.
  The default version do nothing, it should be modify by
  the user if he wants to change the schemes.
  '''
  return

def postprocess(bodies, *args, **kwargs):
  '''
  This function is call at the end of the schemes but
  before checking if the postions are a valid configuration.
  The default version do nothing, it should be modify by
  the user if he wants to change the schemes.
  '''
  return


# Override force interactions by user defined functions.
# This only override the functions implemented in python.
# If user_defined_functions is empty or does not exists
# this import does nothing.
user_defined_functions_found = False
if os.path.isfile('user_defined_functions.py'):
  user_defined_functions_found = True
if user_defined_functions_found:
  import user_defined_functions

