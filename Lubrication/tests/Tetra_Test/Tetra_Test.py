import argparse
import numpy as np
import scipy.linalg
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
import subprocess
try:
  import pickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle
from functools import partial
import sys
import time
import copy
import scipy.sparse as sp
from sksparse.cholmod import cholesky

# Find project functions
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')


from Lub_Solver import Lub_Solver as LS
from stochastic_forcing import stochastic_forcing as stochastic
from mobility import mobility as mb
from body import body
from read_input import read_input
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_slip_file
import general_application_utils
            
def make_res_coefs(LSolv, a, L):
    R_coefs = np.loadtxt('../../Resistance_Coefs/res_scalars_MB_1.txt')
    h = R_coefs[:, 0]
    
    max_h = np.max(h)
    print('max height: ' + str(max_h))
    
    domain_w = 2*max_h*a
    
    rv = [np.array([L[0]/2.0,L[1]/2.0,2*a]),np.array([L[0]/2.0,L[1]/2.0,4.5*a])]
    LSolv.Init_DPStokesSolver(z_max=domain_w, r_vecs_np=rv, domain='DP')
    
    Rmb_coefs = 0*R_coefs
    
    for j in range(int(len(h)/2)):
        print(str(200.0*j/len(h))+'% done with coefs')
        h_test = h[2*j]
        RXa11_test = R_coefs[2*j,1]
        RXa12_test = R_coefs[2*j+1,1]
        RYa11_test = R_coefs[2*j,2]
        RYa12_test = R_coefs[2*j+1,2]
        RYb11_test = R_coefs[2*j,3]
        RYb12_test = R_coefs[2*j+1,3]
        RXc11_test = R_coefs[2*j,4]
        RXc12_test = R_coefs[2*j+1,4]
        RYc11_test = R_coefs[2*j,5]
        RYc12_test = R_coefs[2*j+1,5]
        
        
        rv = [np.array([L[0]/2.0,L[1]/2.0,domain_w/2.0-h_test*a/2.0]),np.array([L[0]/2.0,L[1]/2.0,domain_w/2.0+h_test*a/2.0])]
        
                
        Mob = LSolv.Form_Mobility(rv)
        Res = np.matrix(Mob).I
        
        fact_tt = 1.0/(6*np.pi*eta*a)
        fact_tr = 1.0/(6*np.pi*eta*a*a)
        fact_rr = 1.0/(6*np.pi*eta*a*a*a)
        
        RmbXa11 = fact_tt*Res[3-1,3-1]
        RmbYa11 = fact_tt*Res[1-1,1-1]
        RmbYb11 = fact_tr*Res[1-1,5-1]
        RmbXc11 = fact_rr*Res[6-1,6-1]
        RmbYc11 = fact_rr*Res[5-1,5-1]
        
        
        RmbXa12 = fact_tt*Res[3-1,6+3-1]
        RmbYa12 = fact_tt*Res[1-1,6+1-1]
        RmbYb12 = -1.0*fact_tr*Res[1-1,6+5-1]
        RmbXc12 = fact_rr*Res[6-1,6+6-1]
        RmbYc12 = fact_rr*Res[5-1,6+5-1]
        
        Rmb_coefs[2*j,0] = h_test
        Rmb_coefs[2*j+1,0] = h_test
        Rmb_coefs[2*j,1] = RmbXa11
        Rmb_coefs[2*j+1,1] = RmbXa12
        Rmb_coefs[2*j,2] = RmbYa11
        Rmb_coefs[2*j+1,2] =  RmbYa12
        Rmb_coefs[2*j,3] = RmbYb11
        Rmb_coefs[2*j+1,3] = RmbYb12
        Rmb_coefs[2*j,4] = RmbXc11
        Rmb_coefs[2*j+1,4] = RmbXc12
        Rmb_coefs[2*j,5] = RmbYc11
        Rmb_coefs[2*j+1,5] = RmbYc12
        
    
    np.savetxt('../../Resistance_Coefs/DP_res_scalars_MB_1.txt',Rmb_coefs,fmt='%1.12f')
            
def make_wall_coefs(LSolv, a, L):
    R_coefs = np.loadtxt('../../Resistance_Coefs/res_scalars_wall_MB.txt')
    M2562_coefs = np.loadtxt('../../Resistance_Coefs/mob_scalars_wall_MB_2562.txt')
    h_2562 = M2562_coefs[:, 0]
    
    
    max_h = np.max(h_2562)
    print('max height: ' + str(max_h))
    
    rv = [np.array([L[0]/2.0,L[1]/2.0,2*a])]
    LSolv.Init_DPStokesSolver(z_max=2*max_h*a, r_vecs_np=rv)
    
    Spect_R = 0*M2562_coefs
    Eig_thresh_R = 0*M2562_coefs
    Error = 0*h_2562
    
    for j in range(len(h_2562)):
        print(str(100.0*j/len(h_2562))+'% done with coefs')
        h_test = h_2562[j]
        RXa_test = R_coefs[j,1]
        RYa_test = R_coefs[j,2]
        RYb_test = R_coefs[j,3]
        RXc_test = R_coefs[j,4]
        RYc_test = R_coefs[j,5]
        
        
        M2562Xa = M2562_coefs[j,1]
        M2562Ya = M2562_coefs[j,2]
        M2562Yb = M2562_coefs[j,3]
        M2562Xc = M2562_coefs[j,4]
        M2562Yc = M2562_coefs[j,5]
        
        
        M2562denom = M2562Ya*M2562Yc - M2562Yb*M2562Yb
        R2562Xa = 1.0/M2562Xa
        R2562Ya = M2562Yc/M2562denom
        R2562Yb = -M2562Yb/M2562denom
        R2562Xc = 1.0/M2562Xc
        R2562Yc = M2562Ya/M2562denom 
        
        R_2562 = np.array([[R2562Ya, 0, 0, 0, R2562Yb, 0],
               [0, R2562Ya, 0, -R2562Yb, 0, 0],
               [0, 0, R2562Xa, 0, 0, 0],
               [0, -R2562Yb, 0, R2562Yc, 0, 0],
               [R2562Yb, 0, 0, 0, R2562Yc, 0],
               [0, 0, 0, 0, 0, R2562Xc]])
        
        
        rv = [np.array([L[0]/2.0,L[1]/2.0,h_test*a])]    
        
        Mob = LSolv.Form_Mobility(rv)
        
        fact = 6*np.pi*eta*a
        Xa = fact*Mob[3-1,3-1]
        Ya = fact*Mob[1-1,1-1]
        Yb = fact*a*Mob[1-1,5-1]
        Xc = fact*a*a*Mob[6-1,6-1]
        Yc = fact*a*a*Mob[5-1,5-1]
        
        denom = Ya*Yc - Yb*Yb
        RXa = 1.0/Xa
        RYa = Yc/denom
        RYb = -Yb/denom
        RXc = 1.0/Xc
        RYc = Ya/denom 
        
        
        #######################
        Spect_R[j,0] = h_test
        Spect_R[j,1] = RXa
        Spect_R[j,2] = RYa
        Spect_R[j,3] = RYb
        Spect_R[j,4] = RXc
        Spect_R[j,5] = RYc
        #######################
        
        
        R_mb = np.array([[RYa, 0, 0, 0, RYb, 0],
                [0, RYa, 0, -RYb, 0, 0],
                [0, 0, RXa, 0, 0, 0],
                [0, -RYb, 0, RYc, 0, 0],
                [RYb, 0, 0, 0, RYc, 0],
                [0, 0, 0, 0, 0, RXc]])
                
        Delta_R = R_2562-R_mb
        u, s, vh = np.linalg.svd(Delta_R, full_matrices=True)
        s[s<1e-10] = 0
        Delta_clip = np.dot(u * s, vh)
        R_2562_clip = Delta_clip+R_mb
        
        
        RcXa = R_2562_clip[3-1,3-1]
        RcYa = R_2562_clip[1-1,1-1]
        RcYb = R_2562_clip[1-1,5-1]
        RcXc = R_2562_clip[6-1,6-1]
        RcYc = R_2562_clip[5-1,5-1]
        
        
        denom_c = RcYa*RcYc - RcYb*RcYb
        MclipXa = 1.0/RcXa
        MclipYa = RcYc/denom_c
        MclipYb = -RcYb/denom_c
        MclipXc = 1.0/RcXc
        MclipYc = RcYa/denom_c 
        
        
        Eig_thresh_R[j,0] = h_test
        Eig_thresh_R[j,1] = MclipXa
        Eig_thresh_R[j,2] = MclipYa
        Eig_thresh_R[j,3] = MclipYb
        Eig_thresh_R[j,4] = MclipXc
        Eig_thresh_R[j,5] = MclipYc
        
        
        Spect_R[j,0] = h_test
        Spect_R[j,1] = RXa
        Spect_R[j,2] = RYa
        Spect_R[j,3] = RYb
        Spect_R[j,4] = RXc
        Spect_R[j,5] = RYc
        
        Error[j] = np.linalg.norm(R_2562_clip-R_2562)
        
        #print('RXa: '+str(RXa)+' --- '+str(RXa_test))
        #print('RYa: '+str(RYa)+' --- '+str(RYa_test))
        #print('RYb: '+str(RYb)+' --- '+str(RYb_test))
        #print('RXc: '+str(RXc)+' --- '+str(RXc_test))
        #print('RYc: '+str(RYc)+' --- '+str(RYc_test))
    
    

    np.savetxt('../../Resistance_Coefs/DP_case_res_scalars_wall_MB.txt',Spect_R,fmt='%1.12f')
    np.savetxt('../../Resistance_Coefs/DP_case_mob_scalars_wall_MB_2562_eig_thresh.txt',Eig_thresh_R,fmt='%1.12f')

if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description='Run a multi-body simulation and save trajectory.')
    parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', help='name of the input file')
    parser.add_argument('--print-residual', action='store_true', help='print gmres and lanczos residuals')
    args = parser.parse_args()
    input_file = args.input_file

    # Read input file
    read = read_input.ReadInput(input_file)

    # Set some variables for the simulation
    eta = read.eta
    a = read.blob_radius
    output_name = read.output_name
    structures = read.structures
    print(structures)
    structures_ID = read.structures_ID


    # Create rigid bodies
    bodies = []
    body_types = []
    body_names = []
    for ID, structure in enumerate(structures):
        print('Creating structures = ', structure[1])
        # Read vertex and clones files
        struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
        num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(
            structure[1])
        # Read slip file if it exists
        slip = None
        if(len(structure) > 2):
            slip = read_slip_file.read_slip_file(structure[2])
        body_types.append(num_bodies_struct)
        body_names.append(structures_ID[ID])
        # Create each body of type structure
        for i in range(num_bodies_struct):
            b = body.Body(
                struct_locations[i], struct_orientations[i], struct_ref_config, a)
            b.ID = structures_ID[ID]
            # Calculate body length for the RFD
            if i == 0:
                b.calc_body_length()
            else:
                b.body_length = bodies[-1].body_length
            # Append bodies to total bodies list
            bodies.append(b)
    bodies = np.array(bodies)

    # Set some more variables
    num_of_body_types = len(body_types)
    num_bodies = bodies.size
    num_particles = len(bodies)
    Nblobs = sum([x.Nblobs for x in bodies])
    
    cutoff = read.Lub_Cut
    
    #L = read.periodic_length
    Lbig = 100*a
    L = np.array([Lbig, Lbig, 0])
    
    n_steps = read.n_steps 
    n_save = read.n_save
    dt = read.dt 
    
    print(L)
    
    for b in bodies:
      b.location[0] += L[0]/2.0
      b.location[1] += L[1]/2.0
    
    
    firm_delta = read.firm_delta
    debye_length_delta = 2.0*a*firm_delta/np.log(1.0e1) 
    repulsion_strength_delta = read.repulsion_strength_firm
    
    LSolv = LS(bodies,a,eta,cutoff,L,debye_length=firm_delta)
    LSolv.dt = dt
    LSolv.kT = read.kT
    LSolv.tolerance = 1e-7
    
    
    #make_wall_coefs(LSolv, a, L)
    #make_res_coefs(LSolv, a, L)
    
    max_h = (500 + 1.0) * 0.01 + 2*a
    print('max height: ' + str(max_h))
    
    rv = [b.location for b in bodies]
    # LSolv.Init_DPStokesSolver(z_max=2*max_h, r_vecs_np=rv)
    
    
    d_count = -1
    Mob_list = []
    for h in range(500): #range(500)
      print(h) 
      height = 1.0 + (h + 1.0) * 0.01 #
      dist = (h + 1.0) * 0.01
      d_count += 1
      new_bodies = []
      for k, b in enumerate(bodies):
        n_b = copy.deepcopy(b)
        new_bodies.append(n_b)
        new_bodies[k].location[0] -= L[0]/2.0
        new_bodies[k].location[1] -= L[1]/2.0
        new_bodies[k].location[2] -= 2.0
        factor = (2*a + dist*a)/3.0
        new_bodies[k].location *= factor
        new_bodies[k].location[2] += height
        new_bodies[k].location[0] += L[0]/2.0
        new_bodies[k].location[1] += L[1]/2.0
        #print(new_bodies[k].location)
      LSolv.bodies = new_bodies
      LSolv.Set_R_Mats()
      Mob = LSolv.Form_Lub_Mobility();
      Mobf = Mob.flatten()
      h_n_mob = np.append([height],Mobf)
      Mob_list.append(h_n_mob)
      #if h == 0:
        #f = open("./Tet_Data/tetra_Spec_Lub_mob.dat", "w")
        #np.savetxt(f, np.array([height]), newline=" ",fmt='%10.2f')
        #f = open("./Tet_Data/tetra_Lub_mob.dat", "a")
        #np.savetxt(f, Mobf, newline=" ",fmt='%10.7f')
        #f.write('\n')
      #else:
        #f = open("./Tet_Data/tetra_Spec_Lub_mob.dat", "a")
        #np.savetxt(f, np.array([height]), newline=" ",fmt='%10.2f')
        #np.savetxt(f, Mobf, newline=" ",fmt='%10.7f')
        #f.write('\n')
      
    Mob_of_h = np.array(Mob_list)
    print(Mob_of_h)
    f = "./Tet_Data/tetra_lub_pybind.dat"
    np.savetxt(f, Mob_list,fmt='%10.7f')
    
    
