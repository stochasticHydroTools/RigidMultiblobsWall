#!/bin/bash
N=4
for n in `seq 1 $N`;
do

echo "# Select integrator 
scheme                                   stochastic_traction_RFD1
#stochastic_Slip_Trapz
#stochastic_traction_RFD1

# Select implementation to compute M and M*f
mobility_blobs_implementation            C++
mobility_vector_prod_implementation      pycuda

# Select implementation to compute the blobs-blob interactions
blob_blob_force_implementation           pycuda
body_body_force_torque_implementation	 None

# Set time step, number of steps and save frequency
dt                                       0.01
n_steps                                  1000000
n_save                                   1
initial_step				 0

# Set periodic length
periodic_length				 45.339607409 45.339607409 -1

# Solver params
solver_tolerance                         1.0e-4

# Set fluid viscosity (eta), gravity (g) and blob radius
eta                                      8.9e-04
g                                        0.0003078768
blob_radius                              0.324557390919

# Stochastic parameters
kT                                       0.0041419464

# RFD parameters
rf_delta                                 1.0e-5

# Set parameters for the blob-blob interation
repulsion_strength                       0.095713728509
debye_length                             0.162278695459

# Set interaction with the wall
repulsion_strength_wall                  0.095713728509
debye_length_wall                        0.162278695459

# Set output name
output_name                              /hydro/bsprinkle/ManyBooms/EM_dt_01_tol_1e4_2g/em$n
save_clones                              one_file

# Seed the RNG
# seed                                     1

# Load rigid bodies configuration, provide
# *.vertex and *.clones files
structure Structures/boomerang_N_15.vertex Structures/many_booms/boomerang_N_15.suspension.N.256.phi.0.25.file.10$n.clones
" > inputfile.disp."$n"

CUDA_DEVICE=1 python multi_bodies.py --input-file inputfile.disp."$n" &>/dev/null &

echo "I just ran case " $n " for you"

done 

#for n in `seq 1 $N`;
#do
#rm inputfile.disp."$n"
#echo "I just removed file " $n " for you"
#done
