# Colloidal Crystals
Author: Brennan Sprinkle

This example contains the code to reproduce the simulation data shown in 
`Direct Observation of Colloidal Quasicrystallization`

### Run the code
To run the examples in this directory, ensure that the makefile has been succesfully run in the `Lubrication` base directory.
Individual examples are contained in the subdirectories: `Rhombus_to_Sigma`, `Ladder_to_Ring`, and `Square_to_Diamond`
To run an example, navigate to one of these directories and run
```
python main_chain.py --input-file inputfile_suspension.dat
```
or 
```
OMP_NUM_THREADS=1 OMP_PROC_BIND=false python main_chain.py --input-file inputfile_suspension.dat
```
which can be much faster on some systems. The `Rhombus_to_Sigma` to sigma example must be run in 3 stages, all of which require specific seeds (storred in the `random_states` sub-directory) to achieve the desired final configuration. To run different stages, simply coment/uncomment the relevant initial configuration in the input file `inputfile_suspension.dat` e.g
```
#### Uncomment for the first 10s of dipole equilibriation ####
structure ../blob.vertex ./suspension_rhombus_N_12_random.clones
```
and the correct random state will be used automatically. Other examples e.g `Ladder_to_Ring` are less sensitive and only require one stange/one random state. 

In each example, the main file `main_chain.py` contains the logic for the trajectory of values of the repulsive field E_z.

After running each stage of each example, data will be placed in the `./data` directory and can be visualized with the MATLAB code `Plot_Data.m`. Follow the comments in this MATLAB code to vizualize different cases.

The MATLAB code uses `mArrow3.m` -- an arrow rendering code from MATLAB file exchange:

Georg Stillfried (2025). mArrow3.m - easy-to-use 3D arrow (https://www.mathworks.com/matlabcentral/fileexchange/25372-marrow3-m-easy-to-use-3d-arrow), MATLAB Central File Exchange. Retrieved January 7, 2025. 
