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
which can be much faster on some systems. Each example directory contains relevant notes on the individual example.


Data from will be placed in the `./data` directory and can be visualized with the MATLAB code `Ring_Out.m`.
The MATLAB code uses `mArrow3.m` -- an arrow rendering code from MATLAB file exchange:

Georg Stillfried (2025). mArrow3.m - easy-to-use 3D arrow (https://www.mathworks.com/matlabcentral/fileexchange/25372-marrow3-m-easy-to-use-3d-arrow), MATLAB Central File Exchange. Retrieved January 7, 2025. 
