# Documentation RigidMultiblobsWall/boomerang/

This package contains several scripts for doing some simple
simulations of rigid bodies made out of "blob" particles rigidly
connected.  These simulations consider the particles near a single
wall (floor), which is always at z = 0. The blob mobility used for
this is that from appendix C of the paper:
'Simulation of hydrodynamically interacting particles near a no-slip boundary'
by James Swan and John Brady, Phys. Fluids 19, 113306 (2007).

For the theory consult:

1. **Brownian Dynamics of Confined Rigid Bodies**, S. Delong, F. Balboa Usabiaga and A. Donev. 
The Journal of Chemical Physics, **143**, 144107 (2015). [DOI](http://dx.doi.org/10.1063/1.4932062) [arXiv](http://arxiv.org/abs/1506.08868)

The software is organized as follows.  

The quaternion_integrator and fluids subfolders have some general
functions and objects that are used to perform simulations.  

The boomerang folder contains an example with mobilities and functions
used to simulate the diffusion of a rigid boomerang particle near a
single wall, as described in Section IV.E in [1].
It also contains some scripts to analyze trajectory data
and plot results.

The sphere folder contains an example to simulate a sphere
whose center of mass is displaced from the geometric center
(i.e., gravity generates a torque), sedimented near a no-slip wall
in the presence of gravity, as described in Section IV.C in [1].
Unlike the boomerang example this code does not use a rigid
multiblob model of the sphere but rather uses the best known
(semi)analytical approximations to the sphere mobility.

The utils.py file has some general functions that would be useful for
general rigid bodies (mostly for analyzing and reading trajectory
data and for logging).

Trajectory data is saved as a list of locations and orientations. Each
timestep is saved as 7 floats printed on a separate line, with
location as the first 3 floats, and the quaternion representing
orientation as the last 4 floats.  The timestep 'dt' and number of
steps is also saved in the data file, to recreate the time for each
point in the trajectory.  This is human readable and could be analyzed
by other codes if this is convenient.  One can also extract the locations,
orientations, and parameters using the read_trajectory_data_from_txt
function in utils.py.


## To run the Boomerang example:

1. Define the directory where you want to save your data by making a copy 
of config.py called "config_local.py" and define DATA_DIR to your
liking. (In the future, any additional configuration variables will be set in
this file.)

2. Some code (the fluid mobility) uses C++ through the Boost Python
library for speedup.  There is a Makefile provided in the fluids
subfolder, which will need to be modified slightly to reflect your
Python version, etc.  Running make in this folder will then compile
the .so files used by the python programs.

(NOTE: If you do not have boost, or do not want to
use it, read the section below entitled "Without Boost").  


3. You should now be ready to run some scripts to produce trajectory
data.  To test this, cd into ./boomerang/, and try to run:

```
python boomerang.py -dt 0.01 -N 100 -gfactor 1.0 --data-name=testing-1
```

You should see a log tracking the progress of this run in
`./boomerang/logs/`, and after its conclusion, you should have a few `.txt`
trajectory data file in `<DATA_DIR>/boomerang/`. 

To get a description of all command line arguments available for
boomerang.py, run:

```
python boomerang.py --help
```
	 
Note that when running multiple runs to be analyzed for MSD, you
**MUST** end data-name with a hyphen and an integer, starting at 1 and 
increasing successively.  e.g. `--data-name=heavy-masses-1`, 
`--data-name=heavy-masses-2`, etc.

One can create frames of this trajectory to make a animation using the script:

```
python plot_boomerang_trajectory boomerang-trajectory-dt-0.01-N-100-scheme-RFD-g-1.0-testing-1.txt
```

4. One can now analyze the scripts to calculate the MSD (new scripts
can be made to calculate other quantities of interest from trajectory
data.)  There is a script in ./boomerang which takes command line
arguments to specify which files to analyze.  From the boomerang
directory, run:
	
```	 
python boomerang.py -dt 0.01 -N 1000 -gfactor 1.0 --data-name=testing-1
python boomerang.py -dt 0.01 -N 1000 -gfactor 1.0 --data-name=testing-2
```

	(These above might take some time to run)

Then when the trajectories are done, run:

```
python calculate_boomerang_msd_from_trajectories.py -dt 0.01 -N 1000 --data-name=testing -n_runs 2 -gfactor 1.0 -end 1.0
```

This will create the file:
  `boomerang-msd-dt-0.01-N-1000-end-1.0-scheme-RFD-g-1.0-runs-2-testing.pkl`
in  `<DATA_DIR>/boomerang`.

5) Plotting the MSD can then be done by running:
  
```
python plot_boomerang_msd.py boomerang-msd-dt-0.01-N-1000-end-1.0-scheme-RFD-g-1.0-runs-2-testing.pkl
```

which will create a (noisy) pdf in the boomerang/figures/ folder.


## To run the sphere example:

1. Define the directory where you want to save your data by making a copy 
of config.py called `config_local.py` and define DATA_DIR to your
liking. (In the future, any additional configuration variables will be set in
this file.)

2. Define the sphere mobility function. In the file `sphere/sphere.py`,
chose one of the two mobility functions implemented. One function is based on an 
expansion in terms of "h", the sphere-wall distance, to order "h**5". 
The other function use a combination of theories (see function for details)
and a fit to the sphere mobility computed from a higher resolution model.





## WITHOUT BOOST

If you do not want to use boost for the brownian dynamics simulations,
the ./fluids/mobility.py file has a "single_wall_fluid_mobility"
function which is identical to "boosted_single_wall_fluid_mobility"
but doesn't use boost (and is somewhat slower).  Replace calls to the
boosted version with calls to the python version wherever mobility of
a rigid body is calculated (for example in "force_and_torque_boomerang_mobility"
in ./boomerang/boomerang.py).  Then remove the "import mobility_ext as me" 
line from ./fluids/mobility.py and everything should run without having 
to compile any of the C++ code.





				
