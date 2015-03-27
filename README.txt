This package contains several scripts for doing some simple
simulations of rigid bodies made out of "blob" particles rigidly
connected.  These simulations consider the particles near a single
wall (floor), which is always at z = 0. The blob mobility used for
this is that from appendix C of the Swan and Brady paper:
		 'Simulation of hydrodynamically interacting particles near a
		 no-slip boundary.'


The software is organized as follows.  

The quaternion_integratorand fluids subfolders have some general
functions and objects that are used to perform simulations.  

The icosahedron, tetrahedron, sphere, and boomerang have mobilities
and functions used to simulate specific rigid bodies, along with
scripts to create trajectory data, analyze it, plot results, and
several tests for each body.

The utils.py file has some functions that are shard among all of the
different rigid bodies (mostly for analyzing data, etc).

The constrained_integrator and constrained_integrator_test files and
the files in the cosine_curve folder define objects used for more
general constrained diffusion, and are somewhat orthogonal to the
rest of the code.  I will not discuss them here.


To use:

1) Define the data where you want to save your data by making a copy 
of config.py called "config_local.py" and define DATA_DIR to your
liking. (In the future, any additional configuration variables will be set in
this file.)

2) Some code uses c++ through the Boost Python library for speedup.
There are Makefiles provided, which may need to be modified slightly
to reflect your Python version, etc.  Running make in these folders
will then compile the .so files used by the python programs.

(for the orientation simulations it should just be the mobility code
in the fluids directory.  If you do not have boost, or do not want to
use it, read the section below entitled "Without Boost").  



3) You should now be ready to run some scripts to produce trajectory
data.  To test this, cd into ./tetrahedron/, and try to run:

   python tetrahedron_free.py -dt 1.0 -N 500 --data-name=testing

You should see a log tracking the progress of this run in
./tetrahedron/logs/, and after its conclusion, you should have a few .txt
trajectory data file in <DATA_DIR>/tetrahedron/.  

Trajectory data is saved as a list of locations, each of which is 3
floats printed on a separate line, and orientations, each of which is
4 floats printed on a separate line.  The timestep 'dt' and number of
steps is also saved in the data file, to recreate the time for each 
point in the trajectory.

NOTE: My current approach is to also bin the equilibrium distribution while
saving the trajectory, since it requires little extra computation and
storage, and allows one to quickly check the distribution.  This data
will be saved in ./tetrahedron/data, ./icosahedron/data, etc. as a
python specific .pkl file which can be read by corresponding plotting
scripts.

4) One now analyzes the scripts to calculate the MSD (new scripts can
be made to calculate other quantities of interest from trajectory
data.) For now, these scripts have hardcoded data files in them to
analyze, but I will change this in the future.

5) Plotting the analyzed data is done with individual plotting scripts
located in each folder.   



#############
WITHOUT BOOST
#############
If you do not want to use boost for the brownian dynamics simulations,
the ./fluids/mobility.py file has a "single_wall_fluid_mobility"
function which is identical to "boosted_single_wall_fluid_mobility"
but doesn't use boost (and is somewhat slower).  Replace calls to the
boosted version with calls to the python version wherever mobility of
a rigid body is calculated (for example in "force_and_torque_mobility"
in ./tetrahedron/tetrahedron_free.py).  Then remove the 
"import mobility_ext as me" 
line from ./fluids/mobility.py and everything should run without having 
to compile any of the c++ code.







				
