## Documentation
This package contains several scripts for doing 
simulations of rigid bodies made out of "blob" particles rigidly
connected.  These simulations consider the particles near a single
wall (floor), which is always at z = 0. The blob mobility used for
this is that from appendix C of the paper:
'Simulation of hydrodynamically interacting particles near a no-slip boundary'
by James Swan and John Brady, Phys. Fluids 19, 113306 (2007).

For the theory consult:
* **Brownian Dynamics of Confined Rigid Bodies**
  S. Delong, F. Balboa Usabiaga and A. Donev. The Journal of Chemical Physics, **143**, 144107 (2015). 
[DOI](http://dx.doi.org/10.1063/1.4932062) [arXiv](http://arxiv.org/abs/1506.08868)

* **Hydrodynamics of suspensions of passive and active rigid particles: a
  rigid multiblob approach** F. Balboa Usabiaga, B. Kallemov, B. Delmotte,
  A. Pal Singh Bhalla, B. E. Griffith and A. Donev. [arXiv](http://arxiv.org/abs/1602.02170)


### Prepare the mobility functions
The functions to compute the blob mobility matrix **M** and the
product **M f** are defined in the directory `mobility/`.
Some of the functions use pycuda or C++ (through the Boost Python
library) for speed up. To use the C++ implementation compile
`mobility_ext.cc` to a `.so` file using the Makefile provided (which will need 
to be modified slightly to reflect your Python version, etc.).
To use the pycuda implementation you will need pycuda and GPU compatible with CUDA.

Note, if you do not want to use the C++ or the pycuda implementations
edit the file mobility/mobility.py and comment the lines

`import mobility_ext as me`

`import mobility_pycuda`



### How to run 
The main program `multi_bodies/multi_bodies.py` can be run like

`python multi_bodies inputfile`

`inputfiles` contains the options for the simulation (time steps, number of bodies...),
see `multi_bodies/data.main` for an example. Trajectory data is saved as a list of 
locations and orientations. Each timestep is saved as 7 floats per body printed on a 
separate line, with location as the first 3 floats, and the quaternion representing
orientation as the last 4 floats.

You can modify the following
functions in `multi_bodies/multi_bodies.py`:

* `mobility_blobs`: it computes the blobs mobility matrix **M**. If
you are not using the C++ implementation select the python version.

* `mobility_vector_prod`: it computes the matrix vector product **Mf**.
If you are not using pycuda select the C++ or the python version.

* `force_torque_calculator_sort_by_bodies`: it computes the external forces
and torques on the rigid bodies. The current implementation only
includes gravity forces plus interactions between the blobs and the wall.



### Software organization
* **body/**: it contains a class to handle a single rigid body.
* **boomerang/**: See next section.
* **doc/**: documentation.
* **mobility/**: it has functions to compute the blob mobility matrix **M** and the
product **Mf**.
* **multi_bodies/**: the main code to run simulations of rigid bodies.
* **quaternion_integrator/**: it has a small class to handle quaternions and
the schemes to integrate the equations.
* **sphere/**: the folder contains an example to simulate a sphere
whose center of mass is displaced from the geometric center
(i.e., gravity generates a torque), sedimented near a no-slip wall
in the presence of gravity, as described in Section IV.C in [1].
Unlike the boomerang example this code does not use a rigid
multiblob model of the sphere but rather uses the best known
(semi)analytical approximations to the sphere mobility.
* **stochastic_forcing/**: it contains functions to compute the product
 **M**^{1/2}**z** necessary to perform Brownian simulations.
* **utils.py**: this file has some general functions that would be useful for
general rigid bodies (mostly for analyzing and reading trajectory
data and for logging).



### To run the Boomerang example:
This file permits to simulate the dynamics of a single boomerang close to a wall.
1) Define the directory where you want to save your data by making a copy 
of config.py called "config_local.py" and define DATA_DIR to your
liking. (In the future, any additional configuration variables will be set in
this file.)

2) Some code (the fluid mobility) uses C++ through the Boost Python
library for speedup.  There is a Makefile provided in the fluids
subfolder, which will need to be modified slightly to reflect your
Python version, etc.  Running make in this folder will then compile
the .so files used by the python programs.

(NOTE: If you do not have boost, or do not want to
use it, read the section below entitled "Without Boost").  


3) You should now be ready to run some scripts to produce trajectory
data.  To test this, cd into ./boomerang/, and try to run:

   python boomerang.py -dt 0.01 -N 100 -gfactor 1.0 --data-name=testing-1

You should see a log tracking the progress of this run in
./boomerang/logs/, and after its conclusion, you should have a few .txt
trajectory data file in <DATA_DIR>/boomerang/. 

To get a description of all command line arguments available for
boomerang.py, run:

	 python boomerang.py --help
	 
Note that when running multiple runs to be analyzed for MSD, you
*MUST* end data-name with a hyphen and an integer, starting at 1 and 
increasing successively.  e.g. --data-name=heavy-masses-1, 
--data-name=heavy-masses-2, etc.

One can create frames of this trajectory to make a animation using the script:
		python plot_boomerang_trajectory boomerang-trajectory-dt-0.01-N-100-scheme-RFD-g-1.0-testing-1.txt


4) One can now analyze the scripts to calculate the MSD (new scripts
can be made to calculate other quantities of interest from trajectory
data.)  There is a script in ./boomerang which takes command line
arguments to specify which files to analyze.  From the boomerang
directory, run:
		 
	python boomerang.py -dt 0.01 -N 1000 -gfactor 1.0 --data-name=testing-1
	python boomerang.py -dt 0.01 -N 1000 -gfactor 1.0 --data-name=testing-2

	(These above might take some time to run)

	Then when the trajectories are done, run:

	python calculate_boomerang_msd_from_trajectories.py -dt 0.01 -N 1000
	--data-name=testing -n_runs 2 -gfactor 1.0 -end 1.0

This will create the file:
  "boomerang-msd-dt-0.01-N-1000-end-1.0-scheme-RFD-g-1.0-runs-2-testing.pkl"
in  <DATA_DIR>/boomerang.

5) Plotting the MSD can then be done by running:
  
  python plot_boomerang_msd.py boomerang-msd-dt-0.01-N-1000-end-1.0-scheme-RFD-g-1.0-runs-2-testing.pkl

which will create a (noisy) pdf in the boomerang/figures/ folder.


####WITHOUT BOOST
If you do not want to use boost for the brownian dynamics simulations,
the ./fluids/mobility.py file has a "single_wall_fluid_mobility"
function which is identical to "boosted_single_wall_fluid_mobility"
but doesn't use boost (and is somewhat slower).  Replace calls to the
boosted version with calls to the python version wherever mobility of
a rigid body is calculated (for example in "force_and_torque_boomerang_mobility"
in ./boomerang/boomerang.py).  Then remove the "import mobility_ext as me" 
line from ./fluids/mobility.py and everything should run without having 
to compile any of the C++ code.


## To run the sphere example:
1) Define the directory where you want to save your data by making a copy 
of config.py called "config_local.py" and define DATA_DIR to your
liking. (In the future, any additional configuration variables will be set in
this file.)

2) Define the sphere mobility function. In the file "sphere/sphere.py",
chose one of the two mobility functions implemented. One function is based on an 
expansion in terms of "h", the sphere-wall distance, to order "h**5". 
The other function use a combination of theories (see function for details)
and a fit to the sphere mobility computed from a higher resolution model.








				
