# Documentation
This package contains several python codes to run simulations of 
rigid bodies made out of _blob particles_ rigidly connected near
a single wall (floor). These codes allow the user to calculate the
mobility of complex shape objects, solve mobility or resistance problems
for suspension of rigid bodies or run deterministic or stochastic 
dynamic simulations.

We explain in the next sections how to use the package.
For the theory consult the references:

1. **Brownian Dynamics of Confined Rigid Bodies**
  S. Delong, F. Balboa Usabiaga and A. Donev. The Journal of Chemical Physics, **143**, 144107 (2015). 
[DOI](http://dx.doi.org/10.1063/1.4932062) [arXiv](http://arxiv.org/abs/1506.08868)
2. **Hydrodynamics of suspensions of passive and active rigid particles: a
  rigid multiblob approach** F. Balboa Usabiaga, B. Kallemov, B. Delmotte,
  A. Pal Singh Bhalla, B. E. Griffith and A. Donev. [arXiv](http://arxiv.org/abs/1602.02170)


## 1. Prepare the package
The codes are implemented in python and it is not necessary to compile the package to use it. 
However, we provide alternative implementations in _C_ and _pycuda_ for some of the most computationally 
expensive functions. You can skip to section 2 but come back if you
want to take fully advantage of this package.


### 1.1 Prepare the mobility functions
The codes use functions to compute the blob mobility matrix **M** and the
matrix vector product **Mf**. For some functions we provide
a _C_ implementation which can be around 5 times faster than the python version. We also
provide _pycuda_ implementations which, for large systems, can be orders of magnitude faster.
To use the _C_ implementation move to the directory `mobility/` and compile
`mobility_ext.cc` to a `.so` file using the Makefile provided (which you will need 
to modified slightly to reflect your Python version, etc.).

To use the _pycuda_ implementation all you need is _pycuda_ and a GPU compatible with CUDA;
you don't need to compile any additional file in this package.

### 1.2 Blob-blob forces
In dynamical simulations it is possible to include blob-blob interactions to,
for example, simulate a colloid suspension with a given steric repulsion. 
Again, we provide versions in _python_, _C_ and _pycuda_. To use the _C_
version move to the directory `multi_bodies/` and compile `forces_ext.cc` to
a `.so` file using the Makefile provided (which you will need 
to modified slightly to reflect your Python version, etc.).

To use the _pycuda_ implementation all you need is _pycuda_ and a GPU compatible with CUDA;
you don't need to compile any additional file in this package.

## 2. Rigid bodies configuration
We use a vector (3 numbers) and a quaternion (4 numbers) to represent the 
location and orientation of each body, see Ref. [1](http://dx.doi.org/10.1063/1.4932062) for details.
This information is saved by the code in the `*.clones` files,
with format:

```
number_of_rigid_bodies
vector_location_body_0 quaternion_body_0
vector_location_body_1 quaternion_body_1
.
.
.
```

For example, see the file `multi_bodies/Structures/boomerang_N_15.clones` to 
see the representation of a boomerang-like particle with location (0, 0, 10)
and orientation given by the quaternion (0.5, 0.5, 0.5, 0.5).

The coordinates of the blobs forming a rigid body in the default configuration
(location (0, 0, 0) and default quaternion (1, 0, 0, 0)) are given to the codes 
with `*.vertex` files. The format of these files is:

```
number_of_blobs_in_rigid_body
vector_location_blob_0
vector_location_blob_1
.
.
.
```

For example, the file `multi_bodies/Structures/boomerang_N_15.vertex` gives the
structure of a boomerang-like particle formed by 15 blobs.



## 3. Run static simulations
We start explaining how to compute the mobility of a rigid body close to a wall.
First, move to the directory `multi_bodies/` and inspect the input file 
`inputfile_bodies_mobility.dat`:

---

```
# Select problem to solve
scheme                                   bodies_mobility

# Select implementation to compute the blobs mobility 
# Options: python and boost
mobility_blobs_implementation            python

# Set fluid viscosity (eta) and blob radius
eta                                      1.0 
blob_radius                              0.25

# Set output name
output_name                              data/run.bodies_mobility

# Load rigid bodies configuration
structure	Structures/boomerang_N_15.vertex Structures/boomerang_N_15.clones
```

---

Now, to run the code, use

`python multi_bodies_utilities.py --input-fule inputfile.dat`



### How to run 
The main program `multi_bodies/multi_bodies.py` can be run like

`python multi_bodies inputfile`

`inputfile` contains the options for the simulation (time steps, number of bodies...),
see `multi_bodies/data.main` for an example. The trajectory data is saved as a list of 
locations and orientations in the output file `.bodies`. Each timestep is saved as 7 
floats per body printed on a separate line, with location as the first 3 floats, and 
the quaternion representing orientation as the last 4 floats.

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
the schemes to integrate the equations of motion.
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


### To run the sphere example:
1) Define the directory where you want to save your data by making a copy 
of config.py called "config_local.py" and define DATA_DIR to your
liking. (In the future, any additional configuration variables will be set in
this file.)

2) Define the sphere mobility function. In the file "sphere/sphere.py",
chose one of the two mobility functions implemented. One function is based on an 
expansion in terms of "h", the sphere-wall distance, to order "h**5". 
The other function use a combination of theories (see function for details)
and a fit to the sphere mobility computed from a higher resolution model.








				
