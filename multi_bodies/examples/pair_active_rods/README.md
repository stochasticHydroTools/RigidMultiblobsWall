# Example: Pair of active rods near a wall
This example reproduces the results obtained in **Section V.C** of the paper **Hydrodynamics of suspensions of passive and active rigid particles: a
rigid multiblob approach**, F. Balboa Usabiaga, B. Kallemov, B. Delmotte, 
A. Pal Singh Bhalla, B. E. Griffith and A. Donev. [arXiv](http://arxiv.org/abs/1602.02170)

Two active extensile rods placed near a wall rotate counterclockwise about the wall normal. 
If each resolution is correctly adjusted, the angular velocity of the rods should be similar for all the resolutions.
In this example we compute the instantenous angular velocity for each resolution.

In this test, gravity is set to zero and an ad-hoc repulsive force is added to balance the active slip velocity that pulls the rods towards the wall.

## 1. Description of the input files and slip distributions

Each resolution has a specific `inputfiles_*.dat` and `force*.dat` file. 

To generate the apropriate slip distribution on the rod surface, a special `slip_function.py` has been created. 
This function generate a tangential slip velocity on a given portion of the rod surface which does not include the extremities.

Each resolution has its own `.vertex` and `.clone` file in the folder `multi_bodies/Structures`.



## 2. How to run the example (e.g. with the low resolution)

First, copy the file
`RigidMultiblobsWall/multi_bodies/multi_bodies_utilities.py` to this folder.
Then, to run a given resolution you just need the command: 

`python multi_bodies_utilities.py --input-file inputfile_low_resolution.dat`

**Note:** If you have compiled the C++ boost library, it is recommended to change `mobility_blobs_implementation` and `mobility_vector_prod_implementation` to `C++` in the input file.
If you have PyCuda, change `mobility_vector_prod_implementation` to `pycuda`.


## 3. Output files and comparison with reference values (e.g. with the low resolution)

After running a resolution go to the folder specified for the output files (`data`) and open `run_low_res.velocity.dat`.
Each line provides the translational and angular velocity of each body.
Compare these value to the file `run_low_res.velocity.dat.reference` in the folder `example_pair_active_rods`. 

The difference between the computed and the reference value should not exceed `solver_tolerance` (typically `1e-8`) in the input file `inputfiles_low_resolution.dat`.
The angular velocity about the z-axis (i.e. the last values of each line) should be similar for all resolutions, i.e. close to 4.0/(2*pi) = 0.64 Hz. 
