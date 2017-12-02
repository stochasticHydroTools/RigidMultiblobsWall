# Boomerangs examples

Here we provide codes and input files to simulate a two boomerang
problem and a suspension of boomerangs as in the paper:

1. **Brownian dynamics of rigid body suspensions**, Brennan Sprinkle,
Florencio Balboa Usabiaga, Neelesh Patankar, and Aleksandar Donev. In
preraration (2017).

In this folder we provide the python files

```
user_defined_functions.py
forces_pycuda_user_defined.py
```

the input files

```
inputfile_two_boomerangs.dat
inputfile_boomerang_suspension.dat
```

and the clones files

```
two_boomerangs_N_15.clones
boomerang_suspension_N_15.clones
```

The python files override the default force implementations 
(described in RigidMultiblobsWall/doc/USAGE.pdf) with new implementation 
as we explain below.
The file `user_defined_functions.py` re-implements the python functions
to compute the blob-blob, blob-wall and body-body interactions.
The file `forces_pycuda_user_defined.py` re-implements the pycuda
function to compute the blob-blob interactions.

To run these examples you only need to move to this folder and 
copy the code `RigidMultiblobsWall/multi_bodies/multi_bodies.py` to
this folder
(i.e. `cp ../../multi_bodies.py ./`). Then you can run the
examples like 

```
python multi_bodies.py --input-file inputfile_two_boomerangs.dat
```

## Two-boomerangs problem
The input file `inputfile_two_boomerangs.dat` can be used to run a Brownian dynamic simulation
of two boomerang colloidal particles attached by an harmonic spring as
described in Ref 1. The initial configuration of the boomerangs is
given in the clones file `two_boomerangs_N_15.clones`.

This input file select the python implementations to compute the blob
mobility **M** and the matrix vector product **Mf**. However, we
suggest to use the C++ implementations which are much faster, see
documentation for details

## Boomerang suspension
The input file `inputfile_boomerang_suspension.dat` can be used to run a Brownian dynamic simulation
of a boomerang colloidal suspension with pseudo-periodic boundary
conditions, see Ref 1. The initial configuration of the boomerangs is
given in the clones file `boomerang_suspension_N_15.clones`.

Note, since this is a large system (256 rigid bodies) we use a pycuda implementation to
compute the matrix vector product **Mf**. Therefore, it is necessary
to have a GPU compatible with CUDA and pycuda installed to run this
example. 



## Blob-blob interactions
The pairwise blob-blob forces are derived from the potential

```
U(r) = eps * exp(-r / b) / r

with:
eps = repulsion_strength.
b = debye_length.
r = distance between blobs.
```

The values of `repulsion_strength` and `debye_length` are specified in the
input files.


## Blob-wall interactions
The blob-wall interactions are derived from the potential

```
  U = eps * a * exp(-(h-a) / b) / (h - a)

with:
eps = repulsion_strength_wall.
b = debye_length_wall.
a = blob_radius.
h = distance from blob to wall.
```

The values of repulsion_strength_wall, debye_length_wall and
blob_radius are specified in the input files.

## Body-body interactions
The torque is zero and the pairwise forces between bodies are derived from the
potential

```
U(r) = 0.5 * eps * (r - 1.0)**2

with:
eps = repulsion_strength
r = distance between bodies location  points.
```
