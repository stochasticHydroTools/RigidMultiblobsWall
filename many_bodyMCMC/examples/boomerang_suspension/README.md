# Boomerangs example

Here we provide codes and input files to simulate a Monte-Carlo
simulation for a suspension of boomerangs as in the paper:

1. **Brownian dynamics of rigid body suspensions**, Brennan Sprinkle,
Florencio Balboa Usabiaga, Neelesh Patankar, and Aleksandar Donev. In
preraration (2017).

In this folder we provide the python files

```
potential_pycuda_user_defined.py
```

the input files

```
inputfile_boomerang_suspension.dat
```

and the clones files

```
boomerang_suspension_N_15.clones
```

The python file override the default potential implementations 
(described in RigidMultiblobsWall/doc/USAGE.pdf) with new implementation 
as we explain below.

To run these examples you only need to move to this folder and 
copy the code `RigidMultiblobsWall/many_bodyMCMC/many_body_MCMC.py` to
this folder
(i.e. `cp ../../many_body_MCMC.py ./`). Then you can run the
examples like 

```
python many_body_MCMC.py  inputfile_two_boomerangs.dat
```

## Boomerang suspension
The input file `inputfile_boomerang_suspension.dat` can be used to run
a Monte-Carlo simulation of a boomerang colloidal suspension with
periodic boundary conditions, see Ref 1. The initial configuration of
the boomerangs is given in the clones file `boomerang_suspension_N_15.clones`.

We only provide pycuda implementation to compute the blob potentials,
therefore, it is necessary to have a GPU compatible with CUDA and
pycuda installed to run this example. 


## Blob-blob interactions
The pairwise blob-blob potential is

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
The blob-wall potential is

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

