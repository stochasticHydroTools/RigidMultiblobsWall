# Colloids with slip-boundary conditions
This example shows how to run a simulation with a particle with partial slip boundary conditions.
It simulates a colloid of geometric radius `Rg=1` falling under gravity.
The slip length in on the surface of the colloid is set to `slip_length`.

To run this example you need to:

1. Copy the file `quaternion_integrator_multi_bodies.py` to the folder `RigidMultiblobsWall/quaternion_integrator/`.

2. Run the code as 

```
python multi_bodies.py --input-file inputfile.dat
```


To modify the inputs you can edit the inputfile `inputfile.dat`.
Note that the details of about colloid are given in the line

```
structure   ../../Structures/shell_N_162_Rg_1_Rh_1_0530.vertex  ../../Structures/blob.clones ../../Structures/shell_N_162_Rg_1_slip_length_1e+03.slip_length
```

The first two files give the discretization of the colloid (`vertex` file)
and the configuration of the colloids (`clones` file) in this example just one.
The third file (`slip_length` file) contains the normals of the surface defined at the blobs, the slip length and the weights.

Finally, the utility code `create_slip_length_file.py` can be used to generated `slip_length` files for spherical colloids.


