To run the code, ensure that you've successfully compiled both the `libMobility` directory as well as `DoublyPeriodicStokes` with double precision enabled.
In an ubuntu system the make commands in these directories may look like
```
DOUBLEPRECISION=-DDOUBLE_PRECISION make -B LAPACK_LIBS="-llapacke -lblas"
```

Given successfuly comilation of `libMobility` and `DoublyPeriodicStokes`, simply run `make` in this directory to compile the rigid body (`c_rigid_obj`) and 
fiber dynamics (`c_fiber_obj`) code.

To run an example that compares rotaing rigid rods in various domains run
```
python3 Rigid_Object_Main.py --input-file Rigid_Rods/rods_input.dat
```
where `Rigid_Object_Main.py` is a generic main file and the specific codes/files for the example are in the directory `./Rigid_Rods`. In `./Rigid_Rods`, the 
file `multi_bodies_functions.py` specifies the forces on torques on the rigid bodies, while `rods_input.dat`. The format of the imput file is largely similar 
to those used in the pythonic `../multi_bodies` bodies codes except for the following differences:
* If `blob_radius` is non-positive, then the code uses half of the minimum blob separation in the specified `.vertex` file.
* The parameter `domType` specifies the geometry of the domain, and if the domain is `DP...` a value of `zmax` (representing the height of the domain) must be specified.

In this example the `SC_config.clones` gives an initial configuration that can work in Slit channels (of width `zmax > 1`) and all other domains, while `DP_config.clones`
can only be used in larger domains due to the initial height specified. 

If one runs the example and sees many rejections (output including `Bad Midpoint!!` or `Bad Timestep!!`) then reduce the timestep size.
