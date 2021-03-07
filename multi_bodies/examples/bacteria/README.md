# Flagellated bacteria
Example to simulated a flagellated bacteria with the rigid articulated scheme.

## 1. Format of list_vertex files
The files `*.list_vertex` contain a list of all the vertex files forming an articulated rigid body.
It should have one vertex file per line. Our model of flagellated bacteria is formed by two rigid
bodies to represent the spherical body of the bacterium and the flagellum. The file looks like
```
../../Structures/shell_N_162_Rg_0_9497_Rh_1.vertex
../../Structures/flagellum_L_10_alpha_0.35.vertex
```

## 2. Format of clones files
As explained in the documentation (`doc/README.md`) these files give the initial configuration
of the rigid bodies. For a single bacteria the file should give the configuration of the
bacteria's body and flagellum, inspect the file `multi_bodies/Structures/bacteria.clones`.
To simulate, for example, two bacteria it is necessary to give the position of four rigid bodies like this
```
4
0.0 0.0 3.0 1.0 0.0 0.0 0.0
0.0 0.0 4.262 1.0 0.0 0.0 0.0
10.0 0.0 3.0 1.0 0.0 0.0 0.0
10.0 0.0 4.262 1.0 0.0 0.0 0.0
```

## 3. Format of const files
The constraints connecting rigid bodies to form an articulated rigid body are defined in the `const` files.
We provide two examples: one with two time independent constraints `multi_bodies/Structures/bacteria_passive.const`,
and one with an additional time dependent constraint `multi_bodies/Structures/bacteria_active.const`.

## 4. User defined functions, applying a constant torque
You can override many functions defined in the code by writing your own
implementation in a `user_defined_functions.py` as done in this example.

Here we modify the body-body interactions to apply a constant torque
(in the body frame of reference) when the name of the articulated body
is `bacteria_constant_torque`. This is an approach to simulate active
particles using only time independent constraints.

The value of the torque is given in the input file in the line `omega_one_roller`.
You can edit the file `bacteria_constant_torque` to apply more complex torques,
for example time dependent ones.

## 5. How to run the example
First, copy the file
`RigidMultiblobsWall/multi_bodies/multi_bodies.py` to this folder.
Then, to run the example with a constant angular velocity use the command

```
python multi_bodies.py --input-file inputfile_bacteria_constant_angular_velocity.dat
``` 

To run the example with a constant torque
use the command

```
python multi_bodies.py --input-file inputfile_bacteria_constant_torque.dat
``` 

