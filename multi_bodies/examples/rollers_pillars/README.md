# Free bodies and obstacles
This example shows how to simulate microrollers and an obstacle.

### Microrollers
In the input file (`inputfile.dat`) the 2000 microrollers are initialized on a lattice 1.2 radii above the floor. They are  defined in the line

```
structure Structures/blob.vertex Structures/Lattice_blobs_Np_2000_a_0_19134_Z_1_2a_dx_dy_2_2_2_2a.clones
```

As usual the `vertex` and `clones` files give the location of the blobs in the reference configuration and the initial configuration respectively.


### Obstacles
Obstacles are rigid particles that do not move. The code solves a resistance problem to find the constraint force and torque that keep the particle fixed.

In this example the obstacles are 7 pillars centered around `(x,y)=(0,0)`. The files describing the obstacles are given in the line

```
structure Structures/pillar_R_0_5_h_4_64_Ntheta_8_Nz_15.vertex Structures/pillars.clones Structures/slip_zero_N1536.dat Structures/velocities_zero.d    at

```

