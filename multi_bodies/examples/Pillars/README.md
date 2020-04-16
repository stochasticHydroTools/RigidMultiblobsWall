# Free bodies and obstacles
This example shows how to simulate an active rod and an obstacle.

### Active rod
In the input file (`inputfile.dat`) the active rod is defined in the line

```
structure ../../Structures/rod_Lg_1.845_Rg_0.1308_Nx_16_Ntheta_6.vertex rod_resolved.clones rod_resolved.slip
```

As usual the `vertex` and `clones` files give the location of the blobs in the reference configuration and the initial configuration respectively.

The file `rod_resolved.slip` contains the slip on the blobs on the
reference configuration (quaternion = (1,0,0,0)). The format is

```
number_of_blobs_in_rigid_body
vector_slip_blob_0
vector_slip_blob_1
vector_slip_blob_2
.
.
.
```

This slip is rotated with the orientation of the body every step.

### Obstacles
Obstacles are rigid particles that do not move. The code solves a resistance problem to find the constraint force and torque that keep the particle fixed. See `latex` document notesMixedKinematics.tex (it can be compiled to a `pdf`).

In this example the only obstacle is pillar centered around `(x,y)=(0,0)`. The files describing the obstacle are given in the line

```
obstacle  pillar_R_2_h_2_Ntheta_96_Nz_17_caps_4.vertex pillar.clones 
```

