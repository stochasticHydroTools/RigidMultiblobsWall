# Squirmer with Brownian motion
This example shows how to include slip files to simulate active
bodies. Here, we consider a squirmer, a spherical particle with a
tangential slip. In spherical coordinates the slip is given by the
expressions (see Ref. [2] in the documentation for details)

```
u_r = 0, 
u_phi = 0,
u_theta = sin(theta) 
```

## 1. Format of slip files

The file `squirmer.slip` contains the slip on the blobs on the
reference configuratin (quaternion = (1,0,0,0)). The format is

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

The name of the slip file has to be included in the inputfile in the line
`structure`, see `inputfile_squirmer.dat`. If no name is given the
code assumes that the slip is zero for that particular structure.

## 2. How to run the example

First, copy the file
`RigidMultiblobsWall/multi_bodies/multi_bodies.py` to this folder.
Then, to run the example use the command

```
python multi_bodies.py --input-file inputfile_squirmer.dat
``` 

Note that the squirmer is initially aligned with the x-axis but the  
Brownian motion change its orientation over time, therefore its trajectory is
not a simple straight line, see Figure \ref{fig:squirmer} where the
small dots represent the past locations of the blobs.  

![\label{fig:squirmer} Trajectory of a squirmer under Brownian
  fluctuations. The small dots represent the past locations of the blobs.](squirmer.png)


## 3. User defined slip functions

In some situations you may want to use a more complex slip, for
example, you may want to make the slip a function of the time or of the
distance to the wall. In those case you can write your own slip
function without modifying the main code, see a simple example in
`RigidMultiblobsWall/multi_bodies/examples/pair_active_rods/`.

