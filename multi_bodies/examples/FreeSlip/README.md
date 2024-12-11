# Colloids with slip-boundary conditions
To run this example you need to:

1. Copy the file `quaternion_integrator_multi_bodies.py` to the folder `RigidMultiblobsWall/quaternion_integrator/`.

2. Create a `slip_length` with the code `create_slip_length_file.py`.
   The file contains the normals of the colloid surface, the slip lengths and the weights.

3. Run the code as 

```
python multi_bodies.py --input-file inputfile.dat
```


