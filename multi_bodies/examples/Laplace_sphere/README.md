# Spherical phoretic colloid
Example to simulate a spherical phoretic colloid.
To run this example you need to do:

1. Generate a `Laplace` file with the information about the colloidal surface normals, emitting rate, reaction rate, surface mobility and weights for the Laplace equation.
The `Laplace` file can be generated with the file `create_laplace_file_sphere.py`.

2. Copy the file `RigidMultiblobsWall/multi_bodies/multi_bodies.py` to this folder and run it like

```
python multi_bodies.py --input-file inputfile.dat
```


