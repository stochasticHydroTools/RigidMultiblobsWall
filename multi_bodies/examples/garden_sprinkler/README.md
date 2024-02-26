# Chiral phoretic microrotors
Example to simulate chiral phoretic microrotors with sprinkle shapes as in Ref.

To run this example do:

1. Run the utility code `create_sprinkler_vertex_file.py` to generate the files that describe a sprinkle shape colloid.
It generates two files, a `.vertex` file with the position of the blobs/nodes discretizing the colloids and
a `.Laplace` file with the surface normals, reaction rate, emitting rate, surface mobility, weights to solve the Laplace equation.
The geometric parameters of the colloid can be changed up the top of the `main` function.

2. Copy the code `RigidMultiblobsWall/multi_bodies/multi_bodies.py` to this folder and run it like

```
python multi_bodies.py --input-file inputfile_sprinkler.dat
```

