# Chiral phoretic microrotors
Example to simulate chiral phoretic microrotors with sprinkle shapes as in Ref.

To run this example do:

1. Run the utility code `create_sprinkler_vertex_file.py` to generate the vertex file of sprinkle shape colloid.
The geometric parameters of the colloid can be changed up the top of the `main` function.

2. Copy the code `RigidMultiblobsWall/multi_bodies/multi_bodies.py` to this folder and run it like

```
python multi_bodies.py --input-file inputfile_sprinkler.dat
```

