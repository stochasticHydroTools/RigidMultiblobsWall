# Blobs of different radii
Simple example of a simulation with blobs of different radii.
We simulate two shells of hydrodynamic radius `Rh=1` and `Rh=0.5`.


In the input file we provide a default blob radius in the line

```
blob_radius                              0.416420683
```

Then we use two structures,

```
structure Structures/shell_N_12_Rg_0_7921_Rh_1.vertex shell_1.clones
structure Structures/shell_N_12_Rg_0.3960_Rh_0.5.vertex shell_2.clones
```

In the first one we do not specify the radius of the blobs and therefore the
code will use the default value. In the second structure we provide the radius
of each blob in the fourth column; examine the file `Structures/shell_N_12_Rg_0.3960_Rh_0.5.vertex`.

By running the code one time step it is possible to verify that the translational mobility of the small
shell is twice the one of the large shell as expected for Stokes flows. Run the code as

```
python multi_bodies.py --input-file inputfile.dat
```





