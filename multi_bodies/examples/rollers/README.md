# Rollers example

Input files to simulate active rollers moving above a wall in the
presence of thermal fluctuations. For details see reference

1. **Brownian dynamics of condined suspensions of active microrollers**, F. Balboa Usabiaga, B. Delmotte and A. Donev,
The Journal of Chemical Physics, **146**, 134104 (2017). [arXiv](https://arxiv.org/abs/1612.00474)
[DOI](http://dx.doi.org/10.1063/1.4979494)

To use this example is necessary to have pyCuda installed and a GPU
compatible with CUDA. To run this example move to the folder
`RigidMultiblobsWall/multi_bodies/examples/rollers/`
and copy the code `RigidMultiblobsWall/multi_bodies/multi_bodies.py`
to this folder like

```
cp ../../multi_bodies.py ./
```

You can run the simulation with the command

```
python multi_bodies.py --input-file inputfile_rollers.dat
```

See options for the simulations in the input file
`inputfile_rollers.dat`.

