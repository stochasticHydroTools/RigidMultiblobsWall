This example shows two 'multi blob' rollers which are translating in a hydrodynamically boud state. The hydrodynamic radius of the rollers is set to `R_h = 1`for simplicity and the resolution can be changed by changing line 64 in `inputfile_rollers.dat` from 
`structure	../../Structures/shell_N_12_Rg_0_7921_Rh_1.vertex two_rollers.clones`
to, say
`structure	../../Structures/shell_N_42_Rg_0_8913_Rh_1.vertex two_rollers.clones`

AND 

changing line 41 to the corresponding value for `blob_radius`. e.g changing

`blob_radius				                     0.41642068`
to
`blob_radius				                     0.2435531`
in the previous example. 

a stochastic or deterministic simulation can be done by swapping the comment from lines 9 and 10. 

To run a deterministic simulation using 12 blobs per roller, simply execute 

`python multi_bodies.py --input-file inputfile_rollers.dat`

The result of a simulation is sved in the directoy `./run_two_rollers` and can be visualized using the MATLAB file `config_plot.m`
