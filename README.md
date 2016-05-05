# RotationalDiffusion

Rotational and Translational Diffusion of Confined Rigid Bodies
by Steven Delong, Florencio Balboa, Blaise Delmotte and Aleksandar Donev (donev@courant.nyu.edu)
Courant Institute of Mathematical Sciences.

These mostly Python scripts implement the numerical algorithms described in the papers:
Donev: This is confusing even to me. Is this code still a mix of Steven's codes (only works for one body), and the new many-body codes? That is, can one actually do Brownian dynamics with this code (for a single body)? Are Steven's codes in a separate directory? Please explain what is actually implemented here.

1. **Brownian Dynamics of Confined Rigid Bodies**, S. Delong, F. Balboa Usabiaga, and A. Donev,
The Journal of Chemical Physics, **143**, 144107 (2015). 
[DOI](http://dx.doi.org/10.1063/1.4932062) [arXiv](http://arxiv.org/abs/1506.08868)

2. **Hydrodynamics of suspensions of passive and active rigid particles: a
  rigid multiblob approach** F. Balboa Usabiaga, B. Kallemov, B. Delmotte,
  A. Pal Singh Bhalla, B. E. Griffith, and A. Donev. [arXiv](http://arxiv.org/abs/1602.02170)

Donev: Adjust description below
This code consists of tools to compute hydrodynamic interactions among rigid multiblobs,
a quaternion integrator object used to simulate
Brownian trajectories of rigid bodies using quaternions and random-finite difference schemes, and
many small related scripts and tests.

Several example scripts for simulating immersed rigid bodies near a single
wall are present in subfolders.

For usage see **doc/USAGE.md**.
