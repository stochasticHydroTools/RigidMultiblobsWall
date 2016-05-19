# RotationalDiffusion

Rotational and Translational Diffusion of Confined Rigid Bodies
by Steven Delong, Florencio Balboa, Blaise Delmotte and Aleksandar Donev (donev@courant.nyu.edu)
Courant Institute of Mathematical Sciences.

These mostly Python scripts implement the numerical algorithms described in the papers:
* **Brownian Dynamics of Confined Rigid Bodies**, S. Delong, F. Balboa Usabiaga, and A. Donev,
The Journal of Chemical Physics, **143**, 144107 (2015). 
[DOI](http://dx.doi.org/10.1063/1.4932062) [arXiv](http://arxiv.org/abs/1506.08868)
* **Hydrodynamics of suspensions of passive and active rigid particles: a
  rigid multiblob approach** F. Balboa Usabiaga, B. Kallemov, B. Delmotte,
  A. Pal Singh Bhalla, B. E. Griffith, and A. Donev. [arXiv](http://arxiv.org/abs/1602.02170)

This code consists of a quaternion integrator object used to simulate
Brownian trajectories of rigid bodies using quaternions, along with
many small related scripts and tests.  

Several example scripts for simulating immersed rigid bodies near a single
wall are present in subfolders.

For usage see **doc/USAGE.md** (or **doc/USAGE.pdf**).
