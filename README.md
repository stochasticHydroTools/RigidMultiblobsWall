# RotationalDiffusion

Rotational and Translational Diffusion of Confined Rigid Bodies
by Steven Delong, Florencio Balboa, Blaise Delmotte and Aleksandar Donev (donev@courant.nyu.edu)
Courant Institute of Mathematical Sciences.

This package contains several python codes to run simulations of 
rigid bodies made out of _blob particles_ rigidly connected near
a single wall (floor). These codes can compute the
mobility of complex shape objects, solve mobility or resistance problems
for suspensions of many bodies or run deterministic or stochastic 
dynamic simulations.

For the theory behind the numerical methods consult the references:

1. **Brownian Dynamics of Confined Rigid Bodies**, S. Delong, F. Balboa Usabiaga, and A. Donev,
The Journal of Chemical Physics, **143**, 144107 (2015). 
[DOI](http://dx.doi.org/10.1063/1.4932062) [arXiv](http://arxiv.org/abs/1506.08868)

2. **Hydrodynamics of suspensions of passive and active rigid particles: a
  rigid multiblob approach** F. Balboa Usabiaga, B. Kallemov, B. Delmotte,
  A. Pal Singh Bhalla, B. E. Griffith, and A. Donev, submitted to CAMCoS. [arXiv](http://arxiv.org/abs/1602.02170)

Several example scripts for simulating immersed rigid bodies near a single
wall are present in subfolders.

For usage see **doc/USAGE.md** (or **doc/USAGE.pdf**).

### Software organization
* **body/**: it contains a class to handle a single rigid body.
* **boomerang/**: stochastic example, see documentation `doc/boomerang.txt`.
* **doc/**: documentation.
* **mobility/**: it has functions to compute the blob mobility matrix **M** and the
product **Mf**.
* **multi_bodies/**: codes to run simulations of rigid bodies.
* **quaternion_integrator/**: it has a small class to handle quaternions and
the schemes to integrate the equations of motion.
* **sphere/**: the folder contains an example to simulate a sphere
whose center of mass is displaced from the geometric center
(i.e., gravity generates a torque), sedimented near a no-slip wall
in the presence of gravity, as described in Section IV.C in [1](http://dx.doi.org/10.1063/1.4932062).
Unlike the boomerang example this code does not use a rigid
multiblob model of the sphere but rather uses the best known
(semi)analytical approximations to the sphere mobility.
See documentation `doc/boomerang.txt`.
* **stochastic_forcing/**: it contains functions to compute the product
 **M**^{1/2}**z** necessary to perform Brownian simulations.
* **utils.py**: this file has some general functions that would be useful for
general rigid bodies (mostly for analyzing and reading trajectory
data and for logging).

