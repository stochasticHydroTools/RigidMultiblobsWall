## Projects of [Zhe Chen](chenzhesms@pku.edu.cn)


Assume that you have git cloned RigitMultiblobsWall and HydroGrid and you can go into steps below.

### 1. Preparation

Firtly, please set directory of RigitMultiblobsWall and HydroGrid in the RigidMultiblobsWall\multi_bodies\multi_bodies.py.

Recommended setting is:

sys.path.append('../../RigidMultiblobsWall/')

sys.path.append('../')

Also, compile HydroGrid to  get the calculateConcentration.so(Notice should be paid to that directory of py is different in different system(Redhat ot CentOS). You should change it in the MakefileHeader). And add its directory to RigidMultiblobsWall\multi_bodies\multi_bodies.py too:

sys.path.append('../../HydroGrid/src/')

### 2. Run

This folder is for Zhe's project and you can see 'Summary.pdf' to see a profile of it. There're several sub-folders here and each folder is a project with README and report in it. Please go to the project you want to see to get the details.

#### a). ProbDist: 
##### Probability Distribution parallel or verticle to the wall with or without Hydrodynamics Interaction(HI)

#### b). BrownianDynamics:
##### Simulation of free Brownian walkers.

