# Run-and-tumble bacteria
Simple example of a run-and-tumble bacterium.
See the documentation

`RigidMultiblobsWall/doc/README.md`

and the example

`RigidMultiblobsWall/multi_bodies/examples/bacteria/`

to see how to model articulated bodies.
Here, we implemented a run-and-tumble bacterium.


# How to run
Edit the file

`RigidMultiblobsWall/quaternion_integrator/quaternion_integrator_multi_bodies.py`.

Add the following lines
```
# Check tumbling event
step = kwargs.get('step')     
for c in self.constraints:
  c.tumbling_event(step * dt)
```
to the function `articulated_deterministic_forward_euler` immediately before the lines
```
# Update links to current orientation
step = kwargs.get('step')
for c in self.constraints:
  c.update_links(time = step * dt)
```

Copy the file `RigidMultiblobsWall/multi_bodies/multi_bodies.py` to this folder.
Run the code as
```
python  multi_bodies.py  --input-file   inputfile.dat
```


# Modeling
You can find the tumbling implementation in the file `user_defined_functions.py` in this folder.
The implementation is quite simple and it does not try to reproduce any experimental results.
The parameters that define the tumbling are passed as extra constraint variables, see the file `Structures/bacteria.const`.
The extra parameters are:

1. `dt`: simulation time step size, it should match the value given in the inputfile (`inputfile.dat`).
2. `tau`: the inverse of the tumbling frequency, see below.
3. `tumbling_period`: duration of the tumbling event.
4. `max_angle`: maximum flagellum tilt angle during a tumbling event.
5. `l00`, `l01` and `100`: lengths of the links connecting the body and flagellum.


## Tumbling frequency
If there is no tumbling event active one can be randomly activated with  frequency `1 / tau`, i.e. it is activated if
```
np.random.rand(1) < 1 - np.exp(-dt / tau)
```
where `dt` is the time-step size.


## Tumbling event
During a tumbling event the flagella main axis tilts respect its equilibrium orientation an angle `max_angle` and it lasts `tumbling_period` time.

