import numpy as np
import sys

if __name__ == '__main__':

  prefix = 'data/run_flagellated.step.'
  suffix = '.sphere_radius.16.p.32.velocity_field_sphere.dat'
  start = 0
  num_steps = 24
  step_size = 1
  x_avg = None

  N = 0
  for i in range(start,num_steps,step_size):
    name = prefix + str(i).zfill(8) + suffix
    print('name = ', name)
    N += 1

    x = np.loadtxt(name)
    if x_avg is None:
      x_avg = np.copy(x)
    else:
      x_avg[:,4:] += x[:,4:]

  print('N = ', N)
  x_avg[:,4:] /= N
  name = prefix + 'average' + suffix
  np.savetxt(name, x_avg)
  
  
