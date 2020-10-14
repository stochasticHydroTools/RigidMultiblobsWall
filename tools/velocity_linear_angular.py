'''
Compute linear and angular velocities between frames n and m (n > m).
Linear velocity (correct to order dt^2).

v^m = (x^n - x^m) / ((n-m)*dt)

Angular velocity (correct to order dt^2).
w^m = (Psi^T * Psi)^{-1}*Psi^T theta^n / ((n - m)*dt)

where Psi=Psi^m, theta is the quaternion and
we use the fact Psi^T * Psi = 0.25 * I.
'''

import numpy as np
import sys


if __name__ == '__main__':
  name = sys.argv[1]
  dt = float(sys.argv[2])
  frame_rate = int(sys.argv[3])

  # Read file and set parameters
  x = np.loadtxt(name)
  dt_frames = dt * frame_rate

  if x.size <= 8 * frame_rate:
    v = np.zeros((1,3))
  else:
    # v = (x[frame_rate:,1:4] - x[0:-frame_rate,1:4]) / (frame_rate * dt)
    for m in range(x.size // 8 - frame_rate):
      n = m + frame_rate
      v = (x[n, 1:4] - x[m, 1:4]) / dt_frames
      q_m = np.array(x[m, 4:])
      q_n = np.array(x[n, 4:])
      Psi = 0.5 * np.array([[-q_m[1], -q_m[2], -q_m[3]],
                            [ q_m[0],  q_m[3], -q_m[2]],
                            [-q_m[3],  q_m[0],  q_m[1]],
                            [ q_m[2], -q_m[1],  q_m[0]]])
      w = 4.0 * np.dot(Psi.T, q_n) / dt_frames
    
      # Print velocities
      v_w = np.reshape(np.concatenate([np.array([dt*m]), v,w]), (1,7))
      np.savetxt(sys.stdout, v_w, delimiter=' ')

