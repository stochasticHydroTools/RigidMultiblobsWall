import argparse
import numpy as np
import sys
sys.path.append('../')

from mobility import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator_gmres import QuaternionIntegratorGMRES
from body import body 
from read_input import read_input




if __name__ == '__main__':
  # Get command line arguments
  parser = argparse.ArgumentParser(description='Run a multi-body simulation '
                                   'with a deterministic forward Euler '
                                   'scheme and save trajectory.')
  parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', 
                      help='name of the input file')
  args=parser.parse_args()
  input_file = args.input_file

  # Read input file
  read = read_input.ReadInput(input_file)
  
  # Set some variables for the simulation
  n_steps = read.n_steps 
  relaxation_steps = read.relaxation_steps
  dt = read.dt 
  eta = read.eta 
  g = read.g 
  scheme  = read.scheme 
  output_name = read.output_name 
  structure_names = read.structure_names
  
  # Create rigid bodies
  for structure in structure_names:
    print 's', structure
  


  print '# End'
