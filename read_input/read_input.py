'''
Simple class to read the input files to run a simulation.
'''
import numpy as np
import ntpath

class ReadInput(object):
  '''
  Simple class to read the input files to run a simulation.
  '''

  def __init__(self, entries):
    ''' Construnctor takes the name of the input file '''
    self.entries = entries
    self.input_file = entries
    self.options = {}

    # Read input file
    comment_symbols = ['#']   
    with open(self.input_file, 'r') as f:
      # Loop over lines
      for line in f:
        # Strip comments
        if comment_symbols[0] in line:
          line, comment = line.split(comment_symbols[0], 1)

        # Save options to dictionary, Value may be more than one word
        line = line.strip()
        if line != '':
          option, value = line.split(None, 1)
          self.options[option] = value

    # Set option to file or default values
    self.n_steps = int(self.options.get('n_steps') or 0)
    self.n_save = int(self.options.get('n_save') or 1)
    self.n_relaxation = int(self.options.get('n_relaxation') or 0)
    self.dt = float(self.options.get('dt') or 0.0)
    self.eta = float(self.options.get('eta') or 1.0)
    self.g = float(self.options.get('g') or 1.0)
    self.blob_radius = float(self.options.get('blob_radius') or 1.0)
    self.kT = float(self.options.get('kT') or 1.0)
    self.scheme = str(self.options.get('scheme') or 'deterministic_forward_euler')
    self.output_name = str(self.options.get('output_name') or 'run')
    self.structure_names = str.split(str(self.options.get('structure_names')))
    self.seed = self.options.get('seed')
    self.repulsion_strength_wall = float(self.options.get('repulsion_strength_wall') or 1.0)
    self.debey_length_wall = float(self.options.get('debey_length_wall') or 1.0)
    
    # Create structures ID for each kind (remove directory from structure name)
    self.structure_ID = []
    for name in self.structure_names:
      head, tail = ntpath.split(name)
      self.structure_ID.append(tail)
