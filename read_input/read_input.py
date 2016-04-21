'''
Simple class to read the input files to run a simulation.
'''

import numpy as np

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
    self.relaxation_steps = int(self.options.get('relaxation_steps') or 0)
    self.dt = float(self.options.get('dt') or 0.0)
    self.eta = float(self.options.get('eta') or 1.0)
    self.g = float(self.options.get('g') or 1.0)
    self.scheme = str(self.options.get('scheme') or 'deterministic_forward_euler')
    self.output_name = str(self.options.get('output_name') or 'run')
    
    
