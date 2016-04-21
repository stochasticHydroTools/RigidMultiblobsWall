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
    self.inputfile = entries[0]



