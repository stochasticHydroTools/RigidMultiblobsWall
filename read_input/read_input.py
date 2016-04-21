'''
Simple class to read the input files to run a simulation.
'''

import numpy as np

class ReadInput(object):
  '''
  Simple class to read the input files to run a simulation.
  '''

  def __init__(self, entries):
    ''' Constructor, takes 4 entries = s, p1, p2, p3 as a numpy array. '''
    self.entries = entries


