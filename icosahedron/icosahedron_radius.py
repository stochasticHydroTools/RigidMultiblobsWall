''' 
Script to study the effective hydrodynamic radius of the icosohedron.
These are compared to the table II in:

 "Adolfo Vazquez-Quesada, Florencio
Balboa Usabiaga, and Rafael Delgado-Buscalioni - A multiblob approach
to colloidal hydrodynamics with inherent lubrication."  

We expect some differences because the icosohedron code here does not
discretize space.
'''

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np

import icosahedron as ic
from quaternion_integrator.quaternion import Quaternion

# All of these quantities are in units of h, their meshwidth.
# The radius of a blob is a = 0.91*h
PAPER_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0]
PAPER_TRANSLATION = [1.76, 1.99, 2.35, 2.73, 3.08]
PAPER_ROTATION = [1.5, 1.9, 2.36, 2.64, 3.30]

def plot_icosohedron_ratios(a, icosohedron_ratios):
  ''' 
  Plot the icosohedron ratios from the paper, and compare to
  those given by this code.
  '''
  far_location = [[0., 0., 30000.*a]]
  orientation = [Quaternion([1., 0., 0., 0.])]
  radii_ratios = []
  translation_radii = []
  rotation_radii = []
  for r in icosohedron_ratios:
    ic.VERTEX_A = a
    # Account for meshwidth -> radius with 0.91, and distance between 
    # vertices -> distance to center with 1.05.
    ic.A = r*a/(0.91)/(1.05)
    mobility_far = ic.icosohedron_mobility(far_location, orientation)
    a_translation = 1./(6.*np.pi*ic.ETA*mobility_far[0, 0])
    a_rotation = (1./(mobility_far[3, 3]*8.*np.pi*ic.ETA))**(1./3.)
    radii_ratios.append(a_translation/a_rotation)
    translation_radii.append(a_translation/(a/0.91))
    rotation_radii.append(a_rotation/(a/0.91))

  pyplot.figure(1)
  pyplot.plot(icosohedron_ratios, radii_ratios, 'b-', label='Python Code')
  pyplot.plot(PAPER_RATIOS, np.array(PAPER_TRANSLATION)/
              np.array(PAPER_ROTATION), 'k-', label='Paper')
  pyplot.plot([0.9, 3.6], np.ones(2), 'r--', label='Ratio = 1')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('Ratio of Hydrodynamic Radii')
  pyplot.ylabel('Translation Radius / Rotation Radius')
  pyplot.xlabel('d/h (from paper)')
  pyplot.xlim([0.9, 3.6])
  pyplot.savefig('./figures/IcosohedronRadiiRatio.pdf')

  pyplot.figure(2)
  pyplot.plot(icosohedron_ratios, translation_radii, 'b-', label='Python Code')
  pyplot.plot(PAPER_RATIOS, PAPER_TRANSLATION , 'k-', label='Paper')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('Translation Hydrodynamic Radius')
  pyplot.ylabel('Translation Radius (in units of h)')
  pyplot.xlabel('d/h (from paper)')
  pyplot.xlim([0.9, 3.6])
  pyplot.savefig('./figures/IcosohedronRadiiTranlsation.pdf')

  pyplot.figure(3)
  pyplot.plot(icosohedron_ratios, rotation_radii, 'b-', label='Python Code')
  pyplot.plot(PAPER_RATIOS, PAPER_ROTATION , 'k-', label='Paper')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('Rotation Hydrodynamic Radius')
  pyplot.ylabel('Rotation Radius (in units of h)')
  pyplot.xlabel('d/h (from paper)')
  pyplot.xlim([0.9, 3.6])
  pyplot.savefig('./figures/IcosohedronRadiiRotation.pdf')


if __name__ == '__main__':
  
  a = 0.175
  code_ratios = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 
                 2.25, 2.5, 2.75, 3.0, 3.2, 3.5]
  plot_icosohedron_ratios(a, code_ratios)
  
  

  
    
    
