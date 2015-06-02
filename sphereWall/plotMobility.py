''' 
Script to calculate the average parallel, perpendicular, and
rotational mobilities for a sphere discretized with a different
number of blobs and make a plot.
'''

from matplotlib import pyplot
import numpy as np
import sys

sys.path.append('..')

from sphere162 import sphere162 as sph162
from sphere42 import sphere42 as sph42
from icosahedron import icosahedron as ic
import sphere.sphere as sph
from quaternion_integrator.quaternion import Quaternion
#Add theory

# NOTE: Floren, I changed this. I don't know
# if your codes in /sibm are different?
sys.path.append('../mobilities')
import selfMobilitySwanBradyWithStresslet
import selfMobilitySwanBrady
import selfMobilityBrenner
import selfMobilityFauxcheux
import selfMobilityHuang
import selfMobilityGoldman
       
def plot_mobilities(a, asph42, asph162, heights):
  '''
  Calculate parallel, perpendicular, and rotational mobilities
  at the given heights for an sphere discretized with blobs of radius a.
  Here we vary the distance between the sphere and the wall.
  '''
  orientation = [Quaternion([1., 0., 0., 0.])]
  far_location = [[0., 0., 30000.]]
  sphere_mobility_theory = sph.sphere_mobility(far_location, orientation)
  d=1.0 #Radius of the sphere
  ic.VERTEX_A = a
  ic.A = d
  sph42.VERTEX_A = asph42
  sph42.A = d
  sph162.VERTEX_A = asph162
  sph162.A = d
  x = []
  # Compute theoretical mobility for icosahedron
  location = [[0., 0., 30000.]]
  orientation = [Quaternion([1., 0., 0., 0.])]
  mobility_theory = ic.icosahedron_mobility(location, orientation)
  a_eff = 1.0/(6.*np.pi*ic.ETA*mobility_theory[0, 0])
  a_rot_eff = (1./(mobility_theory[3, 3]*8.*np.pi*ic.ETA))**(1./3.)
  print 'Radius far from the wall, sphere 12-blobs'
  print 'a_effective for d = %f is %f' % (d, a_eff)
  print 'a_effective Rotation for d = %f is %f' % (d, a_rot_eff)
  print 'ratio for d = %f is %f' %(d, a_eff/a_rot_eff)
  print ''
  # Compute theoretical mobility for sphere42
  location = [[0., 0., 30000.]]
  orientation = [Quaternion([1., 0., 0., 0.])]
  mobility_theory_sph42 = sph42.sphere_42_blobs_mobility(location, orientation)
  a_eff_sph42 = 1.0/(6.*np.pi*ic.ETA*mobility_theory_sph42[0, 0])
  a_rot_eff_sph42 = (1./(mobility_theory_sph42[3, 3]*8.*np.pi*ic.ETA))**(1./3.)
  print 'Radius far from the wall, sphere 42-blobs'
  print 'a_effective for d = %f is %f' % (d, a_eff_sph42)
  print 'a_effective Rotation for d = %f is %f' % (d, a_rot_eff_sph42)
  print 'ratio for d = %f is %f' %(d, a_eff_sph42/a_rot_eff_sph42)
  print ''

  # Compute theoretical mobility for sphere162
  location = [[0., 0., 30000.]]
  orientation = [Quaternion([1., 0., 0., 0.])]
  mobility_theory_sph162 = sph162.sphere_162_blobs_mobility(location, orientation)
  a_eff_sph162 = 1.0/(6.*np.pi*ic.ETA*mobility_theory_sph162[0, 0])
  a_rot_eff_sph162 = (1./(mobility_theory_sph162[3, 3]*8.*np.pi*ic.ETA))**(1./3.)
  print 'Radius far from the wall, sphere 162-blobs'
  print 'a_effective for d = %f is %f' % (d, a_eff_sph162)
  print 'a_effective Rotation for d = %f is %f' % (d, a_rot_eff_sph162)
  print 'ratio for d = %f is %f' %(d, a_eff_sph162/a_rot_eff_sph162)
  print ''


  #Mobility for the 1-blob sphere
  sphere_parallel = []
  sphere_perp  = []
  sphere_rotation = []
  sphere_rotation_perp = []
  sphere_40 = []
  x_sphere = []
  for h in np.array(heights)*sph.A:
    #if((h-sph.A) >= 0):
    if((h-d) >= 0.05):
      location = [[0., 0., h]]
      sphere_mobility = sph.sphere_mobility(location, orientation)
      sphere_parallel.append(sphere_mobility[0, 0]*6.*np.pi*1.0*a)
      sphere_perp.append(sphere_mobility[2, 2]*6.*np.pi*sph.ETA*sph.A)
      sphere_rotation.append(sphere_mobility[3, 3] / sphere_mobility_theory[3, 3])
      sphere_rotation_perp.append(sphere_mobility[5, 5] / sphere_mobility_theory[5, 5])
      sphere_40.append( -sphere_mobility[4,0] * 6.0 * np.pi * a**2 ) #We cannot normalize here * 6 * np.pi * a*a
      x_sphere.append(h/a)
    

  pyplot.figure(1)
  #pyplot.plot(x_sphere, sphere_parallel, 'm-', label='1-blob')
  pyplot.figure(2)
  #pyplot.plot(np.array(heights), sphere_perp, 'k--', label='1-blob')
  pyplot.figure(3)
  #pyplot.plot(np.array(heights), sphere_rotation, 'k--', label='1-blob parallel')
  #pyplot.plot(np.array(heights), sphere_rotation_perp, 'k-', label='1-blob perp')
  pyplot.figure(4)
  #pyplot.plot(x_sphere, sphere_40, 'm-', label='1-blob')

  #Mobility for the 12-blob sphere
  average_x = []
  average_mu_parallel = []
  std_mu_parallel = []
  average_mu_perp = []
  std_mu_perp = []
  average_mu_rotation_parallel = []
  std_mu_rotation_parallel = []
  average_mu_rotation_perp = []
  std_mu_rotation_perp = []
  average_mu_40 = []
  std_mu_40 = []  
  x_sphere_12_blobs = []
  
  for r in heights:
    #if((r-a_eff) >= 0):
    if((r-d) >= 0.0):
      # Calculate 2 random orientations for heights.
      h = r*a_eff
      x_r = []
      mu_parallel_r = []
      mu_perp_r = []
      mu_rotation_parallel_r = []
      mu_rotation_perp_r = []
      mu_40_r = []
      x_sphere_12_blobs.append(r)
      count = 0
      for k in range(20):
        count += 1.0
        theta = np.random.normal(0., 1., 4)
        theta = Quaternion(theta/np.linalg.norm(theta))
        location = [0., 0., h]
        mobility = ic.icosahedron_mobility([location], [theta])
        
        #Do not scatter
        x_r.append(h/a_eff)
        mu_parallel_r.append(mobility[0, 0] / mobility_theory[0, 0])
        mu_perp_r.append(mobility[2, 2] / mobility_theory[2, 2])
        mu_rotation_parallel_r.append(mobility[3, 3] / mobility_theory[3, 3])
        mu_rotation_perp_r.append(mobility[5, 5] / mobility_theory[5, 5])
        mu_40_r.append(mobility[4, 0] * 6 * np.pi * a_eff**2)

      #Average values for a fixed distance and compute standard deviations
      average_x.append(np.mean(x_r))
      average_mu_parallel.append(np.mean(mu_parallel_r, axis=0))
      std_mu_parallel.append( np.std(mu_parallel_r, axis=0) )  #We want the standard deviation do not divide by np.sqrt(count)
      average_mu_perp.append(np.mean(mu_perp_r, axis=0))
      std_mu_perp.append( np.std( mu_perp_r, axis=0) )
      average_mu_rotation_parallel.append(np.mean(mu_rotation_parallel_r, axis=0))
      std_mu_rotation_parallel.append( np.std(mu_rotation_parallel_r, axis=0))
      average_mu_rotation_perp.append(np.mean(mu_rotation_perp_r, axis=0))
      std_mu_rotation_perp.append(np.std( mu_rotation_perp_r, axis=0) )
      average_mu_40.append(np.mean(mu_40_r, axis=0))
      std_mu_40.append( np.std(mu_40_r, axis=0) ) 


  #Scale standard deviation to plot error bars with 2 std deviations
  std_mu_parallel[:]            = [xx * 2 for xx in std_mu_parallel]
  std_mu_perp[:]                = [xx * 2 for xx in std_mu_perp]
  std_mu_rotation_parallel[:]   = [xx * 2 for xx in std_mu_rotation_parallel]
  std_mu_rotation_perp[:]       = [xx * 2 for xx in std_mu_rotation_perp]
  std_mu_40[:]                  = [xx * 2 for xx in std_mu_40]

  pyplot.figure(1)
  pyplot.errorbar(x_sphere_12_blobs, average_mu_parallel, yerr=std_mu_parallel, fmt='o', color='k', label='12-blobs', markersize=8)
  pyplot.figure(2)
  pyplot.errorbar(x_sphere_12_blobs, average_mu_perp, yerr=std_mu_perp, fmt='o', color='k', label='12-blobs', markersize=8)
  pyplot.figure(3)
  pyplot.errorbar(x_sphere_12_blobs, average_mu_rotation_parallel, yerr=std_mu_rotation_parallel, fmt='o', color='k', 
                  label='12-blobs parallel', markersize=8)
  pyplot.errorbar(x_sphere_12_blobs, average_mu_rotation_perp, yerr=std_mu_rotation_perp, fmt='^', color='b', 
                  label='12-blobs perp', markersize=8)
  pyplot.figure(4)
  pyplot.errorbar(x_sphere_12_blobs, average_mu_40, yerr=std_mu_40, fmt='o', color='k', label='12-blobs', markersize=8)

  file = open('mobility.12-blob.dat', 'w')
  line = '#columns: distance, mu_parallel, mu_perpendicular, mu_rotation_parallel, mu_rotation_perpendicular, mu_rotation_translation_40 \n'
  file.write(line)
  count=0
  for x in x_sphere_12_blobs:
    mu_parallel          = average_mu_parallel[count]
    mu_perp              = average_mu_perp[count]
    mu_rotation_parallel = average_mu_rotation_parallel[count]
    mu_rotation_perp     = average_mu_rotation_perp[count]
    mu_40                = average_mu_40[count]
    line = str(x) + ' ' + str(mu_parallel) + ' ' + str(mu_perp) + ' ' + str(mu_rotation_parallel) + ' ' + str(mu_rotation_perp) + ' ' + str(mu_40) + '\n'
    file.write(line)
    count += 1
  file.close()


  #Mobility for the 42-blob sphere
  average_x = []
  average_mu_parallel = []
  std_mu_parallel = []
  average_mu_perp = []
  std_mu_perp = []
  average_mu_rotation_parallel = []
  std_mu_rotation_parallel = []
  average_mu_rotation_perp = []
  std_mu_rotation_perp = []
  average_mu_40 = []
  std_mu_40 = []
  x_sphere_42_blobs = []

  for r in heights:
    if((r-d) >= 0.0):
      x_sphere_42_blobs.append(r)
      # Calculate 2 random orientations for heights.
      h = r*a_eff_sph42
      x_r = []
      mu_parallel_r = []
      mu_perp_r = []
      mu_rotation_parallel_r = []
      mu_rotation_perp_r = []
      mu_40_r = []
      count = 0
      for k in range(20):
        count += 1.0
        theta = np.random.normal(0., 1., 4)
        theta = Quaternion(theta/np.linalg.norm(theta))
        location = [0., 0., h]
        mobility = sph42.sphere_42_blobs_mobility([location], [theta])
        
        #Do not scatter
        x_r.append(h/a_eff_sph42)
        mu_parallel_r.append(mobility[0, 0] / mobility_theory_sph42[0, 0])
        mu_perp_r.append(mobility[2, 2] / mobility_theory_sph42[2, 2])
        mu_rotation_parallel_r.append(mobility[3, 3] / mobility_theory_sph42[3, 3])
        mu_rotation_perp_r.append(mobility[5, 5] / mobility_theory_sph42[5, 5])
        mu_40_r.append( mobility[4, 0] * 6 * np.pi * a_eff_sph42**2 )

      #Average values for a fixed distance and compute standard deviations
      average_x.append(np.mean(x_r))
      average_mu_parallel.append(np.mean(mu_parallel_r, axis=0))
      std_mu_parallel.append( np.std(mu_parallel_r, axis=0) )
      average_mu_perp.append(np.mean(mu_perp_r, axis=0))
      std_mu_perp.append(np.std( mu_perp_r, axis=0) )  
      average_mu_rotation_parallel.append(np.mean(mu_rotation_parallel_r, axis=0))
      std_mu_rotation_parallel.append( np.std(mu_rotation_parallel_r, axis=0) )
      average_mu_rotation_perp.append(np.mean(mu_rotation_perp_r, axis=0))
      std_mu_rotation_perp.append( np.std(mu_rotation_perp_r, axis=0) )
      average_mu_40.append(np.mean(mu_40_r, axis=0))
      std_mu_40.append( np.std(mu_40_r, axis=0) )


  #Scale standard deviation to plot error bars with 2 std deviations
  std_mu_parallel[:]            = [xx * 2 for xx in std_mu_parallel]
  std_mu_perp[:]                = [xx * 2 for xx in std_mu_perp]
  std_mu_rotation_parallel[:]   = [xx * 2 for xx in std_mu_rotation_parallel]
  std_mu_rotation_perp[:]       = [xx * 2 for xx in std_mu_rotation_perp]
  std_mu_40[:]                  = [xx * 2 for xx in std_mu_40]

  pyplot.figure(1)
  pyplot.errorbar(x_sphere_42_blobs, average_mu_parallel, yerr=std_mu_parallel, fmt='s', color='r', label='42-blobs', markersize=6)
  pyplot.figure(2)
  pyplot.errorbar(x_sphere_42_blobs, average_mu_perp, yerr=std_mu_perp, fmt='s', color='r', label='42-blobs', markersize=6)
  pyplot.figure(3)
  pyplot.errorbar(x_sphere_42_blobs, average_mu_rotation_parallel, yerr=std_mu_rotation_parallel, fmt='s', color='r', label='42-blobs parallel', markersize=6)
  pyplot.errorbar(x_sphere_42_blobs, average_mu_rotation_perp, yerr=std_mu_rotation_perp, fmt='v', color='m', label='42-blobs perp',  markersize=6)
  pyplot.figure(4)
  pyplot.errorbar(x_sphere_42_blobs, average_mu_40, yerr=std_mu_40, fmt='s', color='r', label='42-blobs', markersize=6)


  file = open('mobility.42-blob.dat', 'w')
  line = '#columns: distance, mu_parallel, mu_perpendicular, mu_rotation_parallel, mu_rotation_perpendicular, mu_rotation_translation_40 \n'
  file.write(line)
  count=0
  for x in x_sphere_42_blobs:
    mu_parallel          = average_mu_parallel[count]
    mu_perp              = average_mu_perp[count]
    mu_rotation_parallel = average_mu_rotation_parallel[count]
    mu_rotation_perp     = average_mu_rotation_perp[count]
    mu_40                = average_mu_40[count]
    line = str(x) + ' ' + str(mu_parallel) + ' ' + str(mu_perp) + ' ' + str(mu_rotation_parallel) + ' ' + str(mu_rotation_perp) + ' ' + str(mu_40) + '\n'
    file.write(line)
    count += 1
  file.close()








  #Mobility for the 162-blob sphere
  average_x = []
  average_mu_parallel = []
  std_mu_parallel = []
  average_mu_perp = []
  std_mu_perp = []
  average_mu_rotation_parallel = []
  std_mu_rotation_parallel = []
  average_mu_rotation_perp = []
  std_mu_rotation_perp = []
  average_mu_40 = []
  std_mu_40 = []
  x_sphere_162_blobs = []

  for r in heights:
    #if((r-a_eff_sph162) >= 0):
    if((r-d) >= 0.0):
      x_sphere_162_blobs.append(r)
      # Calculate 2 random orientations for heights.
      h = r*a_eff_sph162
      x_r = []
      mu_parallel_r = []
      mu_perp_r = []
      mu_rotation_parallel_r = []
      mu_rotation_perp_r = []
      mu_40_r = []
      count = 0
      for k in range(20):
        count += 1.0
        theta = np.random.normal(0., 1., 4)
        theta = Quaternion(theta/np.linalg.norm(theta))
        location = [0., 0., h]
        mobility = sph162.sphere_162_blobs_mobility([location], [theta])
        
        #Do not scatter
        x_r.append(h/a_eff_sph162)
        mu_parallel_r.append(mobility[0, 0] / mobility_theory_sph162[0, 0])
        mu_perp_r.append(mobility[2, 2] / mobility_theory_sph162[2, 2])
        mu_rotation_parallel_r.append(mobility[3, 3] / mobility_theory_sph162[3, 3])
        mu_rotation_perp_r.append(mobility[5, 5] / mobility_theory_sph162[5, 5])
        mu_40_r.append( mobility[4, 0] * 6 * np.pi * a_eff_sph162**2 )

      #Average values for a fixed distance and compute standard deviations
      average_x.append(np.mean(x_r))
      average_mu_parallel.append(np.mean(mu_parallel_r, axis=0))
      std_mu_parallel.append( np.std(mu_parallel_r, axis=0) )
      average_mu_perp.append(np.mean(mu_perp_r, axis=0))
      std_mu_perp.append(np.std( mu_perp_r, axis=0) )  
      average_mu_rotation_parallel.append(np.mean(mu_rotation_parallel_r, axis=0))
      std_mu_rotation_parallel.append( np.std(mu_rotation_parallel_r, axis=0) )
      average_mu_rotation_perp.append(np.mean(mu_rotation_perp_r, axis=0))
      std_mu_rotation_perp.append( np.std(mu_rotation_perp_r, axis=0) )
      average_mu_40.append(np.mean(mu_40_r, axis=0))
      std_mu_40.append( np.std(mu_40_r, axis=0) )


  #Scale standard deviation to plot error bars with 2 std deviations
  std_mu_parallel[:]            = [xx * 2 for xx in std_mu_parallel]
  std_mu_perp[:]                = [xx * 2 for xx in std_mu_perp]
  std_mu_rotation_parallel[:]   = [xx * 2 for xx in std_mu_rotation_parallel]
  std_mu_rotation_perp[:]       = [xx * 2 for xx in std_mu_rotation_perp]
  std_mu_40[:]                  = [xx * 2 for xx in std_mu_40]

  pyplot.figure(1)
  pyplot.errorbar(x_sphere_162_blobs, average_mu_parallel, yerr=std_mu_parallel, fmt='D', color='m', label='162-blobs', markersize=6)
  pyplot.figure(2)
  pyplot.errorbar(x_sphere_162_blobs, average_mu_perp, yerr=std_mu_perp, fmt='D', color='m', label='162-blobs', markersize=6)
  pyplot.figure(3)
  pyplot.errorbar(x_sphere_162_blobs, average_mu_rotation_parallel, yerr=std_mu_rotation_parallel, fmt='D', color='m', label='162-blobs parallel', markersize=6)
  pyplot.errorbar(x_sphere_162_blobs, average_mu_rotation_perp, yerr=std_mu_rotation_perp, fmt='v', color='c', label='162-blobs perp',  markersize=6)
  pyplot.figure(4)
  pyplot.errorbar(x_sphere_162_blobs, average_mu_40, yerr=std_mu_40, fmt='D', color='m', label='162-blobs', markersize=6)



  file = open('mobility.162-blob.dat', 'w')
  line = '#columns: distance, mu_parallel, mu_perpendicular, mu_rotation_parallel, mu_rotation_perpendicular, mu_rotation_translation_40 \n'
  file.write(line)
  count=0
  for x in x_sphere_162_blobs:
    mu_parallel          = average_mu_parallel[count]
    mu_perp              = average_mu_perp[count]
    mu_rotation_parallel = average_mu_rotation_parallel[count]
    mu_rotation_perp     = average_mu_rotation_perp[count]
    mu_40                = average_mu_40[count]
    line = str(x) + ' ' + str(mu_parallel) + ' ' + str(mu_perp) + ' ' + str(mu_rotation_parallel) + ' ' + str(mu_rotation_perp) + ' ' + str(mu_40) + '\n'
    file.write(line)
    count += 1
  file.close()






  #Add theoretical curves
  mobility_SwanBrady_parallel = []
  mobility_SwanBrady_perp = []
  mobility_SwanBrady_rotation_parallel = []
  mobility_SwanBrady_rotation_perp = []
  mobility_SwanBrady_40 = []

  mobility_SwanBrady_parallel_stresslet = []
  mobility_SwanBrady_perp_stresslet = []
  mobility_SwanBrady_rotation_parallel_stresslet = []
  mobility_SwanBrady_rotation_perp_stresslet = []
  mobility_SwanBrady_40_stresslet = []

  mobility_Brenner_perp = []
  mobility_Fauxcheux_parallel = []
  mobility_Huang_parallel = []
  mobility_Huang_perp = []
  mobility_Goldman_parallel = []
  mobility_Goldman_rotation = []
  mobility_Goldman_40 = [] #Rotation translation coupling
  x_Goldman = []
  x_Huang = []
  x_theory = []
  #heights1 = np.linspace(1.0001, 1.1, 100) 
  #heights2 = np.linspace(1.15, 2, 20)
  #heights = np.concatenate([heights1, heights2])
  heights = np.log(np.logspace(1.000001, 1.10, num=200, base=np.e))
  for r in heights:
    x_theory.append(r)
    #Swan and Brady 
    mobilityTheory = selfMobilitySwanBrady.selfMobilitySwanBrady(1.0, r)
    mobility_SwanBrady_parallel.append(mobilityTheory[0][0])
    mobility_SwanBrady_perp.append(mobilityTheory[2][2])
    mobility_SwanBrady_rotation_parallel.append(mobilityTheory[3][3] / 0.75) #0.75 is the value far from the wall
    mobility_SwanBrady_rotation_perp.append(mobilityTheory[5][5] / 0.75)
    mobility_SwanBrady_40.append(mobilityTheory[4][0] / (1.0))

    #Swan and Brady with stresslet
    mobilityTheory = selfMobilitySwanBradyWithStresslet.selfMobilitySwanBradyWithStresslet(1.0, r)
    mobility_SwanBrady_parallel_stresslet.append(mobilityTheory[0][0])
    mobility_SwanBrady_perp_stresslet.append(mobilityTheory[2][2])
    mobility_SwanBrady_rotation_parallel_stresslet.append(mobilityTheory[3][3] / 0.75) #0.75 is the value far from the wall
    mobility_SwanBrady_rotation_perp_stresslet.append(mobilityTheory[5][5] / 0.75)
    mobility_SwanBrady_40_stresslet.append(mobilityTheory[4][0] / (1.0))

    #Brenner 1961
    mobilityTheory = selfMobilityBrenner.selfMobilityBrenner(1.0, r, 64)
    mobility_Brenner_perp.append(mobilityTheory)

    #Fauxcheux
    mobilityTheory = selfMobilityFauxcheux.selfMobilityFauxcheux(1.0, r)
    mobility_Fauxcheux_parallel.append( mobilityTheory ) 

    #Huang2007
    mobilityTheory = selfMobilityHuang.selfMobilityHuang(1.0, r)
    mobility_Huang_perp.append(mobilityTheory[1])
    if(r < 1.10):
      x_Huang.append(r)
      mobility_Huang_parallel.append(mobilityTheory[0])

    #Goldman
    mobilityTheory = selfMobilityGoldman.selfMobilityGoldman(1.0, r)
    if( r < 1.10 ):
      x_Goldman.append(r)
      mobility_Goldman_parallel.append( mobilityTheory[0][0] )
      mobility_Goldman_rotation.append( mobilityTheory[1][1] / 0.75)
      mobility_Goldman_40.append( mobilityTheory[0][1] )

  pyplot.figure(1)
  pyplot.plot(x_theory, mobility_SwanBrady_parallel, '--', color='k', label='1-blob')
  pyplot.plot(x_theory, mobility_SwanBrady_parallel_stresslet, '--', color='r', label='Swan & Brady')
  pyplot.plot(x_theory, mobility_Fauxcheux_parallel, '-', color='g', label='Fauxcheux')
  #pyplot.plot(x_Huang, mobility_Huang_parallel, ':', color='b', label='Huang')
  pyplot.plot(x_Goldman, mobility_Goldman_parallel, '-.', color='m', label='Goldman')
  pyplot.figure(2)
  pyplot.plot(x_theory, mobility_SwanBrady_perp, '--', color='k', label='1-blob')
  pyplot.plot(x_theory, mobility_SwanBrady_perp_stresslet, '--', color='r', label='Swan & Brady')
  pyplot.plot(x_theory, mobility_Brenner_perp, '-', color='g', label='Brenner')
  pyplot.plot(x_theory, mobility_Huang_perp, ':', color='b', label='Huang')
  pyplot.figure(3)
  pyplot.plot(x_theory, mobility_SwanBrady_rotation_parallel, '-', color='k', label='1-blob parallel')
  pyplot.plot(x_theory, mobility_SwanBrady_rotation_perp, '--', color='k', label='1-blob perp')
  pyplot.plot(x_theory, mobility_SwanBrady_rotation_parallel_stresslet, '-', color='r', label='Swan & Brady parallel')
  pyplot.plot(x_theory, mobility_SwanBrady_rotation_perp_stresslet, '--', color='r', label='Swan & Brady perp')
  pyplot.plot(x_Goldman, mobility_Goldman_rotation, 'x-', color='c', label='Goldman parallel')
  pyplot.figure(4)
  pyplot.plot(x_theory, mobility_SwanBrady_40, '--', color='k', label='1-blob')
  pyplot.plot(x_theory, mobility_SwanBrady_40_stresslet, '--', color='r', label='Swan & Brady')
  pyplot.plot(x_Goldman, mobility_Goldman_40, 'x-', color='m', label='Goldman')


  #Parallel mobility
  pyplot.figure(1)
  pyplot.ylim(0.0, 0.5)
  pyplot.xlim(1.0, 1.11)
  pyplot.rcParams.update({'font.size': 16})
  pyplot.title('Parallel Mobility')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('H / a_effective')
  pyplot.ylabel('Mobility / Bulk Mobility')
  pyplot.savefig('./figures/ParallelMobility.pdf')
  pyplot.clf()

  #Perpendicular mobility
  pyplot.figure(2)
  pyplot.ylim(-0.1, 0.5)
  pyplot.xlim(1.0, 1.11)
  pyplot.title('Perpendicular Mobility')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('H / a_effective')
  pyplot.ylabel('Mobility / Bulk Mobility') 
  pyplot.savefig('./figures/PerpendicularMobility.pdf')
  pyplot.clf()

  #Rotational mobility
  pyplot.figure(3)
  pyplot.ylim(0.0, 1.0)
  pyplot.xlim(1.0, 1.11)
  pyplot.title('Rotational Mobility')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('H / a_effective')
  pyplot.ylabel('Mobility / Bulk Mobility')
  pyplot.savefig('./figures/RotationalMobility.pdf')
  pyplot.clf()

  #Rotarion-translation coupling
  pyplot.figure(4)
  pyplot.xlim(1.0, 1.11)
  pyplot.ylim(0.0, 0.5)
  pyplot.title('Rotation-translation mobility coupling')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('H / a_effective')
  pyplot.ylabel('Mobility x (6 pi eta a)')
  pyplot.savefig('./figures/RotationTranslationMobilityCoupling.pdf')
  pyplot.clf()
  



if __name__ == '__main__':
  
  a = 0.52573111211900003 * 1
  asph42 = 0.2732666211340000206320155 * 1
  asph162 = 0.1379519651259999979409088 * 1
  heights = np.linspace(1.000, 1.1, 10)
  plot_mobilities(a, asph42, asph162, heights)

