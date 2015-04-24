''' Script to plot the height of the CoM of the free tetrahedron v. 
the IBAMR FIB method and theory. '''

import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os

import tetrahedron_free as tf

IBAMR_BUCKETS = np.array(
  [0.041666668, 0.125, 0.20833334, 0.29166669, 0.375, 0.45833334, 0.54166669, 0.625, 0.70833337, 0.79166669, 0.875, 0.95833337, 1.0416667, 1.125, 1.2083334, 1.2916667, 1.375, 1.4583334, 1.5416667, 1.625, 1.7083334, 1.7916667, 1.875, 1.9583334, 2.0416667, 2.125, 2.2083335, 2.2916667, 2.375, 2.4583335, 2.5416667, 2.625, 2.7083335, 2.7916667, 2.875, 2.9583335, 3.0416667, 3.125, 3.2083335, 3.2916667, 3.375, 3.4583335, 3.5416667, 3.625, 3.7083335, 3.7916667, 3.875, 3.9583335, 4.041667, 4.125, 4.2083335, 4.291667, 4.375, 4.4583335, 4.541667, 4.625, 4.7083335, 4.791667, 4.875, 4.9583335, 5.041667, 5.125, 5.2083335, 5.291667, 5.375, 5.4583335, 5.541667, 5.625, 5.7083335, 5.791667, 5.875, 5.9583335, 6.041667, 6.125, 6.2083335, 6.291667, 6.375, 6.4583335, 6.541667, 6.625, 6.7083335, 6.791667, 6.875, 6.9583335, 7.041667, 7.125, 7.2083335, 7.291667, 7.375, 7.4583335, 7.541667, 7.625, 7.7083335, 7.791667, 7.875, 7.9583335, 8.041667, 8.125, 8.208334, 8.291667, 8.375, 8.458334, 8.541667, 8.625, 8.708334, 8.791667, 8.875, 8.958334, 9.041667, 9.125, 9.208334, 9.291667, 9.375, 9.458334, 9.541667, 9.625])

IBAMR_HEIGHT_PDF = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.00016362722, 0.001804999, 0.008731749, 0.025426527, 0.065377451, 0.11898809, 0.20434552, 0.28891435, 0.3621815, 0.41390688, 0.47103478, 0.52470066, 0.55978978, 0.58630633, 0.59373572, 0.59870666, 0.58981084, 0.57641556, 0.54082188, 0.50282576, 0.45503682, 0.41518324, 0.3812658, 0.34825913, 0.31418176, 0.28920474, 0.26989921, 0.24970623, 0.22898739, 0.20797949, 0.18802668, 0.17552027, 0.15954974, 0.14340056, 0.11988096, 0.10528827, 0.095538669, 0.08268395, 0.077479498, 0.068770892, 0.060564998, 0.055432152, 0.048079361, 0.046548747, 0.041057594, 0.03323098, 0.030264686, 0.025251097, 0.022834306, 0.021998069, 0.020500914, 0.017412232, 0.015891893, 0.013566843, 0.011633356, 0.011052122, 0.01084445, 0.0093264198, 0.0078583059, 0.0061832399, 0.0060315248, 0.0046111069, 0.0045738013, 0.0037832811, 0.0033536999, 0.0029780228, 0.0028575082, 0.0029287259, 0.0033788728, 0.003873339, 0.0029562341, 0.0027608996, 0.0024379995, 0.0017596554, 0.0016605531, 0.0013265696, 0.0015332804, 0.0016332411, 0.0015495885, 0.0013001159, 0.0016375327, 0.0016922338, 0.0018767826, 0.0021278947, 0.0013266468, 0.0018571183, 0.0012557147, 0.001371048, 0.0012284027, 0.0012651562, 0.0012010907, 0.0013958622, 0.0011361669, 0.00081154778, 0.001103705, 0.00071416205, 0.00035708102, 0.00037331198, 0.00021100242, 0.00025969529, 9.7385734e-05, 4.8692867e-05])




IBAMR_HEIGHT_STD = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 9.9134208e-05, 0.0004826588, 0.0018538702, 0.0038670874, 0.0071828824, 0.0094317394, 0.013795536, 0.018689622, 0.02074595, 0.020668026, 0.020192627, 0.019550605, 0.021015738, 0.019476591, 0.020917931, 0.021594424, 0.020888806, 0.020768723, 0.020902619, 0.019834425, 0.016831116, 0.015291128, 0.012820219, 0.01227613, 0.011516209, 0.01109446, 0.012600609, 0.011533456, 0.0099352951, 0.011440802, 0.012111104, 0.013517728, 0.012973561, 0.012734543, 0.011668602, 0.010968397, 0.010852096, 0.0099903967, 0.010864025, 0.0095840791, 0.0089071712, 0.0093583118, 0.0078861122, 0.0085969012, 0.0077768035, 0.0064760442, 0.0063044863, 0.0057341463, 0.0060829819, 0.0058367432, 0.0061311266, 0.0049479644, 0.0048963709, 0.0045376736, 0.0039262593, 0.003898314, 0.0037683465, 0.0031539999, 0.0030955789, 0.0026844624, 0.0033978274, 0.0026894057, 0.0028829684, 0.0023545012, 0.0021594141, 0.0019034455, 0.0019479546, 0.0020647438, 0.0025534215, 0.0027557581, 0.0020990125, 0.0022067157, 0.0021771737, 0.0013423729, 0.0012334962, 0.0009990866, 0.001143624, 0.0012641378, 0.0013773252, 0.0010462028, 0.0013224502, 0.0014623297, 0.0017411159, 0.001846847, 0.0012299867, 0.0016643996, 0.0010641033, 0.0012028138, 0.0011190296, 0.0012297056, 0.0011821747, 0.0013738787, 0.0011182734, 0.00079876671, 0.0010863227, 0.0007029147, 0.00035145735, 0.00036743268, 0.00020767934, 0.00025560535, 9.5852005e-05, 4.7926002e-05])


def get_mean_and_std_heights(data_files, ind):
  ''' Given a list of names of pkl data files, return the buckets,
  mean pdf, and std of pdf for geometric center.'''

  heights_list = []
  for file_name in data_files:
    data_path = os.path.join('.', 'data', file_name)
    with open(data_path, 'rb') as f:
      heights_data = cPickle.load(f)
      heights_list.append(heights_data['heights'][ind][4])

  heights_mean = np.mean(heights_list, axis=0)
  heights_std = np.std(heights_list, axis=0)/np.sqrt(len(data_files))

  return [heights_data['buckets'], heights_mean, heights_std]


if __name__ == '__main__':
  rfd_height_data = [
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-1.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-2.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-3.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-4.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-5.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-6.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-7.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-8.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-9.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-10.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-11.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-12.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-13.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-14.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-15.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-RFD-com-pdf-16.pkl']
  fixman_height_data = [
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-1.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-2.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-3.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-4.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-5.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-6.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-7.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-8.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-9.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-10.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-11.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-12.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-13.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-14.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-15.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-FIXMAN-com-pdf-16.pkl']
  em_height_data = [
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-1.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-2.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-3.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-4.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-5.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-6.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-7.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-8.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-9.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-10.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-11.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-12.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-13.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-14.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-15.pkl',
    'free-tetrahedron-heights-dt-0.8-N-300000-scheme-EM-com-pdf-16.pkl']


  [rfd_buckets, rfd_means, rfd_std] = get_mean_and_std_heights(
    rfd_height_data, 0)
  [fixman_buckets, fixman_means, fixman_std] = get_mean_and_std_heights(
    fixman_height_data, 0)
  [em_buckets, em_means, em_std] = get_mean_and_std_heights(
    em_height_data, 0)

  # Get equilibirum from all
  [gb_buckets, gb_means, gb_std] = get_mean_and_std_heights(
    rfd_height_data + fixman_height_data + em_height_data, 1)

  err_subsample = 3

  plt.plot(fixman_buckets, fixman_means,
           c='g', label='Fixman')
  plt.plot(rfd_buckets, rfd_means,
           c='b', label='RFD')
  plt.plot(em_buckets, em_means,
           c='m', label='EM')
  plt.plot(IBAMR_BUCKETS, IBAMR_HEIGHT_PDF,
           c='r', label='FIB')
  plt.plot(gb_buckets, gb_means,
           c='k', label='MCMC')

  plt.errorbar(fixman_buckets[::err_subsample], fixman_means[::err_subsample],
               yerr = 2.*fixman_std[::err_subsample],
                c='g', linestyle = '')
  plt.errorbar(rfd_buckets[::err_subsample], rfd_means[::err_subsample],
               yerr=2.*rfd_std[::err_subsample],
                c='b', linestyle = '')
  plt.errorbar(em_buckets[::err_subsample], em_means[::err_subsample],
               yerr=2.*em_std[::err_subsample],
                c='m', linestyle='')
  plt.errorbar(IBAMR_BUCKETS[::err_subsample], 
               IBAMR_HEIGHT_PDF[::err_subsample],
               yerr=IBAMR_HEIGHT_STD[::err_subsample],
                c='r', linestyle='')
  plt.errorbar(gb_buckets[::err_subsample], gb_means[::err_subsample], 
               yerr=2.*gb_std[::err_subsample],
               c='k', linestyle='')

  plt.xlim([0., 9.])
  plt.title('PDF of Height of Center of for Free Tetrahedron.')
  plt.xlabel('Height')
  plt.ylabel('PDF')
  plt.legend(loc='best', prop={'size': 10})

  plt.savefig(os.path.join('.', 'figures', 
                           'FreeTetrahedronCenterPDF.pdf'))

  with open(os.path.join('.', 'data', 
                         'TetrahedronIBAMRComparisonHeights.txt'), 'wb') as f:
    f.write('Buckets:\n')
    f.write('%s\n' % fixman_buckets)
    f.write('Fixman PDF:\n')
    f.write('%s\n' % fixman_means)
    f.write('Fixman PDF std dev (x1):\n')
    f.write('%s\n' % fixman_std)
    f.write('--------------\n')
    f.write('\n')

    f.write('RFD PDF:\n')
    f.write('%s\n' % rfd_means)
    f.write('RFD PDF std dev (x1):\n')
    f.write('%s\n' % rfd_std)
    f.write('--------------\n')
    f.write('\n')

    f.write('EM PDF:\n')
    f.write('%s\n' % em_means)
    f.write('EM PDF std dev (x1):\n')
    f.write('%s\n' % em_std)
    f.write('--------------\n')
    f.write('\n')
    
  
