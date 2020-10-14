
import numpy as np
import sys







if __name__ == "__main__":

    # Read names
    prefix = sys.argv[1]
    suffix = sys.argv[2]
    firstFile = int(sys.argv[3])
    lastFile = int(sys.argv[4])
    num_columns = int(sys.argv[5])

    # Define variables
    xm = np.zeros([num_columns,400000])
    xerror = np.zeros([num_columns,400000])
    x = np.zeros([num_columns])
    xorigin = np.zeros([num_columns,400000])

    # Read files
    comment_symbols = ['#']   
    count = 0
    for i in range(firstFile, lastFile+1):
      name = prefix + str(i) + suffix
      f = open(name, 'r')
      j = 0
      # data = f.readline()
      for line in f:
        if comment_symbols[0] in line:
          continue
        data = line.split()
        for k in range(num_columns):             
          x[k] = float(data[k])
          xerror[k,j] += count * (x[k] - xm[k,j])*(x[k] - xm[k,j]) / (count+1)
          xm[k,j] += (x[k]-xm[k,j]) / (count+1)
        j += 1
      count += 1
      f.close()

    for i in range(j):
      for j in range(num_columns):
        print(xm[j,i], end=' ')
      for j in range(num_columns):
        print(np.sqrt(xerror[j,i]) / np.sqrt(count * np.maximum(1.0,(count-1.0))), end=' ')
      print()

