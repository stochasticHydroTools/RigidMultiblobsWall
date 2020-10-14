
import numpy as np
import sys
import mmap

if __name__ == '__main__':
  # Init variables
  name_file = sys.argv[1]
  use_column = int(sys.argv[2])
  start = float(sys.argv[3])
  end = float(sys.argv[4])
  num_intervales = int(sys.argv[5])
  dx = (end - start) / num_intervales
  histogram = np.zeros(num_intervales)
  comment_symbols = ['#']   

  # Read file to memory
  with open(name_file, 'r') as f:
    f_read = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    
    # Read file and create histogram
    for line_b in iter(f_read.readline, b''):
      line = line_b.decode()
      # Strip comments
      if comment_symbols[0] in line:
        line, comment = line.split(comment_symbols[0], 1)
      else:
        data = line.split()
        n = int((float(data[use_column]) - start) / dx)
        if n < num_intervales and n >= 0:
          histogram[n] += 1


  # Print histogram
  norm = sum(histogram) * dx
  for i in range(num_intervales):
    print(start + (i+0.5) * dx, histogram[i] / norm)
    
