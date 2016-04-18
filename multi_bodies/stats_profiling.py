import pstats
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-filename', dest='filename', type=str,
                      default='logs/profiling.txt')
args=parser.parse_args()

src = 'logs/profiling.log'

log_filename = args.filename
prof_filename = log_filename.replace('rod','profiling')
if os.path.isfile(prof_filename) == False:
 print prof_filename, " does not exist, I use ", src, " and rename it"
 os.rename(src,prof_filename)

# Filename of the file to be saved in human readable format
dest_filename = prof_filename.replace('.log','.txt')
my_stream = open(dest_filename, 'w');

print "Write stats in ", dest_filename

p = pstats.Stats(prof_filename, stream=my_stream)
#p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats('time').print_stats(20)
