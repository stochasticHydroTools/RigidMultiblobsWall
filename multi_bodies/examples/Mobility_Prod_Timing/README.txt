to run the code, use 
python3 main.py --input-file Multiblob.inputfile

In the file 'Multiblob.inputfile' parameters are specified for the simulations. Te important ones are
1) 'blob_radius' - line 19: this actually specifies the radius of the sperical particle. The main program will modify the actual blob radius 'a' to account for a multiblob model of a paricle with radius 'blob_radius'
   NOTE: This is an important distiction between particle radius and blob radius, but it is taken care of for you. 
   The code will both print the new blob radius 'a' and save it to the header of a configuration file e.g './data/Cfg_12_blobs_per.txt' with the positions of all of the blobs
   
2) Comment or Uncomment lines 52-55 to use different multiblob models in the product. e.g. for a 1-blob model of a particle use line 52, for a 12 blob model of a particle use line 53, ...
   
   
it will output:
1) the vertex positions of the 'refference' (centered at the origin) multiblob sphere based on what was chosen from the input file
2) the blob radius 'a' after it has been modified to account for the multiblob model being used
   NOTE: the value of a specified in the input file specifies the **particle radius** and the blob radius (I also use 'a' for this) is smaller if more than one blob is used to represent the particle
3) Timings for each block of the mobility product (e.g M^tr*T or M^tt*F)  
4) timing for the whole trans-rot mobility product
 

The code will save:
1) configuration file e.g './data/Cfg_12_blobs_per.txt' with the positions of all of the blobs
2) a data file with the random forces and torques, as well as the result of each mobility block-product (e.g M^tr*T or M^tt*F) 
   e.g './data/Mob_Product_12_blobs_per.txt'
