import numpy as np
import ntpath
import sys
from datetime import datetime
import os
import shutil

import fibers.c_fibers_obj as fibers

class RigidFibers(object):
    '''
    Simple class to read the input files to run a simulation. TODO replace
    '''

    def __init__(self, fname, calling_script_name):
        ''' Construnctor takes the name of the input file and name of simulation file'''
        self.input_file = fname
        self.options = {}
        self.cf = fibers.CManyFibers()
        self.precision = np.float32 if self.cf.precision == "float" else np.float64


        if calling_script_name.find("/") != -1: # parses out absolute path
            self.calling_script = calling_script_name.split("/")[-1].split(".")[0] # assumes calling_script was passed in using __file__
        else:
            self.calling_script = calling_script_name

        # Read input file
        comment_symbol = '#'
        with open(self.input_file, 'r') as f:
            # Loop over lines
            for line in f:
                # Strip comments
                if comment_symbol in line:
                    line, comment = line.split(comment_symbol, 1)

                # Save options to dictionary, Value may be more than one word
                line = line.strip()
                if line != '':
                    option, value = line.split(maxsplit=1)

                    self.options[option.lower()] = value

        # Set option to file or default values with some error checking

        # geom params
        self.a = float(self.options.get('blob_radius') or 1.0)
        self.ds = float(self.options.get('ds') or 1.0)
        self.N_fibers = int(self.options.get('n_fibers') or 1)
        self.N_blobs = int(self.options.get('n_blobs_per_fiber') or 1)

        # solvers
        self.solver = str(self.options.get('solver') or 'RPY')
        self.time_stepper = str(self.options.get('time_solver') or 'be')

        # TODO should a domain type be specified instead of a solver? e.g. one wall, triply periodic, etc
        match self.solver.lower():
            case "rpy":
                self.domainInt = 0
            case "rpy_batched":
                self.domainInt = 1
            case "rpy_wall":
                self.domainInt = 2
            case "rpy_wall_batched":
                self.domainInt = 3
            case "dp_stokes": # doubly periodic
                self.domainInt = 4
            case "pse": # triply periodic
                self.domainInt = 5
            case _:
                raise Exception(f"Solver \'{self.solver}\' not found")
            
        match self.time_stepper.lower():
            case "cn":
                self.impl_c = 0.5
            case "be":
                self.impl_c = 1.0
        
        # physical params
        self.kBT = float(self.options.get('kbt') or 0.004142) # default is boltzman constant in attojoules at 300 Kelvin
        self.k_bend = float(self.options.get('k_bend') or 1.0)  # TODO default to persistence length of 1?
        self.eta = float(self.options.get('viscosity') or 1.0)
        self.L_f = float(self.options.get('fiber_length') or self.ds*self.N_blobs)

        # scale params
        default_M0 = np.log(self.L_f/self.a)/(8.0*np.pi*self.eta*self.a)
        self.M0 = float(self.options.get('mobility_scale') or default_M0)

        # time params
        self.N_steps = int(self.options.get('n_steps') or 1) # doesn't go into C code, just useful to know
        self.alpha = float(self.options.get('alpha') or ((self.ds**3)/self.M0)/self.k_bend )
        self.alpha_factor = float(self.options.get('alpha_factor') or 1.0)
        self.dt = float(self.alpha*self.alpha_factor)

        # file output params
        self.save_pos = bool(int(self.options.get('save_positions') or 0))
        self.save_e2e = bool(int(self.options.get('save_end_to_end') or 0))

        if self.save_pos:
            self.n_save_pos = int(self.options.get('n_save_positions') or 100)
            self.pos_file = str(self.options.get('output_positions_file'))

            if self.pos_file is None:
                raise Exception("Error: to save positions, you must set output_position_file in the input file.")
            
        if self.save_e2e:
            self.n_save_pos = int(self.options.get('n_save_end_to_end') or 100)
            self.e2e_file = str(self.options.get('end_to_end_file'))
            self.e2e_write = False

            if self.e2e_file is None:
                raise Exception("Error: to save end to end distances, you must set output_end_to_end_dist_file in the input file.")


        # Solver specific parameters
        if self.solver.lower() == "pse":
            L_p = self.options.get('periodic_length')
            if L_p is None:
                raise Exception("Error: PSE solver requires a triply periodic length. Set parameter periodic_length in your input file.")
            self.L_p = float(L_p)
            self.psi = str(self.options.get('psi') or 0.15/self.a) # PSE splitting param
            # TODO set psi in c code?
        else:
            self.L_p = -1 # needs to get set for C constructor


        # currently we set precision based on how libmobility is compiled
        # precision = str(self.options.get('precision') or 'double')
        # match precision.lower():
        #     case "double":
        #         self.precision = np.float64
        #     case "single":
        #         self.precision = np.float32


        # other unusual params
        self.clamp = bool(self.options.get("clamp" or False))
        if self.clamp:
            fixed_tangent = self.options.get("fixed_tangent" or None)
            if fixed_tangent is None:
                raise Exception("Error: Clamping a fiber requires a specified fixed tangent vector for the base of the fiber." +
                                "Specify fixed_tangent in the format 'a b c' where a,b,c represent the end points of a vector in Cartesian coordinates")
            coords = fixed_tangent.split(" ")
            self.T_fix = np.array([float(coords[0]), float(coords[1]), float(coords[2])], dtype=self.precision)
        else:
            self.T_fix = np.array([0.0,1.0,0.0], dtype=self.precision) # still needed in C constructor

        self.cf.setParameters(self.domainInt, self.N_fibers, self.N_blobs, 
                            self.a, self.ds, self.dt, self.k_bend, self.M0,
                            self.impl_c, self.kBT, self.eta,
                            self.L_p, self.clamp, self.T_fix)

        # cf.setParameters(DomainInt, Nfibs, Nblobs, a, ds, dt, k_bend, M0, impl_c, kBT, eta, Lperiodic, Clamp, T_base)


        # def save_positions(self):
        #         pos = self.cf.multi_fiber_Pos(All_Taus,All_X)
        #         pos = np.asarray(pos)
        #         with open("./Free_Fiber_Data/fiber_pos.txt", status) as outfile:
        #             pos.tofile(outfile, " ")
        #             outfile.write("\n")

    def make_run_directory(self, dir=None, prefix="./sim_output/"):
        if dir is None:
            time = datetime.now()
            time = time.strftime("%m-%d-%y_%H:%M")
            # calling_script = __file__.split("/")[-1].split(".")[0] # old version, now in __init__

            dirname = prefix + self.calling_script + "_" + time + "/"
            os.makedirs(dirname)
            self.out_dir = dirname

            partial_input = self.input_file.split("/")[-1]
            shutil.copy(self.input_file, dirname + partial_input)


            




    def save_e2e_dist(self, all_taus):
        if not self.e2e_write: # create new file if first write of the class
            status = 'w'
            self.e2e_write = True
        else:
            status = 'a'

        e2e = self.cf.end_to_end_distance(all_taus)
        with open(self.out_dir + self.e2e_file, status) as outfile:
            for ee in e2e:
                outfile.write(str(ee) + ",")
            outfile.write("\n")
        

