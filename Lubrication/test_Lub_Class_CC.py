import numpy as np
import Lubrication_Class as Lub_cc
Lcc =  Lub_cc.Lubrication(0.01)
r_hat = np.array([0.605965, -0.79543, 0.00988565])
r_norm = 3.20447
Lcc.ResistPairSup_py(r_norm, 1.0, 1.0/6.0/np.pi, r_hat)
