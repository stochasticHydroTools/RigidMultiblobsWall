import sys
import numpy as np
import scipy.sparse as sp
found_functions = False
path_to_append = ''
while found_functions is False:
    try:
        from DoublyPeriodicStokes.python_interface.common_interface_wrapper import FCMJoint as FCMJoint
        found_functions = True
    except:
        path_to_append += '../'
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
            print('\n DoublyPeriodicStokes not found, check '+__file__)
            sys.exit()


def shift_heights(r_vectors, blob_radius):
    '''
    Return an array with the blobs' height

    z_effective = maximum(z, blob_radius)

    This function is used to compute positive
    definite mobilites for blobs close to the wall.
    '''
    r_effective = np.copy(r_vectors)
    r_effective[r_vectors[:,2] <= blob_radius, 2] = blob_radius

    return r_effective


def damping_matrix_B(r_vectors, blob_radius):
    '''
    Return sparse diagonal matrix with components
    B_ii = 1.0               if z_i >= blob_radius
    B_ii = z_i / blob_radius if z_i < blob_radius

    It is used to compute positive definite mobilities
    close to the wall.
    '''
    B = np.ones(r_vectors.size)
    overlap = False
    for k, r in enumerate(r_vectors):
        if r[2] < blob_radius:
            B[k*3]     = r[2] / blob_radius
            B[k*3 + 1] = r[2] / blob_radius
            B[k*3 + 2] = r[2] / blob_radius
            overlap = True
    return (sp.dia_matrix((B, 0), shape=(B.size, B.size)), overlap)

def put_r_vecs_in_periodic_box(r_vecs_np, L):
    r_vecs = np.copy(r_vecs_np)
    for r_vec in r_vecs:
        for i in range(3):
            if L[i] > 0:
                while r_vec[i] < 0:
                    r_vec[i] += L[i]
                while r_vec[i] > L[i]:
                    r_vec[i] -= L[i]
    return r_vecs    
    
class FCMJoint_MB(FCMJoint):
    
    def Mdot_MB(self, r_vectors_blobs, force, eta, a, *args, **kwargs):
        """Computes the product of the Mobility tensor with the provided forces and torques. 
           If torques are not present, they are assumed to be zero and angular displacements will not be computed
        """
        r_vectors_effective = shift_heights(r_vectors_blobs, a)
        # Compute damping matrix B
        B, overlap = damping_matrix_B(r_vectors_blobs, a)
        # Compute B * force
        if overlap is True:
            force = B.dot(force.flatten())            
        L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
        r_vectors_effective = put_r_vecs_in_periodic_box(r_vectors_effective, L)
        self.SetPositions(r_vectors_effective.flatten())
        MF,_ = self.Mdot(forces=force);
        if overlap is True:
            MF = B.dot(MF)
        return MF
