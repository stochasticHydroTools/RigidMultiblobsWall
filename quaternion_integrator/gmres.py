'''
Wrapper for scipy gmres to use right preconditioner by David Stein at Flatiron Institute.
'''
try:
  from scipy.sparse.linalg.isolve.utils import make_system
  from scipy.sparse.linalg.isolve import _iterative
except:
  from scipy.sparse.linalg._isolve.utils import make_system
  from scipy.sparse.linalg._isolve import _iterative
from scipy._lib._util import _aligned_zeros
import numpy as np
import scipy
from functools import partial
import scipy.sparse.linalg as scspla


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        mydict['resnorms'] = []
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
        try:
            mydict['resnorms'].append(rk)
        except:
            pass
mydict = {
}

def _stoptest(residual, atol):
    resid = np.linalg.norm(residual)
    if resid <= atol:
        return resid, 1
    else:
        return resid, 0
_type_conv = {'f':'s', 'd':'d', 'F':'c', 'D':'z'}

w1 = scipy.__version__.split('.')
scipy_version = int(w1[0]) + int(w1[1])/10.0 + int(w1[2])/100.0

def direct_gmres(A, b, verbose, **kwargs):
    counter = gmres_counter(verbose)
    if 'callback' in kwargs and kwargs['callback'] is not None:
        _cb = kwargs['callback']
        def callback(rk=None):
            _cb(rk)
            counter(rk)
        kwargs.pop('callback')
    else:
        callback = counter
    out = scipy.sparse.linalg.gmres(A, b, callback=callback, **kwargs)
    return out[0], out[1], mydict['resnorms']

def presid_gmres(A, b, verbose, x0=None, tol=1e-05, restart=None, maxiter=None, M=None, **kwargs):
    callback = gmres_counter(verbose)

    A, M, x, b, postprocess = make_system(A, M, x0, b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    if restart is None:
        restart = 20
    restart = min(restart, n)

    matvec = A.matvec
    psolve = M.matvec
    ltr = _type_conv[x.dtype.char]
    revcom = getattr(_iterative, ltr + 'gmresrevcom')

    bnrm2 = np.linalg.norm(b)
    Mb_nrm2 = np.linalg.norm(psolve(b))
    get_residual = lambda: np.linalg.norm(matvec(x) - b)
    atol = tol

    if bnrm2 == 0:
        return postprocess(b), 0

    # Tolerance passed to GMRESREVCOM applies to the inner iteration
    # and deals with the left-preconditioned residual.
    ptol_max_factor = 1.0
    ptol = Mb_nrm2 * min(ptol_max_factor, atol / bnrm2)
    resid = np.nan
    presid = np.nan
    ndx1 = 1
    ndx2 = -1
    # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
    work = _aligned_zeros((6+restart)*n,dtype=x.dtype)
    work2 = _aligned_zeros((restart+1)*(2*restart+2),dtype=x.dtype)
    ijob = 1
    info = 0
    ftflag = True
    iter_ = maxiter
    old_ijob = ijob
    first_pass = True
    resid_ready = False
    iter_num = 1
    while True:
        ### begin my modifications
        if presid/bnrm2 < atol:
            resid = presid/bnrm2
            info = 1
        if info: ptol = 10000
        ### end my modifications
        x, iter_, presid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
           revcom(b, x, restart, work, work2, iter_, presid, info, ndx1, ndx2, ijob, ptol)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)
        if (ijob == -1):  # gmres success, update last residual
            if resid_ready and callback is not None:
                callback(presid / bnrm2)
                resid_ready = False
            break
        elif (ijob == 1):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(x)
        elif (ijob == 2):
            work[slice1] = psolve(work[slice2])
            if not first_pass and old_ijob == 3:
                resid_ready = True

            first_pass = False
        elif (ijob == 3):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(work[slice1])
            if resid_ready and callback is not None:
                callback(presid / bnrm2)
                resid_ready = False
                iter_num = iter_num+1

        elif (ijob == 4):
            if ftflag:
                info = -1
                ftflag = False
            resid, info = _stoptest(work[slice1], atol)

            # Inner loop tolerance control
            if info or presid > ptol:
                ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
            else:
                # Inner loop tolerance OK, but outer loop not.
                ptol_max_factor = max(1e-16, 0.25 * ptol_max_factor)

            if resid != 0:
                ptol = presid * min(ptol_max_factor, atol / resid)
            else:
                ptol = presid * ptol_max_factor

        old_ijob = ijob
        ijob = 2

        if iter_num > maxiter:
            info = maxiter
            break

    if info >= 0 and not (resid <= atol):
        # info isn't set appropriately otherwise
        info = maxiter
    
    return postprocess(x), info, mydict['resnorms']

def gmres(A, b, verbose=False, convergence='resid', **kwargs):
    """
    Interface function to Scipy's GMRES to allow convergence with respect
        to either the actual residual or the preconditioned residual

    A, b:    see documentation to scipy.linalg.sparse.gmres
    verbose: whether to print residuals to screen during run
    kwargs:  keyword args to be passed on to scipy gmres
    convergence:
        'resid':  converge when ||Ax-b|| < tol
        'presid': converge when ||M^{-1}(Ax-b)|| < tol
        when 'presid' is specified, only the following kwargs are accepted:
            x0, tol, restart, maxiter, M

    The use of convergence 'presid' is useful in certain circumstances
        particularly when A is ill-conditioned
    """
    if convergence == 'resid':
        if scipy_version > 1.0:
            gmres_func = direct_gmres
        else:
            raise Exception("Your version of scipy does not support GMRES with residual convergence.  Set convergence='presid', or upgrade scipy to a version > 1.0.")
    elif convergence == 'presid':
        if scipy_version > 1.0:
            gmres_func = presid_gmres
        else:
            gmres_func = direct_gmres
    else:
        raise Exception("convergence must be set to 'resid' or 'presid'")
    return gmres_func(A, b, verbose, **kwargs)

def right_gmres(A, b, verbose=False, **kwargs):
    """
    Interface function to Scipy's GMRES to allow the use of right-preconditioning

    A, b:    see documentation to scipy.linalg.sparse.gmres
    verbose: whether to print residuals to screen during run
    kwargs:  keyword args to be passed on to scipy gmres
        for this function, M must be specified!

    (thanks to Floren Balboa-Usabiaga for the code)
    """
    if 'M' not in kwargs:
        raise Exception('M must be a kwarg for right_gmres')

    M = kwargs['M']
    kwargs.pop('M')

    A_LO = scipy.sparse.linalg.aslinearoperator(A)
    M_LO = scipy.sparse.linalg.aslinearoperator(M)
    
    # Define new LinearOperator A*P^{-1}
    def APinv(x,A,M):
        return A.matvec(M.matvec(x))
    APinv_partial = partial(APinv, A=A_LO, M=M_LO)
    APinv_partial_LO = scipy.sparse.linalg.LinearOperator(A.shape, matvec=APinv_partial, dtype=A.dtype) 

    # Solve system A*P^{-1} * y = b
    y, info, resnorms = gmres(APinv_partial_LO, b, verbose, **kwargs, convergence='presid')

    # Solve system P*x = y
    x = M_LO.matvec(y)

    # Return solution and info
    return x, info, resnorms


