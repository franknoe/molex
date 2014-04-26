import numpy as np
import scipy.optimize


def fingerprint_positive(t, y, d = None, timescales = None):
    """
    Computes a positive dynamical fingerprint from a 1D relaxation curve

    Computes the timescale components in a one-dimensional time-dependent
    experimental relaxaiton curve assuming that all coefficients are
    nonnegative. The dynamical fingerprint is conceptually equivalent to 
    (although numerically quite different from) computing an inverse 
    Laplace transform of the signal.
    Fingerprints with nonnegativity constraints are useful for analyzing
    experimental autocorrelation functions as the occur e.g. in 
    fluorescence correlation spectroscopy and elsewhere.

    Parameters
    ==========
    t : ndarray (n)
        time points of the signal
    y : ndarray (n)
        measurement values of the signal
    d : ndarray (n)
        measurement error (uncertainty) for every time point. If set to None
        (default), all measurement errors are treated constant. The smaller
        d, the more a data point will enter the fit.

    """
    n = len(t)
    if (len(y) != n):
        raise ValueError('Inconcistent lengths in time points and measurements')

    if (d is None):
        d = np.ones(n)

    if (timescales is None):
        timescales = t
    
    T_DAT = np.matrix(t).transpose() * np.matrix(np.ones(n))
    T_FP = np.transpose(np.matrix(timescales).transpose()*np.matrix(np.ones(n)))
    M = np.exp(-np.divide(T_DAT,T_FP))
    w = np.diag(1./d)

    (fp,norm) = scipy.optimize.nnls(w*M,y/d)
    return fp

# alias
timescale_spectrum_positive = fingerprint_positive

def sample_fingerprint_positive(t, y, d = None, timescales = None, nsample = 10000):
    """
    Conducts a Markov chain Monte Carlo simulation to sample fingerprints.
    """
