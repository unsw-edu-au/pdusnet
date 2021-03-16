#!/usr/bin/python
"""
Python implementation of the icc method implemented originally by Travis Smith in MATLAB from
Smith TB, Smith N. Agreement and reliability statistics for shapes. PLoS One. 2018;
13(8):e0202087. Published 2018 Aug 23. doi:10.1371/journal.pone.0202087
"""

import numpy as np
import scipy.io as sio
import scipy.stats as sstats
from shape_icc import shape_icc

def icc(M):
    """ICC Intraclass correlation coefficient.
    Calculates ICC is for a two-way, fully crossed random efects model.
    This type of ICC is appropriate to describe the absolute agreement 
    among shape measurements from a group of k raters, randomly selected 
    from the population of all raters, made on a set of n items.  
    Shrout and Fleiss: ICC(2,1)
    McGraw and Wong:   ICC(A,1)

    M is the array of measurements
        The dimensions of M are n x k, where
            n is the # of subjects / groups
            k is the # of raters """

    if not isinstance(M, np.ndarray):
        raise TypeError("Input must be a numpy array")

    n, k = M.shape

    u1 = np.mean(M, axis = 0)
    u2 = np.mean(M, axis = 1)
    u = np.mean(M[:])

    SS = np.sum((M - u) ** 2)
    MSR = k/(n-1) * np.sum((u2-u) ** 2)
    MSC = n/(k-1) * np.sum((u1-u) ** 2)
    MSE = (SS - (n-1)*MSR - (k-1)*MSC) / ((n-1)*(k-1))
    icc = (MSR - MSE) / (MSR + (k-1)*MSE + k/n*(MSC-MSE))
    
    #from ICC(A,1) - in IRR package
    ns = n
    nr = k
    coeff = (MSR - MSE)/(MSR + (nr - 1) * MSE + 
                  (nr/ns) * (MSC - MSE))
    r0 = 0
    a = (nr * r0)/(ns * (1 - r0))
    b = 1+ (nr * r0 * (ns - 1))/(ns * (1 - r0))    
    Fvalue = MSR/(a * MSC + b * MSE)

    alpha = 1 - 0.95
    a = (nr * coeff)/(ns * (1 - coeff))
    b = 1 + (nr * coeff * (ns - 1))/(ns * (1 - coeff))
    v = (a * MSC + b * MSE)**2/((a * MSC)**2/(nr - 1) + (b * MSE)**2/((ns - 1) * (nr - 1)))
    #p.value <- pf(Fvalue, df1, df2, lower.tail = FALSE)
    #FL <- qf(1 - alpha/2, ns - 1, v)
    #FU <- qf(1 - alpha/2, v, ns - 1)

    FL = sstats.f.ppf(1 - alpha/2, ns - 1, v)
    FU = sstats.f.ppf(1 - alpha/2, v, ns - 1)

    lbound = (ns * (MSR - FL * MSE))/(FL * (nr * MSC + (nr * ns - nr - ns) * MSE) + ns * MSR)
    ubound = (ns * (FU * MSR - MSE))/(nr * MSC + (nr * ns - nr - ns) * MSE + ns * FU * MSR)
    
    return icc, Fvalue, lbound, ubound

if __name__ == '__main__':
	data = sio.loadmat('./data.mat')
	data = data['data']

	Ny,Nx,Npts,Nraters,Nreps = data.shape

	area_icc = np.zeros([Nreps,1])
	shape_icc = np.zeros([Nreps,1])

	# calculate shape and area ICC for each repetition of the simulated experiment
	for rr in np.arange(0,Nreps):    
	    print('Analyzing rep {} of {}\n'.format(rr,Nreps))    
	    
	    measured_shapes = data[:,:,:,:,rr]
	    measured_areas = np.squeeze(np.sum(np.sum(measured_shapes,0),0))
	    
	    area_icc[rr] = icc(measured_areas)
	    shape_icc[rr] = shape_icc(measured_shapes) # shape ICC for all shapes for this rep
