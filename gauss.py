'''Module containing a DensityFunc abstract class, with common probability densities

@since: 2014

@author: Ben
'''

from __future__ import division

import numpy as np


class Gauss(object):
    '''
    Class for representing a multi-dimensional Gaussian distribution of dimension d, 
    given mean and covariance.
    The covariance matrix has to be  positive definite and non-singular.
    
    Parameters
    ----------
    
    mean : (d,) ndarray
       mean of the distribution
    cov  : (d,d) ndarray
       Covariance matrix. 
    
    Methods
    -------
    
    f 
       Returns the value of the density function
    logf
       Returns the log of the density function
    likelihood
       Returns the likelihood of the data
    loglik
       Reurns the log-likelihood of the data
    sample
       Returns samples drawn from the normal distribution with the given
       mean and covariance
    
    
    Example
    -------
    >>> from gauss import Gauss
    >>> # Scalar example
    >>> mean = [10.]
    >>> cov  = [[1.]]
    >>> ga   = Gauss(mean,cov)
    >>> ga.f([10.])
        0.398942280401        
    >>> x = np.array([[10.,10.,10.]])
    >>> ga.likelihood(x)
        0.0634936359342
    >>> # Multivariate example
    >>> mean = [10.0, 10.0]
    >>> cov  = [[  1.   0.],[  0.  10.]]
    >>> ga   = Gauss(mean,cov)
    >>> ga.f(np.array([10.,10.])
           0.050329212104487035
    >>> x = np.array([[10.,10.,10.,10.],[10.,10.,10.,10.]])
    >>> ga.likelihood(x)
           336.4162389091777101e-06
    
    '''
    def __init__(self, mean=[0.,0.], cov=[[1.,0.],[0.,1.]]):
        
        
        mean = np.array(mean); cov = np.array(cov)
        d,n = cov.shape
        
        self._dim = d
        self._mean = mean.flatten()
        self._cov = cov
        self._covdet = np.linalg.det(2.0*np.pi*cov)
        
        if self._covdet < 10e-25:
            raise ValueError('The covariance matrix is singular.')
        
            
    def f(self, x):
        '''
        Calculate the value of the normal distributions at x
        
        Parameters
        ----------
        x : (d,) ndarray
           Evaluate a single d-dimensional samples x
           
        Returns
        -------
        val : scalar
           The value of the normal distribution at x.
        
        '''
        
        return np.exp(self.logf(x))
    
    def logf(self, x):
        '''
        Calculate  the log-density at x
        
        Parameters
        ----------
        x : (d,) ndarray
           Evaluate the log-normal distribution at a single d-dimensional 
           sample x
           
        Returns
        -------
        val : scalar
           The value of the log of the normal distribution at x.
        '''
        if not len(x.shape)==1:
            raise ValueError('The input vector needs to be of shape (d,)')
        trans = x - self._mean
        mal   = -trans.dot(np.linalg.solve(self._cov,trans))/2.
        return -0.5*np.log(self._covdet) + mal


    def likelihood(self, x):
        '''
        Calculates the likelihood of the data set x for the normal
        distribution.
        
        Parameters
        ----------
        x :  (d,n) ndarray
           Calculate the likelihood of n, d-dimensional samples
           
        Returns
        -------
        val : scalar
           The likelihood value   
        '''
        return np.exp(self.loglik(x))

    def loglik(self, x):
        '''
        Calculates  the log-likelihood of the data set x for the normal 
        distribution.
        
        Parameters
        ----------
        x :  (d,n) ndarray
           Calculate the likelihood of n, d-dimensional samples
           
        Returns
        -------
        val : scalar
           The log-likelihood value
        '''
        return np.sum(np.apply_along_axis(self.logf, 0, x))


    def sample(self, n=1):
        '''
        Calculates n independent points sampled from the normal distribution
        
        Parameters
        ----------
        n : int
           The number of samples
           
        Returns
        -------
        samples : (d,n) ndarray
           n, d-dimensional samples
        
        '''

        return np.random.multivariate_normal(self._mean, self._cov, n).T
    


       
        

    
