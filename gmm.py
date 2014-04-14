""" Gaussian Mixture Models, trained with the EM algorithm. 
@since: 2014

@author: Ben      
"""

import numpy as np
from scipy.cluster.vq import kmeans2
from gauss import Gauss
from  kmeans import Kmeans
from matplotlib import pyplot as plt

class Gmm(object):
    """ 
    Calculate a Gaussian Mixture Model from partially observed data. 
    Can also be used for classification assuming Gaussian class densities.
 
    Parameters
    ----------

    data : (d,n) ndarray. 
        n, d-dimensional observations
    k : int 
        The number of mixture components. 
    init : char
        The method that will be used for initializing the GMM.
        Available methods:
            init = 'split'
                Use binary split
            init = 'kmeans'
                Use standard kmeans, random initialization
    plot : boolean
        plt = True, turns on a plotting routine to illustrate the 
        binary split algorithm. Can only be used for 2-dimensional data.
    
              
    Properties:
    -----------
    get_gmm_params : property
        Returns the gmm parameters: mixmean, mixcoef, mixcov
        
    Methods:
    -------        
    soft_assign
        Calculates the probabilities of a single observation x for the
        k `classes` of the gmm.
    f 
        Evaluales gmm at a single observation x.
    logf
        Returns the log of gmm value as a single observation x.
    loglikelihood
        Calcluates the log-likelihood of n observations
    get_gmm_params : property
        Returns the gmm parameters: mixmean, mixcoef, mixcov
            
    Example:
    --------
    >>>import numpy as np
    >>>from scipy.cluster.vq import kmeans2
    >>>from gauss import Gauss
    >>>from  kmeans import Kmeans
    >>>from gmm import Gmm
    >>>dat1 = np.array([[0.,1,1,0],[0,0,1,1]])
    >>>dat2 = np.array([[0.,1,1,0],[4,4,5,5]])
    >>>data = np.hstack((dat1,dat2))
    >>>     # Binary split initialization
    >>>g = Gmm(data,k=2)
    >>>mixmean, mixcoef, mixcov = g.get_gmm_params
    >>>print 'mixcoef :',mixcoef
       mixcoef : [ 0.5  0.5]
    >>>print 'means :',mixmean
       means : [[ 0.5  0.5]
        [ 0.5  4.5]]    
    >>>print 'cov :',mixcov
       cov : [[[ 0.25  0.  ]
         [ 0.    0.25]]
        [[ 0.25  0.  ]
         [ 0.    0.25]]]
    >>>prob = g.soft_assign(data)
    >>>print 'probability :',prob
       probability : [[  1.00000000e+00   1.00000000e+00   1.00000000e+00   1.00000000e+00
           3.77513463e-11   3.77513463e-11   4.24835441e-18   4.24835441e-18]
        [  4.24835441e-18   4.24835441e-18   3.77513463e-11   3.77513463e-11
           1.00000000e+00   1.00000000e+00   1.00000000e+00   1.00000000e+00]]
    >>>val  = g.f(data[:,1])
    >>>print 'Value at x :',val
       Value at x : 0.117099663049
    >>>lik  = g.loglikelihood(data)
    >>>print 'Data log-likelihood : ',lik
       Data log-likelihood :  -17.1578390866           
    >>>###################################################################### 
    >>>#             Random initialization using kmeans.      
    >>>g = Gmm(data,k=2,init='kmeans')
    >>>mixmean, mixcoef, mixcov = g.get_gmm_params
    >>>print 'mixcoef :',mixcoef
       mixcoef : [ 0.5  0.5]
    >>>print 'means :',mixmean
       means : [[ 0.5  0.5]
        [ 0.5  4.5]]
    >>>print 'cov :',mixcov
       cov : [[[ 0.25  0.  ]
         [ 0.    0.25]]
        [[ 0.25  0.  ]
         [ 0.    0.25]]]
    """
    def __init__(self, data, k=None, init='split', plot=False):
        """ 
        Initializes the EM algorithm using binary split or the k-means 
        algorithm.                        
        """
        # Sanity check
        if not init == 'kmeans' and not init == 'split':
            raise ValueError('The requested initialization not supported.')
        if k == None:
            raise ValueError('Specify the number of mixture components.')
        
        # Data dimensions
        d,n = data.shape
        
        if not d == 2 and plt==True:
            print 'Warning: Plotting only allowed for 2d data. Ignored.'
            plt = False
        
        self.data = data
        self.k    = k
          
        # Find initial clusters
        if init == 'kmeans': mixmean, label = self._init_kmeans()
        if init == 'split' : mixmean, label = self._init_split(plot)
        
        
        # Calculate the initial mixture covariances, and mixture coefficients
        # from the clusters returned by the initialization


        mixcov       = np.zeros((k,d,d))
        mixcoef   = np.zeros((k,))
        for i in range(k):
            clus        = data[:,label==i]
            d,ni        = clus.shape
            mixcov[i]   = np.cov(clus,bias=1)
            mixcoef[i]  = ni
        mixcoef = mixcoef/float(n)
        
        # Pass the initial estimates to em. Iterate to convergence
        mixmean, mixcoef, mixcov, gam = self._em(mixmean,mixcoef,mixcov,
                                                    tol=1.0e-4,max_iter=50)
        
        # Pass mixture parameters to class
        self.mixmean = mixmean
        self.mixcoef = mixcoef
        self.mixcov  = mixcov
        self.gam     = gam
       
    def _init_kmeans(self):
        """
        Initialize using k-means.
        Uses random intialization for k-means. This is a really bad idea.
        """
        data = self.data
        k    = self.k
        # Estimate the means of the mixture components, using k-means
        km      = Kmeans(data,k)
           
        return km.cluster.T, km.label


    def _init_split(self,plot):
        """
        Initialization using binary split.
        Choose two observations at random. These are seeds for 
        k-means clustering.
        Identify the cluster with the largest scatter. Replace means of this 
        cluster with two random observations from this cluster. Run k-means using 
        the rest of the means as well as the two newly selected observations as 
        seeds. 
        Select the cluster with the largest scatter and repeat until the correct 
        number of clusters is obtained.  
        """
        
        data   = self.data
        k      = self.k
        d,n    = data.shape
        labels = np.zeros((n,), dtype=int)
        
        
        #clus_max_scatter = data
        #cluster = np.mean(data,axis=1)[:,None]
        
        # Select two initial means at random
        ini           = np.random.random_integers(0,n-1,2)  
    
        init_means    = np.hstack((data[:,ini[0]][:,None],
                                   data[:,ini[1]][:,None])) 
        cluster,label = kmeans2(data.T,init_means.T,iter=20,minit='matrix')
        cluster       = cluster.T
            

        for j in range(k-2):
            # Find cluster with maximal scatter
            scat = 0.
            for i in range(j+2):
                clusi     = data[:,label==i]
                meani     = np.mean(clusi,axis=1)
                clus0mean = clusi - meani[:,None]
                scati     = np.linalg.norm(clus0mean)
                if scati  > scat:
                    scat   = scati
                    clscat = clusi
                    remove = i
            # Split the cluster with maximal scattering
            d,n   = clscat.shape
            
            
            ini       = np.random.random_integers(0,n-1,2) # Two random  means 
            ini_means = np.hstack([clscat[:,ini[0]][:,None],
                                              clscat[:,ini[1]][:,None]])      
            # sca_means,sca_labels = kmeans2(cluscat.T, ini_means.T, iter=10, minit='matrix')

            cluster  = np.hstack([cluster,ini_means])
            cluster  = np.delete(cluster,remove,axis=1)

            # Run 20 steps of K-means algorithm to update all the means
            cluster,label  = kmeans2(data.T,cluster.T,iter=20,minit='matrix')
            cluster        = cluster.T

            # Call this function if you want to see how the algorithm splits the clusters
            if plot == True: self._plot_binsplit(clscat,ini_means)                 

        if plot == True:  
            # Different classes cycle through the following
            # colors, if k > 7
            col_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']          
            for i in range(k):
                col = col_list[i%7]
                plt.plot(data[0,label==i],data[1,label==i],'.',color=col)
                plt.plot(cluster[0,i],cluster[1,i],'dk',ms=10)
                plt.title('The final assignment of the clusters.')
            plt.show()
                     
        return cluster, label
                      
                
 
    def _em(self,mixmean0,mixcoef0,mixcov0,tol=1.0e-4,max_iter=50):
        """
        Do the EM iteration for Gaussian mixture models.
        
        Parameters
        ----------
        mixmean : (d,k) ndarray
            The initial estimate of the means of the k mixture components
        mixcoef : (k,) ndarray
            The initial estimate of the k mixture coefficients
        mixcov : (k,d,d) ndarray
            The covariances of the k mixture components 
        tol : float scalar
            Error tolerance. Iterates until the relative error is smaller than 
            tol, or the maximum number of iteration exceeded.
        iter :  int
            Maximum number of iterations. Iteration stops if the `iter` is 
            reached before convergence  
            
        Return
        ------
        mixmean : (d,k) ndarray
            The means of the k mixture components        
        mixcoef : (k,) ndarray
            The coefficient for each mixture component  
        mixcov: (k,d,d) ndarray
            The covariance matrices of the k mixture components
        gam :  (k,n) ndarray
            The responsibilities of data point x_n for class k.                       
        """

        err     = 1.0e+10
        it      = 0
        while (it < iter) and (err > tol):
            # Insert code

        if (it >= iter):
            raise ValueError('Maximum number of iterations exceeded')
            
        return mixmean, mixcoef, mixcov, gam
            
    def _respon(self,mixmean,mixcoef,mixcov):
        """
        Calculate the responsibilities of each data point for each class k.
        
        Parameters
        ----------   
        mixmean   : (d,k) ndarray
            The means of the k mixture components        
        mixcoef : (d,) array
            The mixture coefficient for each component 
        mixcov : (k,d,d) ndarray
            The covariance matrices of the k mixture components
            
        Return
        ------
        gam : (k,n) ndarray
            The responsibilities of data point x_n to the mixture components k
        """

        data = self.data
        d,n  = data.shape
        k    = self.k
        # Insert code
   
        return gam

    def _params(self,gam):
        """
        Estimate the means, covariances, and the mixture coefficients, 
        given the responsibilities.
        
        Parameters
        ----------        
        gam : (k,n) ndarray
            Responsibility of data point n to mixture component k
            
        Return
        ------            
        mixmean : (d,k) ndarray
            The means of the k mixture components        
        mixcoeff : (k,) array
            The mixture coefficients for the k components 
        mixcov: (k,d,d) ndarray
            Covariance matrices  of the k mixture components
        """             
        data = self.data
        d,n  = data.shape
        k    = self.k

        nk   = np.sum(gam,axis=1)
        
        mixmean    = (gam.dot(data.T)/nk[:,None]).T
        mixcov     = np.zeros((k,d,d))
        mixcoef = nk/n
        
        
        for i in range(k):
            # Insert code
            
        return mixmean, mixcoef, mixcov


    def _error(self,mixmean, mixmean0, 
                    mixcoef=None, mixcoef0 = None,
                    mixcov=None, mixcov0 = None):
        """
        Calculate the error in the current estimate.
        
        Note
        ----
        At the moment only convergence of the means are tested. If this does 
        not work, try a more sophisticated, involving cov and/or mixcoef.
        
        Parameters
        ----------   
        mixmean : (d,k) ndarray
            The k, d-dimensional means of the mixture components.
            The latest iterate. 
        mixmean0 : (d,k) ndarray
            The k, d-dimensional means of the mixture components.
            The previous iterate.
        mixcov : (k,d,d) ndarray    (not implemented)
            The k, covariances of the mixture components 
            The latest iterate. 
        mixcov0 : (k,d,d) ndarray   (not implemented)
            The k, covariances of the mixture components (not implemented)
            The previous iterate.
        mixcoef : (k,) ndarray  
            The mixture coeffiecients. 
            The latest iterate.
        mixcoef0 : (k,) ndarray  (not implemented)
            The mixture coeffiecients 
            The previous iterate. (not implemented)
              
        Return
        ------
        err : float   
            Estimate of the current relative error. 
        """

        err  = np.linalg.norm(mixmean-mixmean0)/np.linalg.norm(mixmean)

        return err
                     

        
    def _plot_binsplit(self,clus_max_scatter,init_means,):
        """ 
        Visualize the binary split algorithm for 2-dimensional data.
        Call this function from init_split.  
        
        Parameters
        ----------
        clus_max_scatter : (2,m) ndarray
            The cluster with maximum scatter, that wil be split.
        init_means : (2,2)
            The means according to how the class is split.
        clus_max_means : (2,)
            The current estimate of the cluster with max scatter.
            
        Note
        ----
        Red dots:  denote the cluster to be split. 
        Black dots: shows the whole data set.
        Black squares: seeds for the selected cluster to be split.
        Red diamonds:  k-means estimated cluster means
        """
        data = self.data
        clus_max_mean = np.mean(clus_max_scatter,axis=1)
        
        plt.figure()
        plt.hold('on')
        plt.plot(data[0,:],data[1,:],'.k')
        plt.plot(clus_max_scatter[0,:],clus_max_scatter[1,:],'.r')
        cc = init_means
        plt.plot([cc[0,:]],[cc[1,:]],'ks',ms=10.)       

        cc = clus_max_mean
        plt.plot([cc[0]],[cc[1]],'rD',ms=10.)
        plt.title('Split')  
        plt.show()

    def soft_assign(self,x,mixmean=None,mixcoef=None, mixcov=None):
        """ 
        Calculate the probability of x for each of the k classes.            
        Typically the sample is assigned to class n with n = arg max(resp).
        
        Parameters
        ----------
        x : (d,n) ndarray
            The observations that need to be classified.
        mixmeans : (d,k), ndarray, or None
            The means of the k mixture components.
            If `None` use the values of the class.
        mixcoef : (k,) ndarray, or None
            The k mixture coefficients
            If `None` use the values of  the class.
        mixcov : (k,d,d), ndarray, or None
            The covariances of the k mixture components
            If `None` use the values of  the class.
        
                
        Return
        ------
        prob : (k,) ndarray
            The probabilities for each of the k classes.
        """ 
        if len(x.shape)==1: x = x[:,None]
        d,n = x.shape
        
        # Get the necessary parameters from the class
        mixmean = self.mixmean
        mixcoef = self.mixcoef
        mixcov  = self.mixcov

        k = len(mixcoef)
        
        
        prob    = np.zeros((k,n))  
        for j in range(n):  
            for i in range(k):  
                g       = Gauss(mixmean[:,i],mixcov[i])                     
                prob[i,j] = g.f(x[:,j])*mixcoef[i]
        
        return prob/np.sum(prob,axis=0)[None,:]


        
    def loglikelihood(self, x, mixmean=None, mixcoef=None, mixcov=None):
        """ 
        Calculate the data log-likelihood. 
    
        Parameters
        ----------
        x : (d,n), ndarray
            n, d-dimensional data for which the likelihood is required
        mixmean : (d,k), ndarray, or None
            The means of the k mixture components
            If `None` use the values of  the class.
        mixcoef : (k,) ndarray, or None
            The k mixture coefficients
            If `None` use the values of  the class.        
        mixcov : (k,d,d), ndarray, or None
            The covariances of the k mixture components.
            If `None` use the values of  the class.
        
        Return
        ------
        lik : float
            The likelihood of the data.        
        """ 
        
        if mixmean == None: mixmean = self.mixmean
        if mixcoef == None: mixcoef = self.mixcoef
        if mixcoef == None: mixcov  = self.mixcov

        d,n  = x.shape
        k    = len(mixcoef)
            
        return np.sum(np.apply_along_axis(self.logf, 0, x))
            
    
    def f(self,x,mixmean=None, mixcoef=None, mixcov=None):
        """ 
        Evaluate the gmm at x. 
    
        Parameters
        ----------
        x : (d,) ndarray
            A single d-dimensional observation 
        mixmean : (d,k), ndarray, or None
            The means of the k mixture components
            If `None` use the values of  the class.
        mixcoef : (k,) ndarray, or None
            The k mixture coefficients
            If `None` use the values of  the class.        
        mixcov : (k,d,d), ndarray, or None
            The covariances of the k mixture components.
            If `None` use the values of  the class.
        
        Return
        ------
        val : float
            The value of the gmm at given x.      
        """ 
        
        if mixmean == None: mixmean = self.mixmean
        if mixcoef == None: mixcoef = self.mixcoef
        if mixcov  == None: mixcov  = self.mixcov

        k    = len(mixcoef)
        
        comp = np.zeros((k,))
        for j in range(k):
            g       = Gauss(mixmean[:,j],mixcov[j])
            comp[j] = g.f(x)
        return mixcoef.dot(comp)
        
    def logf(self,x,mixmean=None, mixcoef=None, mixcov=None):
        """ 
        Evaluate the log of the  gmm at x. 
    
        Parameters
        ----------
        x : (d,) ndarray
            A single d-dimensional observation 
        mixmean : (d,k), ndarray, or None
            The means of the k mixture components
            If `None` use the values of  the class.
        mixcoef : (k,) ndarray, or None
            The k mixture coefficients
            If `None` use the values of  the class.        
        mixcov : (k,d,d), ndarray, or None
            The covariances of the k mixture components.
            If `None` use the values of  the class.
        
        Return
        ------
        logval : float
            The log of the value of the gmm at given x.      
        """
        
        return np.log(self.f(x,mixmean, mixcoef, mixcov))
        
    @property
    def get_gmm_params(self):
        """ 
        Returns the mixture coefficients of the gmm.
        
        Return
        ------
        mixmean : (d,k), ndarray
            The means of the k mixture components.
        mixcoef : (k,) ndarray
            The k mixture coefficients.       
        mixcov : (k,d,d), ndarray
            The covariances of the k mixture components.
        """
        
        return self.mixmean, self.mixcoef, self.mixcov     
    

