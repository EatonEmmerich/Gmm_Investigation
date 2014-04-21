""" K-means algorithm. 
@since: 2014

@author: Ben        
"""

import numpy as np
from scipy.cluster.vq import kmeans2
from matplotlib import pyplot as plt 



class Kmeans(object):
    """ 
    Classification using the K-means algorithm. Based on the scipy algorithm, kmeans2. 
    
            
    The K-means algorithm is an unservised classification algorithm. 
    n, d-dimensional observations, and the number, k,  of classes are provided. 
    If the number of classes are provided then initial class means are randomly 
    selected from the data. Else, the algorithm is initialized with the given 
    class means.
    
    
    Parameters
    ----------

    data : (d,n) ndarray 
        n, d-dimensional observations 
    codes : int, or (d,k) ndarray 
        The number of clusters to form as well as the number of
        centroids to generate. If `minit` initialization string is
        'matrix', or if a ndarray is given instead, it is
        interpreted as initial cluster to use instead.
    it : int
        Number of iterations of the k-means algrithm to run.
    minit : string
        Method for initialization. Available methods are 'random',
        'points',  and 'matrix':

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        'points': choose k observations (rows) at random from data for
        the initial centroids.
        
        'matrix': interpret the codes parameter as an (d,k) ndarray (or length k
            array for one-dimensional data) array of initial centroids.
    
        
            
    Attributes:
    -----------
    k : int  
        The number of classes
    cluster : (d,k) ndarray
        The k, d-dimensional cluster means
    label : (n,) ndarray
        The class label of each observation in the training set
        
    Methods:
    -------
    plot :      
        Display the classes, only 2d observations allowed
    classify:
        Returns the class label of observation x

    Examples
    --------
    >>># Random initialization
    >>>from kmeans import Kmeans 
    >>>data = np.array([[1.9,1.5,0.4,0.4,0.1,0.2,2.0,0.3,0.1],
                          [2.3,2.5,0.2,1.8,0.1,1.8,2.5,1.5,0.3]])
    >>>codes = 3
    >>>km = Kmeans(data,codes)
    >>>print 'Class labels = ', km.label
    Class labels =  [1 1 0 2 0 2 1 2 0]
    >>>print('Due to the random initialization, different (wrong) labels
                                  are often returned')    
    >>>km.plot()  
    >>>x = np.array([0.25,2.0])
    >>>km.classify(x)
    >>>print('Verify the answer using the graph.')
    
 
    >>># Specify the initial cluster means.        
    >>>codes =  np.array([data[:,0],data[:,2],data[:,3]]).T 
    >>>km = Kmeans(data,codes)  
    >>>print 'Clusters = ',km.cluster
    Clusters =  [[ 1.8  0.2  0.3 ]
               [ 2.43333333  0.2  1.7]]
    >>>print 'Class labels = ', km.label
    Class labels =  [0 0 1 2 1 2 0 2 1]
    >>>km.plot()
   
           
    """
    def __init__(self, data, codes=3, it=10,  minit='points'):
              
        d,n = data.shape
        
        if type(codes) is int:
            k = codes
        elif not type(codes) is int:
            k = codes.shape[1]
            codes = codes.T
            
        cluster,label = kmeans2(data.T, codes, it,  minit)
        
        self.data     = data
        self.cluster  = cluster.T
        self.label    = label
        self.k        = k
    
    @property    
    def get_means(self):
        """
        Return
        ------
        cluster : (d,k) ndarray
            The k, d-dimensional cluster means
        """
        
        return self.cluster
        
    @property
    def get_labels():
        """
        Return
        ------
        label : (n,) array
            The class labels for each of the observations.        
        """
        
        return self.label
        
    def classify(self,x):
        """ Calculate the class label of the given vector x.
            Parameters
            ----------
            x : (d,) array 
                The observation that needs to be classified.
            
            Return
            ------
            label : int
                class label, one of k classes,  0 through k-1
        """
        cluster = self.cluster
        # Calculate labels
        # insert code
	
	#get length to each mean.
	#print "distances"
	#print x.shape
	#print cluster.shape
	a,b = cluster.shape
	distances = np.zeros((a,b));
    	for temp in range(b):
		distances[:,temp] = x - cluster[:,temp]
	#print distances
	probability = np.add.reduce(distances ** 2,axis=0)
	#print probability
	#get label classifications
	maxnumber = 0
	maxamount = 0
	
	for temp in range (b):
		if(probability[temp]>maxamount):
			maxamount = probability[temp]
			maxnumber = temp
	label = maxnumber
        return label
        
    def plot(self):
        """Plot the data, as well as the clusters
           Note that the data needs to be two dimensional
        """
        dat     = self.data
        cluster = self.cluster
        label   = self.label
        k       = self.k

        d,n  = dat.shape
        # Consistency check
        if not d==2:
            raise ValueError('The data is not two dimensional')
       

        plt.hold('on')
   
        # Plot the means     
        plt.plot(cluster[0,:],cluster[1,:],'rx',ms=10.)
            

        # Color the different classes
        
        for j in range(k):
            index = (label==j)
            cl    = dat[:,index]
            plt.plot(cl[0,:],cl[1,:],'o',ms=10.)    
        plt.show()   
