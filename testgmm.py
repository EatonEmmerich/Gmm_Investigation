import numpy as np
from scipy.cluster.vq import kmeans2
from gauss import Gauss
from  kmeans import Kmeans
from gmm import Gmm
dat1 = np.array([[0.,1,1,0],[0,0,1,1]])
dat2 = np.array([[0.,1,1,0],[4,4,5,5]])
data = np.hstack((dat1,dat2))
     # Binary split initialization
g = Gmm(data,k=2)
mixmean, mixcoef, mixcov = g.get_gmm_params
print 'mixcoef :',mixcoef
#       mixcoef : [ 0.5  0.5]
print 'means :',mixmean
#       means : [[ 0.5  0.5]
#        [ 0.5  4.5]]    
print 'cov :',mixcov
#       cov : [[[ 0.25  0.  ]
##         [ 0.    0.25]]
#        [[ 0.25  0.  ]
#         [ 0.    0.25]]]
prob = g.soft_assign(data)
print 'probability :',prob
#       probability : [[  1.00000000e+00   1.00000000e+00   1.00000000e+00   1.00000000e+00
#           3.77513463e-11   3.77513463e-11   4.24835441e-18   4.24835441e-18]
#        [  4.24835441e-18   4.24835441e-18   3.77513463e-11   3.77513463e-11
#           1.00000000e+00   1.00000000e+00   1.00000000e+00   1.00000000e+00]]
val  = g.f(data[:,1])
print 'Value at x :',val
#       Value at x : 0.117099663049
#lik  = g.loglikelihood(data)
print 'Data log-likelihood : ',lik
#       Data log-likelihood :  -17.1578390866           
###################################################################### 
#             Random initialization using kmeans.      
g = Gmm(data,k=2,init='kmeans')
mixmean, mixcoef, mixcov = g.get_gmm_params
print 'mixcoef :',mixcoef
print 'means :',mixmean
print 'cov :',mixcov

