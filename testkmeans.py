from kmeans import Kmeans 
import numpy as np
data = np.array([[1.9,1.5,0.4,0.4,0.1,0.2,2.0,0.3,0.1],[2.3,2.5,0.2,1.8,0.1,1.8,2.5,1.5,0.3]])
codes = 3
km = Kmeans(data,codes)
print 'Class labels = ', km.label
print('Due to the random initialization, different (wrong) labels\nare often returned')    
x = np.array([0.25,2.0])
print km.classify(x)
#km.label = lab2
km.plot()
print('Verify the answer using the graph.')
# Specify the initial cluster means.        
codes =  np.array([data[:,0],data[:,2],data[:,3]]).T 
km = Kmeans(data,codes)  
print 'Clusters = ',km.cluster
print 'Class labels = ', km.label

