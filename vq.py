import numpy as np
from kmeans import Kmeans as km
import matplotlib.pyplot as plt

def loadimages():
    import matplotlib.pyplot as plt
    import os
    import fnmatch
    """
    Load all the gray scale images in all the subdirectories with suffix `png`.
    The images are flattened and each image is represented as an (d,) array.
    
    Return
    ------
    
    images : (d,n,i) ndarray
       returns n, d-dimensional images. with i dimensional color data
    """
    matches = []
    for root, dirs, files in os.walk("./data"):
        for filename in fnmatch.filter(files, '*.png'):
            matches.append(os.path.join(root, filename))
    m = np.shape(matches)
    data = []
    for m in matches:
        data.append(plt.imread(m))
		
    return data
def clustercolors(X, labels, means):
	"""
	Put colors into the means given
	Parameters
	----------
	X : (n,d) ndarray
	labels : (d,) ndarray
	means : (n,k) ndarray

	Return
	------
	clustereddata (d,n) ndarray with colored by means data
	"""
	n,d = np.shape(X)
	nm,k = np.shape(means)
	clustereddata = np.zeros((n,d))
#	print clustereddata[:,labels == 0].shape
	print means[:,0]
	for temp in range(k):
#		clustereddata[:,labels == temp] = means[:,temp].flatten()
		print np.where(labels == temp)
		ind = np.where(labels == temp)
		for temp2 in range(n):
			np.put(clustereddata[temp2,:],ind,means[temp2,temp])

	return clustereddata


# import picture
X = loadimages()
# picture size: 400 x 267 pixels rgb colours
#print np.shape(X)
plt.imshow(X[0])
rgb = 3
# plt.show()
#print np.shape(X[0].flatten())
# put image through kmeans classification
height, width, three = np.shape(X[0])
data = X[0].reshape(width*height,rgb).T
np.shape(data)
kmobj = []
plt.show()
for k in range(2,12):
	kmobj.append(km(data,k))

for k in range(10):
	clusters = kmobj[k].get_means
	labels = kmobj[k].label
	newdat = clustercolors(data, labels, clusters)
	plt.figure()
	plt.imshow(newdat.T.reshape(height,width,rgb))
	plt.show()
