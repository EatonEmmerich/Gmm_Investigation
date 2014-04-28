import numpy as np
from matplotlib import pyplot as plt
from gmm import Gmm

def loadsignatures(signnum = "sign1"):
	"""
	"""
	import glob
	list_of_files = glob.glob("data/signatures/" + signnum+"/*")
	sign = []
	for a in list_of_files:
		print a
		sign.append(np.loadtxt(a,comments = '%'))
	return sign

def dogmmsig(direct):
	sign = loadsignatures(direct)
	# normalize data:
	# normalize by position
	n = len(sign)
	print n
	d1,d2 = sign[0].shape
	mean = np.zeros((n,d2))
	for i in range(n):
		for j in range (d1):
			mean[i] = sign[i][j,:] + mean[i]
		mean[i] = mean[i]/d1

	for i in range(n):
		sign[i][:,:] = sign[i][:,:] - mean[i]
#		print sign[0][:,0:2].shape

		# stack data on each other
	for i in range(n):
		sign[i] = sign[i].T

	sign = np.column_stack(sign)
	for ki in range (1,6):
		g = Gmm(sign[0:2,:],k = ki)
		print "done with k= ",ki
	return g

directList = ["sign1","sign2","sign3","sign4","sign5"]
gms = []
for d in directList:
	gms.append(dogmmsig(d))

