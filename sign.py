import numpy as np
from matplotlib import pyplot as plt
from gmm import Gmm
from utils import confusion as conv

def loadsignatures(signnum = "sign1"):
	"""
	"""
	import glob
	list_of_files = glob.glob("data/signatures/" + signnum+"/*")
	sign = []
	for a in list_of_files:
		sign.append(np.loadtxt(a,comments = '%'))
	return sign

def dogmmsig(direct, trainnum = 5, ki = 1):
	sign = loadsignatures(direct)
	# normalize data:
	# normalize by position
	n = len(sign)
	meanList = []
	for i in range(n):
		d1,d2 = sign[i].shape
		mean = np.zeros((d2))
		for j in range (d1):
			mean = sign[i][j,:] + mean
		mean = mean/d1
		meanList.append(mean)

	for i in range(n):
		sign[i][:,:] = sign[i][:,:] - meanList[i]
#		print sign[0][:,0:2].shape

		# stack data on each other
	for i in range(n):
		sign[i] = sign[i].T
	traindata = np.column_stack(sign[:trainnum])
	testdata = sign[trainnum:]
	g = Gmm(traindata[:,:],k = ki)
	return g, testdata

def mostlikely(resultList,directList):
	"""
	resultList : [d][n] Float
	directList : [d] String
	-----------------------
	labelsList : [n*d] String
	"""
	labelsList = []
	for i in range(len(resultList)):
		for j in range(len(resultList[i])):
			maxpos = np.argmax(resultList[i][j])
			labelsList.append(directList[maxpos])
	return labelsList
	
	

directList = ["sign1","sign2","sign3","sign4","sign5"]
for kin in range(1,6):
	gms = []
	testdata = []
	for d in directList:
	#d = directList[1]
		model, test = dogmmsig(d,ki = kin)
		gms.append(model)
		testdata.append(test)
	
	# Test Data on model:
	resultList = []
	original = []
	for d in range(len(directList)):
		result = []
		for a in range(len(testdata[d])):
			resulttemp = np.zeros(len(directList))
			for d2 in range(len(directList)):
				resulttemp[d2] = (gms[d2].loglikelihood(testdata[d][a][:,:]))
	#			print result[a]
			result.append(resulttemp)
			original.append(directList[d])
		resultList.append(result)

	# classify most likely model
	classified = mostlikely(resultList,directList)
	#print classified
	#print original
	conv(original,classified)
	print "done with k = ", kin
