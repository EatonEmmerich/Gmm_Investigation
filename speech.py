import numpy as np
from matplotlib import pyplot as plt
import pickle
from utils import confusion as conv
from gmm import Gmm


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

def normalize(tstack,trstack):
	"""
	"""
	for i in range (len(tstack)):
		tstackS = np.column_stack(tstack[i])
		mean = np.mean(tstackS,axis = 1)
		for j in range(len(tstack[i])):
			for k in range ((np.shape(tstack[i][j])[1])):
				tstack[i][j][:,k] = tstack[i][j][:,k]-mean
	
	for i in range (len(tstack)):
                trstackS = np.column_stack(trstack[i])
                mean = np.mean(trstackS,axis = 1)
                for j in range(len(trstack[i])):
                        for k in range ((np.shape(trstack[i][j])[1])):
                                trstack[i][j][:,k] = trstack[i][j][:,k]-mean


	return tstack,trstack

# Read in pickled data:
f = open('./data/speech.dat') 
data = np.load(f)


# put pickled data in useable form
# print data.keys()
testdata = data.get('test')
traindata = data.get('train')
keyList = testdata.keys()
testdataStack = []
traindataStack = []
for a in keyList:
	testdataList = []
	traindataList = []
	for b in range(len(testdata.get(a))):
		testdataList.append((testdata.get(a)[b].T))
	for b in range(len(traindata.get(a))):
		traindataList.append((traindata.get(a)[b].T))
	testdataStack.append((testdataList))
	traindataStack.append((traindataList))
testdataStack,traindataStack = normalize(testdataStack, traindataStack)
# shape is 5, 34, (5,16)
# with dimentions : 16 and number of readings : nx34
for kin in range (1,2):
	g = []
	for i in range (len(keyList)):
		g.append(Gmm(np.column_stack(traindataStack[i]),k = kin))


	# get the loglikelihood
	resultList = []
	original = []
	for i in range (len(keyList)):
		result = []
		for j in range (len(testdataStack[i])):
			resulttemp = np.zeros((len(keyList)))
			for k in range (len(keyList)):
				resulttemp[k] = (g[k].loglikelihood(testdataStack[i][j][:,:]))
			result.append(resulttemp)
			original.append(keyList[i])
		resultList.append(result)

	classified = mostlikely(resultList,keyList)
	conv(original,classified)
	print "completed for k = ", kin
