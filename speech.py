import numpy as np
from matplotlib import pyplot as plt
import pickle


# Read in pickled data:
f = open('./data/speech.dat') 
data = np.load(f)


# put pickled data in useable form
# print data.keys()
testdata = data.get('test')
traindata = data.get('train')
keyList = testdata.keys()

testdataList = []
traindataList = []
for a in keyList:
	testdataList.append(testdata.get(a))
	traindataList.append(traindata.get(a))

#print np.asarray(((testdataList)[0])[0]).shape
# shape is 5, 34, (5,16)
# with dimentions = 16 and number of readings 34 But not for all data... what is the 1st dimention of the data?

