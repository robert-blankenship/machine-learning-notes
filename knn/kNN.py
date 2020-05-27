import numpy
import operator


def createDataSet():
	group = numpy.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = numpy.tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDistIndices = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = numpy.zeros(numpy.shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - numpy.tile(minVals, (m, 1))
	normDataSet = normDataSet / numpy.tile(ranges, (m, 1))

	return normDataSet, ranges, minVals
