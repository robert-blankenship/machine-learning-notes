import kNN
import numpy

import logging
import matplotlib
import matplotlib.pyplot


def categoryToInt(category):
    return {
        'didntLike': 1,
        'largeDoses': 2,
        'smallDoses': 3
    }[category]


def intToCategory(integer):
    return {
        1: 'didntLike',
        2: 'largeDoses',
        3: 'smallDoses'
    }[integer]


def parseDatingData(filename):
    with open(filename) as fr:
        numberOfLines = len(fr.readlines())

    returnMat = numpy.zeros((numberOfLines, 3))
    classLabelVector = []

    with open(filename) as fr:
        for index, line in enumerate(fr.readlines()):
            listFromLine = line.strip().split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(categoryToInt(listFromLine[-1]))

    return returnMat, classLabelVector


def plotData(dataSet, dataLabels, columns):
    figure = matplotlib.pyplot.figure()
    ax = figure.add_subplot(111)
    ax.scatter(dataSet[:,columns[0]], dataSet[:,columns[1]], 15.0*numpy.array(dataLabels), 15.0*numpy.array(dataLabels))
    matplotlib.pyplot.show()


def makePlots():
    datingDataMat, datingLabels = parseDatingData('datingTestSet.txt')

    plotData(datingDataMat, datingLabels, (0, 1))
    plotData(datingDataMat, datingLabels, (1, 2))
    plotData(datingDataMat, datingLabels, (0, 2))

    normalized, _, _ = autoNorm(datingDataMat)

    plotData(normalized, datingLabels, (0, 1))
    plotData(normalized, datingLabels, (1, 2))
    plotData(normalized, datingLabels, (0, 2))


def datingClassTest(data_file='datingTestSet.txt', k=4):
    hoRatio = 0.10
    datingDataMatrix, datingLabels = parseDatingData(data_file)
    normalizedMatrix, ranges, minimumValues = autoNorm(datingDataMatrix)
    m = normalizedMatrix.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = kNN.classify0(inX=normalizedMatrix[i, :],
                                     dataSet=normalizedMatrix[numTestVecs:m, :],
                                     labels=datingLabels[numTestVecs:m],
                                     k=k)
        logging.debug("the classifier came back with: {}, the real answer is: {}".format(classifierResult,
                                                                                         datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    logging.info("the total error count is: {}".format(errorCount/float(numTestVecs)))


def runMultipleTests():
    for k in range(1, 40):
        logging.info("running test for k={}".format(k))
        datingClassTest(k=k)


def classifyPerson():
    percentTats = float(raw_input("percentage of time playing video games?"))
    ffMiles = float(raw_input("frequent flyer miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMatrix, datingLabels = parseDatingData('datingTestSet.txt')
    normalizedMatrix, ranges, minimumValues = kNN.autoNorm(datingDataMatrix)

    classifierResult = kNN.classify0(inX=(numpy.array([ffMiles, percentTats, iceCream]) - minimumValues)/ranges,
                                 dataSet=normalizedMatrix,
                                 labels=datingLabels,
                                 k=3)
    print "Your probable result: {}".format(intToCategory(classifierResult))


# makePlots()
logging.getLogger().setLevel(logging.INFO)
# runMultipleTests()
classifyPerson()
