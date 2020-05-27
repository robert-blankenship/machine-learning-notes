import kNN
import logging
import numpy
import os

def img2Vector(filename):
    with open(filename) as f:
        output = numpy.zeros((1, 1024))
        for i in range(32):
            line = f.readline()
            for j in range(32):
                output[0, 32*i + j] = int(line[j])
        return output



def handwritingClassTest(testDigitsFolder='testDigits', trainingDigitsFolder='trainingDigits'):

    def getClassFromFilename(filename):
        fileStr = filename.split('.')[0]
        return int(fileStr.split('_')[0])

    def getTrainingData():
        trainingFileList = os.listdir(trainingDigitsFolder)
        m = len(trainingFileList)
        trainingMat = numpy.zeros((m, 1024))
        hwLabels = []
        for i, filename in enumerate(trainingFileList):
            hwLabels.append(getClassFromFilename(filename))
            trainingMat[i,:] = img2Vector('trainingDigits/{}'.format(filename))
        return trainingMat, hwLabels

    trainingMat, hwLabels = getTrainingData()

    testFileList = os.listdir(testDigitsFolder)
    errorCount = 0.0
    mTest = len(testFileList)
    for i, filename in enumerate(testFileList):
        classNumStr = getClassFromFilename(filename)
        vectorUnderTest = img2Vector('testDigits/{}'.format(filename))
        classifierResult = kNN.classify0(inX=vectorUnderTest,
                                         dataSet=trainingMat,
                                         labels=hwLabels,
                                         k=5)
        logging.debug("classifier result: {}, actual value: {}".format(classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1
            logging.info("file={},classifierResult={},actual={}".format(filename, classifierResult, classNumStr))

    logging.info("total errors: {}".format(errorCount))
    logging.info("total error rate: {}".format(errorCount/float(mTest)))

logging.getLogger().setLevel(logging.INFO)

handwritingClassTest()

