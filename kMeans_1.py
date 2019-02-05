import numpy as np
import csv
# AHU data is in the following format:
# [Day of week, hour of day, CHWV position, AHU Fan S/S, AHU DAT, Room temp]
# This testData is not used in the code now. Only data from the csv's called in the demonstrationData function will be operated on.
testData = [[6,18.75,33,1,62.71,62.91,80.4],[6,19.2500000000582,33,1,64.81,65.3,81.01],[6,19.5,0,1,73.6,72.7,81.21],[6,19.7499999999418,33,1,63.5,63.41,81.61],[6,20.25,33,1,65.91,65.91,81.61],[6,20.7500000000582,0,1,72.61,73.11,80.71],[6,21,33,1,63.81,63.81,80.4],[6,21.75,0,1,69.91,69.4,80.11],[6,23.25,0,0,71.31,71.6,79.11],[6,23.7500000000582,0,0,71.8,71.91,79.41],[7,0.249999999941792,0,0,71.8,71.91,79.41],[7,0.999999999941792,0,0,71.8,71.91,79.61],[7,1.25000000005821,0,0,71.8,71.91,79.61],[7,3.50000000005821,0,0,73.11,73,79.61],[7,4.25000000005821,0,0,73.11,73.11,79.61],[7,6.24999999994179,0,0,73.2,73.31,79.41],[7,6.75,0,0,73.2,73.31,79.41],[7,7.25000000005821,0,0,73.11,73.31,79.41],[7,9,0,0,72.91,73,79.41],[7,9.24999999994179,0,0,72.91,73,79.41],[7,9.99999999994179,0,0,72.91,73,79.3],[7,11.0000000000582,0,0,72.91,73,79.3],[7,11.25,0,0,72.91,73,79.3],[7,12,0,0,72.91,73,79.3],[7,12.2499999999418,0,0,72.91,73,79.11],[7,12.75,0,0,72.91,73,79.3],[7,13.2500000000582,0,0,73.2,73.31,79.3],[7,14.0000000000582,0,0,74.01,74.1,79.41],[7,15,0,0,75.2,75.2,79.61],[7,15.2499999999418,0,0,75.51,75.51,79.61],[7,15.75,0,0,76.01,76.01,79.9],[7,17.7500000000582,33,1,64.6,64.31,80.4],[7,18.2499999999418,0,0,69.91,70.3,80.11],[7,18.75,33,1,62.71,62.11,80.4],[7,19.7499999999418,0,0,67.71,68.7,80.4],[7,20.25,33,1,70.9,69.8,80.4],[7,21,0,0,67.21,67.3,80.11],[7,23.4999999999418,0,0,65.61,66.4,80.11],[1,0,0,0,67.8,67.8,79.9],[1,0.500000000058208,0,0,70.11,70,80.2],[1,1.25000000005821,0,0,68.5,68.31,80.11],[1,6,0,0,68.5,68.31,79.41],[1,9,0,1,70.41,70.5,78.01],[1,9.99999999994179,0,1,69.91,68.61,77.5],[1,11.7500000000582,0,1,71.2,70.61,77.41],[1,14.25,33,1,61.61,61.81,78.01],[1,15.5000000000582,0,1,72.01,71.91,79],[1,15.75,0,1,72.81,71.6,79.11],[1,17.25,33,1,76.1,75,79.41],[1,18.75,0,1,64.81,67.3,80.71],[1,18.9999999999418,33,1,62.11,63.1,80.71],[1,19.5,33,1,62.31,62.71,80.71],[1,20.7500000000582,0,1,61.3,61.61,80.51],[1,21,33,1,61.61,61.5,80.51],[1,21.9999999999418,33,1,61.5,61,79.7],[1,22.2500000000582,33,1,61.5,60.71,79.7],[1,23.0000000000582,0,0,71.2,71.11,79.3],[1,23.7500000000582,0,0,71.51,71.6,79.41],[2,0.249999999941792,0,0,71.8,71.91,79.7],[2,0.500000000058208,0,0,72.5,72.7,79.7],[2,1.5,0,0,67.01,68.11,79.9],[2,2.25,0,0,70.11,70.3,79.7],[2,2.49999999994179,0,0,70.41,70.5,79.7],[2,5.00000000005821,0,0,69.4,69.71,79.61],[2,5.75000000005821,0,0,71.01,71.2,79.61],[2,7.25000000005821,0,0,71.8,71.91,79.61],[2,7.74999999994179,0,1,71.51,71.71,79.41],[2,8.00000000005821,0,1,71.2,71.6,79],[2,8.49999999994179,0,1,69.6,70.61,78.51],[2,8.75000000005821,0,1,69.6,68.9,78.31],[2,9.99999999994179,0,1,69.71,70.61,77.7],[2,10.7499999999418,0,1,69.6,70.81,77.9],[2,11.4999999999418,0,1,72.5,72.81,77.5],[2,11.7500000000582,0,1,61.61,61,77.9],[2,12.5000000000582,33,1,63.5,63.7,78.01],[2,12.75,0,1,65.1,72.7,78.01],[2,13.2500000000582,0,1,75.31,71.91,78.21],[2,13.5,0,1,71.71,72.5,78.21],[2,14.0000000000582,0,1,72.3,73,78.51],[2,14.4999999999418,33,1,72.5,73.31,78.6]]

def gridSearchDemonstrationData(trainingFileName,testingFileName, trainingFileHeaderNum = 0, testingFileHeaderNum = 0):
    # This function runs demonstration data through an array of penalty factors and percent anomaly percentages. This data should then be converted into a percentage anomaly
    # for each terminal unit and formed into an array of results. this array should be the [x by y] where x is the length of the number of unique terminal units, and y is the percent anomaly
    # each different iteration that is # of PercentToCapture * penaltyCoefficient. The last column is the average of the percentages
    percentPointsArr = [0.75, 0.8, 0.85, 0.9, 0.95]
    penaltyCoeffArr = [0.1, 0.15, 0.2]
    csvMatrixFlagFilename = trainingFileName[0:-4] + "_flag_Matrix" + trainingFileName[-4:]
    csvMatrixFlagFile = open(csvMatrixFlagFilename, 'wb')  # we want to write to this
    writer = csv.writer(csvMatrixFlagFile)
    anomalyMatrix = []
    matrixColumnNumber = 0
    callNum = 1
    numOfTests = len(percentPointsArr) * len(penaltyCoeffArr)
    condensedAnomalyMatrix = np.zeros([1, (numOfTests + 2)], dtype=object)
    for percentPointsToCapture in percentPointsArr:
        for penaltyCoefficient in penaltyCoeffArr:
            print("Running {0} of {1} tests...".format(callNum,numOfTests))
            condensedAnomalyMatrix[0,callNum] = "Test" + str(callNum)
            tagList, anomArrList = demonstrationData(trainingFileName,testingFileName, trainingFileHeaderNum, testingFileHeaderNum,percentPointsToCapture,penaltyCoefficient)
            if matrixColumnNumber == 0 and len(tagList) > 0: #In the event there are no tags (unlikely), this will skip
                #might want to call the class and ask for column type? In case there are 2 rows of tag names?

                # Used a numpy object array to create placeholders. Probably want to use pandas or something different for conversion
                # to C as numpy object arrays don't convert well unless they're structured arrays.
                anomalyMatrix = np.zeros((len(tagList),((numOfTests + 1))),dtype = object)
                anomalyMatrix[:,0] = tagList
                anomalyMatrix[:,1] = anomArrList
                matrixColumnNumber += 1
                condensedAnomalyMatrix[0,0] = "Device"
            else:
                anomalyMatrix[:,matrixColumnNumber] = anomArrList
            matrixColumnNumber += 1
            callNum = matrixColumnNumber
    condensedAnomalyMatrix[0,(numOfTests + 1)] = "Average"
    writer.writerow(condensedAnomalyMatrix[0])
    print(anomalyMatrix)
    print("")

#Condense anomaly Matrix
# next 3 things happen - looping through the matrix rows
#1) look at tag name, if it's the same, then start adding up the true's
#2) if the tag name changes, stop, create new array of different length (unique tag names) take the percent true of each test,
#3) average the percent trues for each test and append that to a new column

    dataOneTagCount = 0
    numOfCols = len(anomalyMatrix[0])
    buildRowArray = np.zeros([1, numOfCols], dtype=object)
    for row in range(1,(len(anomalyMatrix))):
        if anomalyMatrix[row,0] == anomalyMatrix[row-1,0]:
            dataOneTagCount += 1
            for col in range(1,numOfCols):
                value = anomalyMatrix[row-1, col]
                if value == True:
                    buildRowArray[0,col] += 1.
                else:
                    #Had to do an else statement adding 0. to make values in object array type float so that they can be divided properly later
                    buildRowArray[0,col] += 0.
        else:
            buildRowArray[0,1:] = np.divide(buildRowArray[0,1:],dataOneTagCount)
            buildRowArray[0, 1:] = np.around(buildRowArray[0, 1:].astype(np.float), 3)
            buildRowArray[0,0] = anomalyMatrix[row-1,0]
            averagePercentages = np.round(float(np.sum(buildRowArray[0,1:],axis = None, dtype = float) / float(len(buildRowArray[0,1:]))),3)
            buildRowArray = np.append(buildRowArray,averagePercentages,axis = None)
            writer.writerow(buildRowArray)
            condensedAnomalyMatrix = np.vstack([condensedAnomalyMatrix,buildRowArray])
            buildRowArray = np.zeros([1, numOfCols], dtype=object)
            dataOneTagCount = 0

    print(condensedAnomalyMatrix)





def demonstrationData(trainingFileName, testingFileName, trainingFileHeaderNum = 0, testingFileHeaderNum = 0, percentPointsToCapture = 0.9, penaltyCoefficient = 0.2):
    # This function runs training data and testing data through the kMeans algorithm. In this function, results are printed to the console, and anomalies are identified.
    # Inputs containing the number of header rows allow the algorithm to ignore headers.
    #percent Points to Capture need to be prompted to the user as it controls how sensitive the AI is to anomalies

    BestMeanSet, BestThresholdSet, columnInfo, tagList = trainingFromCSVdata(trainingFileName, trainingFileHeaderNum, percentPointsToCapture = percentPointsToCapture, penaltyCoefficient = penaltyCoefficient)
    print("Best Number of Means is {0}".format(len(BestMeanSet)))
    print("Type " + columnInfo.columnList[0].dataType)

    anomArrTrain, anomPercentTrain, BestMeanSet, DecriptedBestMeanSet = testCSVdata(trainingFileName, trainingFileHeaderNum, BestMeanSet, BestThresholdSet, columnInfo)
    print("{0}% of the training data contains anomalies.".format(round(anomPercentTrain * 100, 0)))

    anomArrTest, anomPercentTest, BestMeanSet, DecriptedBestMeanSet = testCSVdata(testingFileName, testingFileHeaderNum, BestMeanSet, BestThresholdSet, columnInfo)
    print("{0}% of the test data contains anomalies.".format(round(anomPercentTest*100,0)))

    print("Best Mean Set is: ")
    print(DecriptedBestMeanSet)

    if anomPercentTest > 1.3 * (1 - percentPointsToCapture):
        print("Probable Anomaly")
    else:
        print("Likely Not Anomaly")

    print("")

    return tagList, anomArrTest

def readCSV(csvFilename, numHeaderLines = 0):
    #Produces data array of the CSV file which is used in functions: "testCSVdata" and "trainingFromCSVdata"
    csvfile = open(csvFilename,"r")
    csvLines = csv.reader(csvfile)
    dataArray = []
    headerLinesEncountered = 0
    for currRow in csvLines:
        if headerLinesEncountered < numHeaderLines:
            # Is a header line, so skip
            headerLinesEncountered+=1
        else:
            # Is actual data, so convert to numbers if possible
            cleanedRowArray = []
            for currCell in currRow:
                try:
                    cleanedRowArray.append(float(currCell.replace(",", "")))
                except ValueError:
                    cleanedRowArray.append(currCell)
            dataArray.append(cleanedRowArray)
    csvfile.close()
    return dataArray

def testCSVdata(csvFilename, testingFileHeaderNum, BestMeanSet,BestThresholdSet, columnInfo):
    #This function determines whether points are anomalies and pastes whether it's TRUE/FALSE as an anomaly into the CSV.
    #It also outputs the mean data into the CSV file that houses the test data.
    testingData = readCSV(csvFilename, testingFileHeaderNum)#does not contain headers
    anomArr, anomPercent = anomaliesDetected(testingData, columnInfo, BestMeanSet, BestThresholdSet)
    #Write to CSV file here
    csvFlagFilename = csvFilename[0:-4] + "_flags" + csvFilename[-4:]
    csvFlagFile = open(csvFlagFilename, 'wb') #we want to write to this
    thewriter = csv.writer(csvFlagFile)
    csvfile = open(csvFilename,"r")
    csvLines = csv.reader(csvfile)
    headerLinesEncountered = 0
    dataLinesEncountered = 0
    unNormalizedMeanToPrint = np.zeros([len(BestMeanSet),len(BestMeanSet[0])])
    for currRow in csvLines:
        if headerLinesEncountered < testingFileHeaderNum:
            # Is a header line, so skip
            headerLinesEncountered += 1
            if headerLinesEncountered == testingFileHeaderNum:
                currRow.append("Flag")
        else:
            currRow.append(anomArr[dataLinesEncountered])
            dataLinesEncountered += 1 #what index we should pull from anomoly array
        thewriter.writerow(currRow)
    for v in range(len(BestMeanSet)):
        unNormalizedMean = columnInfo.unNormalizedMean(BestMeanSet[v])
        unNormalizedMeanToPrint[v] = np.round(unNormalizedMean[1:],2)   #Attention - tried to put indicies on the unNormalizedMeanToPrint (same as Best Mean Set
        #So that we could have a correcly sized array of the best mean set with un-normalized values to be printed.
        unNormalizedMean.append("Mean")
        thewriter.writerow(unNormalizedMean)
    csvfile.close()
    csvFlagFile.close()
    return anomArr, anomPercent, BestMeanSet, unNormalizedMeanToPrint

def trainingFromCSVdata(CSVfilename, trainingFileHeaderNum=0, percentPointsToCapture = 0.9, penaltyCoefficient = 0.2):
    # The AI is trained in this function with data we know to be "normal". This function is later called in the function demonstrationData.
    # Outputs are the best mean set, the best threshold distances, and the instance of the class
    trainingData = readCSV(CSVfilename, trainingFileHeaderNum)
    columnInfo = listOfColumns()
    columnInfo.detectColumnType(trainingData)  # User Confirms after this point
    tagList = columnInfo.tagListFull
    convertedData = columnInfo.normalizeInputArray(trainingData)
    columnInfo.computeMaxContinuousError()
    # print(len(columnInfo.columnList))
    #Excluding Fan S/S and EA Temp from analysis in FanFixed Data
    # columnInfo.columnList[2].columnWeighting = 0
    # columnInfo.columnList[3].columnWeighting = 0

    BestMeanSet, BestNumberOfMeans, BestThresholdSet = optimumMeans(convertedData, columnInfo, percentPointsToCapture = percentPointsToCapture, PenaltyCoefficient = penaltyCoefficient)
    return BestMeanSet, BestThresholdSet, columnInfo, tagList


def anomaliesDetected(dataSet,columnSet, trueMeanSet, ThresholdActual):
    # Returns and array of values that are outside the maximum threshold of the optimum means. Also outputs the percentage of points that are anomalies
    # The function goes through each point and evaluates if it's within the threshold of any mean.
    # Inputs for this function are the dataSet, instance of the class, the mean set, and the thresholds.
    # This function is called in testCSVData and is used to output the percent anomalies and an array of which values are anomalies
    normalizedData = columnSet.normalizeInputArray(dataSet)
    anomalyArr = []
    numberAnomalies = 0
    for u in range(len(normalizedData)):
        currPoint = normalizedData[u, :]
        notWithinThresh = True
        for z in range(len(trueMeanSet)):
            currBestMean = trueMeanSet[z, :]
            dist = columnSet.computeError(currBestMean, currPoint)
            if dist < ThresholdActual[z]:
                notWithinThresh = False
        if notWithinThresh:
            numberAnomalies += 1
        anomalyArr.append(notWithinThresh)
    percentAnomaly = float(numberAnomalies) / float(len(anomalyArr))
    return anomalyArr, percentAnomaly


def optimumMeans(dataSet, columnSet, numIterationsPerMean = 3, percentPointsToCapture = 0.9, PenaltyCoefficient = 0.1):
    # Finds the number of means that give the data set the least amount of error. The penalty for adding a mean is
    # the number of 0.25 * (Number of points * Discrete Penalty) / Number of Means.

    #
    # The number of means to Test is set by this INPUT numberOfMeansTest. We figured 10 means would probably be the most
    # we'd want to do with the types of data we will be using this on. This value can be adjusted.
    numberOfMeansTest = 10
    numberOfTrainingPoints = len(dataSet)
    # print(numberOfMeansTest)
    percentPointsToCapture = max(0.01, percentPointsToCapture)
    MinErr = 12000000000
    bestThresholdArr = []
    bestMeanSet = None
    bestNumberOfMeans = 0
    NewMeanPenalty = PenaltyCoefficient * (numberOfTrainingPoints * columnSet.discretePenalty) / numberOfMeansTest
    for r in range(1,numberOfMeansTest):
        for i in range(numIterationsPerMean):
            meanSet, totalMeanSetErr, currMeanSetMaxThreshold = mainKMeansFunc(dataSet,r,columnSet,percentPointsToCapture)
            totalMeanSetPenalty = totalMeanSetErr + NewMeanPenalty * r
            # print(totalMeanSetPenalty,r)
            if totalMeanSetPenalty < MinErr:
                bestMeanSet = meanSet
                bestNumberOfMeans = r
                MinErr = totalMeanSetPenalty
                bestThresholdArr = currMeanSetMaxThreshold
    return bestMeanSet, bestNumberOfMeans, bestThresholdArr


def mainKMeansFunc(dataSet, numOfMeans, columnSet, percentOfPointsToCapture):
    # data set is a numpy array (sized n (rows) x m (columns)). Each row represents a different set of features (observations),
    # and each column represents multiple values for one feature (i.e., CHWV).
    # numOfMeans is an integer, represented by k

    # calculate the current mean set. This will be k x m, with represents m features for k means. Each row will be an individual mean.
    # The first set of means will be randomly selected observations from the incoming data set (pick k random observations from the n observations, without replacement)
    currentMeanSet = findInitialMeans(dataSet,numOfMeans)
    convergenceThreshold = 2 #can be whatever number you'd like - may need to be an input?
    keepGoingCond = True
    while keepGoingCond:
        lastMeanSet = currentMeanSet.copy() # Needed to use currentMeanSet.copy() to get around the object references inside of numpy
        # current classification is an n x 1 vector, with each entry being an integer ranging from 0 to (k-1).
        # The integer entry represents the index of the mean that each of the n observations is closest to.
        # includes error normalization (comparing continuous and discrete)
        currentClassification, currTotalErr, maximumThresholdForMeans = classifyData(currentMeanSet, dataSet, columnSet, percentOfPointsToCapture)
        # current mean set will be k x m (see above)
        currentMeanSet = recomputeMeans(currentClassification, dataSet,numOfMeans,columnSet,currentMeanSet)
        # error from last will be a number representing the cumulative distance of each mean to the previous mean in the same index.
        # error = summation(i from 0 to (k-1), || currentMeanSet[i, :] - lastMeanSet[i, :] ||)
        errorFromLast = computeDistanceFromLastMeanSet(currentMeanSet, lastMeanSet,columnSet) # calculate the deviance from the last mean set

        if errorFromLast < convergenceThreshold:
            keepGoingCond = False
    return currentMeanSet, currTotalErr, maximumThresholdForMeans


def findInitialMeans(incomingDataSet, numberOfMeans):
    #Used in the mainkMeans function, this sets an initial set of random means to be compared to the first points in the data set
    randomIndices = np.random.choice(range(len(incomingDataSet)), numberOfMeans)
    return incomingDataSet[randomIndices, :]

def vanillaClassifyData(currMeans, dataSetInUse): #original
    currMeansLength = len(currMeans)
    dataSetInUseLength = len(dataSetInUse)
    distance = np.zeros((dataSetInUseLength, currMeansLength)) #created a correctly sized array to fill in the new values into.
    for i in range(currMeansLength):
        sub = dataSetInUse - currMeans[i,:]
        squared = sub **2.
        sumed = np.sum(squared,axis = 1)
        newDistance = sumed **(0.5)
        distance[:, i] = newDistance
    # print(distance)
    classArr = np.argmin(distance,axis = 1)
    return classArr

def classifyData(currMeans, dataSetInUse, columnSet, percentOfMeansToCapture):
    # classifyData is used in the mainKmeans function. It makes a n x 1 vector , with each entry being an integer
    # ranging from 0 to (k-1). The integer entry represents the index of the mean that each of the n observations is closest to.
    # It also creates an array of min values and max thresholds for each mean. This includes error normalization (comparing continuous and discrete)
    currMeansLength = len(currMeans)
    dataSetInUseLength = len(dataSetInUse)
    distance = np.zeros((dataSetInUseLength, currMeansLength)) #created a correctly sized array to fill in the new values into.
    for i in range(currMeansLength):
        for p in range(dataSetInUseLength):
            mean = currMeans[i,:]
            point = dataSetInUse[p,:]
            distance[p,i] = columnSet.computeError(mean, point)
    classArr = np.argmin(distance,axis = 1) #Returns the "column" index of the closest mean to each point (each row)
    classMin = np.min(distance, axis = 1) #Returns the minimum distance between selected mean and given point - total error for the mean set is calculated by the sum of these minimum distances (in output)
    #Next part outputs the distance for each meanSet that contains 80% of the points associated with each mean
    maxThresholdForMeans = []
    for w in range(currMeansLength):
        minDistancesEachMean = classMin[classArr == w]
        if len(minDistancesEachMean) == 0:
            maxThresholdForMeans.append(0)
        else:
            if percentOfMeansToCapture > 1:
                largeThresholdDist = max(minDistancesEachMean) * percentOfMeansToCapture
                maxThresholdForMeans.append(largeThresholdDist)
            else:
                sortedPointsInMean = np.sort(minDistancesEachMean)
                newNumOfPointsInMean = (len(minDistancesEachMean) * percentOfMeansToCapture) - 1.
                indexForPercentage = int(np.ceil(newNumOfPointsInMean))
                maxThresholdForMeans.append(sortedPointsInMean[indexForPercentage])
    return classArr, sum(classMin), maxThresholdForMeans


def VanillaRecomputeMeans(currClassification, dataSetInUse, numOfMeans):
    newMeans = np.zeros((numOfMeans, len(dataSetInUse[0])))
    for j in range(numOfMeans):
        newMeans[j, :] = np.mean(dataSetInUse[currClassification == j, :], axis=0)
    return (newMeans)


def recomputeMeans(currClassification, dataSetInUse,numOfMeans,columnSet,lastMeanSet):
    # This function finds the new means while looping through data in the function it's in. (Used in the kMeans Main function)
    newMeans = np.zeros((numOfMeans,len(dataSetInUse[0])))
    for j in range(numOfMeans):
        pointsBelongingToMean = dataSetInUse[currClassification == j]
        if len(pointsBelongingToMean) == 0:
            newMeans[j,:] = lastMeanSet[j,:]
            # print("Problem with " + str(numOfMeans))  #Shows the number of points that don't belong to a mean
        else:
            newMeans[j, :] = columnSet.calculateMean(dataSetInUse[currClassification == j])
    return(newMeans)

def computeDistanceFromLastMeanSet(currMeans, lastMeans, columnSet):
    # This function is used in the mainKmeans Function and computes the distance between the last mean and the means
    # derived from the recomputeMeans function above
    ErrorBetweenMeanAndLastMean = 0.
    for difference in range(len(currMeans)):
        ErrorBetweenMeanAndLastMean += columnSet.computeError(currMeans[difference,:],lastMeans[difference,:])
    return ErrorBetweenMeanAndLastMean

class listOfColumns:
    def __init__(self):
        self.discretePenalty = 1
        self.columnList = []
        self.tagListFull = []

    def computeMaxContinuousError(self):
        #In the event there are no continuous (only discrete vairables), or all continuous variables have a column weighting of 0, then the discrete penalty should be 1.
        # Otherwise the discrete variable is equal to the euclidian distance in the column space R = sqrt(x^2+x1^2=x2^2)
        # The discretePenalty is also used in optimumMeans Function to help determine the penalty for adding a mean
        cumulativeSquareError = 0
        for currDataColInfo in self.columnList:
            if (currDataColInfo.dataType == "Continuous"):
                cumulativeSquareError += currDataColInfo.columnWeighting ** 2.
        self.discretePenalty = cumulativeSquareError ** 0.5
        if self.discretePenalty == 0:
            self.discretePenalty = 1
        return self.discretePenalty


    def detectColumnType(self,testDataOriginal):
        # Takes a datatype array and build it out into Discrete/Dismissed/Continuous. Only called once per mean set
        self.columnList = []
        self.tagListFull = []
        # Return the corrected numpy array (normalized)
        for j in range(len(testDataOriginal[0])):
            newDataInfo = dataColumnInformation()
            newDataInfo.columnIndex = j
            # First go through the column deciding if it's discrete or continuous (if discrete then we keep looping), if continuous then we find min and max
            for k in range(len(testDataOriginal)):
                currentPoint = testDataOriginal[k][j]
                if j == 0 and type(currentPoint) == str:  #if strings are in the first column, they're likely tag names, so record them.
                    self.tagListFull.append(currentPoint)
                newDataInfo.continuousOrDiscrete(currentPoint)
            if newDataInfo.dataType != "Dismissed":  # If dismissed we don't continue with that column at all
                self.columnList.append(newDataInfo)

    def normalizeInputArray(self, testDataOriginal):
        # Convert Python array into a numpy array while identifying columns that are discrete/Continuous
        # and other properties using class "dataColumnInformation"
        translatedData = []
        for currColInfo in self.columnList:
            # Second go through the information and normalize it (True/False == 0/1)
            # also strings get mapped to unique indices, (Maybe take continuous numbers and map them between 0 and 1)
            for k in range(len(testDataOriginal)):
                currentPoint = testDataOriginal[k][currColInfo.columnIndex]
                if len(translatedData) <= k:
                    translatedData.append([])
                translatedData[k].append(currColInfo.convertData(currentPoint))
        return np.array(translatedData)

    def computeError(self, mean, point):
        # This function computes the error between a given mean and a given point
        # The error for discrete is the discrete penalty (from above) * the columnWeighting (which we set to 1 in the class below)
        # Column weighting should be an input for the user so that they can change rows they don't care about to a column weighting of 0
        # Continuous Err is distance formula sqrt( sum((distance point is from mean * columnweighting)^2))
        discreteERR = 0.
        continuousSquaredERR = 0.
        for i in range(len(self.columnList)):
            if self.columnList[i].dataType == "Discrete":
                if mean[i] != point[i]:
                    discreteERR += self.discretePenalty * self.columnList[i].columnWeighting
            else:
                continuousSquaredERR += ((mean[i] - point[i]) * self.columnList[i].columnWeighting) ** 2.
        return discreteERR + continuousSquaredERR ** 0.5

    def calculateMean(self, pointForNewAverage):
        # used in recalculateMeans, this function finds the mean of a set of data.
        # For discrete variables it finds the mode and for continuous it finds the actual mean
        newAve = np.zeros((len(self.columnList)))
        for i in range(len(self.columnList)):
            if self.columnList[i].dataType == "Discrete":
                #take the mode (most common value)
                newAve[i] = np.argmax(np.bincount(np.int64(pointForNewAverage[:,i])))
            else:
                newAve[i] = np.mean(pointForNewAverage[:,i])
        return newAve

    def unNormalizedMean(self,mean):
        #if the column index does not match that of the expected column number than insert an *
        #this is used in testCSVData to act as a placeholder for inserting a * when a column value does not exist.
        #in testCSVData this function is used to append the means to the bottom of the CSV.
        outputMean = []
        for i in range(len(self.columnList)):
            currCol = self.columnList[i]
            while len(outputMean) < currCol.columnIndex:
                outputMean.append("*")
            if len(outputMean) == currCol.columnIndex:
                outputMean.append(currCol.unConvertData(mean[i]))
            else:
                outputMean[currCol.columnIndex] = currCol.unConvertData(mean[i])
        return outputMean

class dataColumnInformation:
    #Initialize groups in the class
    def __init__(self):
        self.dataType = "Discrete"
        self.uniqueNames = []
        self.minValue = 0
        self.maxValue = 0
        self.maxAllowableUniqueEntries = 10 #default is 10
        self.columnWeighting = 1.
        self.columnIndex = 0

    def continuousOrDiscrete(self, dataValue):
        #continuousOrDiscrete stores the min value and max value (which are used in convertData and unconvertData)
        #Discrete is default
        # if uniqueNames has a length above the maxAllowableUniqueEntries value, then the value will be considered continuous
        if self.dataType == "Dismissed":
            # do nothing - too many items to understand
            return
        elif self.dataType == "Continuous":
            # do nothing (for now muahahaha)
            if dataValue < self.minValue:
                self.minValue = dataValue
            if dataValue > self.maxValue:
                self.maxValue = dataValue
            return
        else:
            #If we find a new variable, add it to uniqueNames list. (this counts discrete)
            if not (dataValue in self.uniqueNames):
                self.uniqueNames.append(dataValue)
                #Only need to check length of uniqueNames to see if it's inside the threshold if we add items to it
                if len(self.uniqueNames) >= self.maxAllowableUniqueEntries:
                    dataType = type(dataValue)
                    if dataType == int or dataType == float:
                        self.dataType = "Continuous"
                        self.maxValue = max(self.uniqueNames)
                        self.minValue = min(self.uniqueNames)
                    else:
                        self.dataType = "Dismissed"

    def convertData(self, dataValue):
        # Converts continuous data to (value - min)/ (max - min)
        # This function is used in normalizeInputArray
        if self.dataType == "Continuous":
            return (dataValue - self.minValue) / (self.maxValue - self.minValue)
        else:
            if not (dataValue in self.uniqueNames):
                self.uniqueNames.append(dataValue)
            return self.uniqueNames.index(dataValue)

    def unConvertData(self, dataValue):
        #used in unNormalizedMean to unconvert the dataValue from the format made by convertData
        if self.dataType == "Continuous":
            return dataValue * (self.maxValue - self.minValue) + self.minValue
        else:
            return self.uniqueNames[int(dataValue)]

if False:
    testColumnInfo = listOfColumns()
    testColumnInfo.detectColumnType(testData)#User Confirms after this point


    #Workflow for general testing - (after user input)
    convertedData = testColumnInfo.normalizeInputArray(testData)
    testColumnInfo.computeMaxContinuousError()
    # daderSet = np.array(testData) #Replaced with convertedData
    # Curr = findInitialMeans(convertedData,  numberOfMEANS)
    # dataClassification = classifyData(Curr, convertedData,testColumnInfo)
    # nMeans = recomputeMeans(dataClassification,convertedData,numberOfMEANS,testColumnInfo)
    # TotalErr = computeDistanceFromLastMeanSet(Curr,nMeans,testColumnInfo)
    # meanset = mainKMeansFunc(convertedData, numberOfMEANS, testColumnInfo)
    AwesomeMeanSet , AwesomeNumberOfMeans, highestThreshold = optimumMeans(convertedData,testColumnInfo)
    anomArr, anomPercent = anomaliesDetected(testData,testColumnInfo, AwesomeMeanSet, highestThreshold)
else:
    # demonstrationData("CHOA TU Broken.csv","CHOA TU Broken.csv", 1, 1)
    gridSearchDemonstrationData("CHOA TU Broken.csv", "CHOA TU Broken.csv", 1, 1)