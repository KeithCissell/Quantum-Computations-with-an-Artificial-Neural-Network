import csv

def openFile(file):
#Pull out text contents of CSV into a 2D list
    rawInputData = []
    inputFile = open(file, 'rt')
    reader = csv.reader(inputFile)
    for row in reader:
        rawInputData.append(row)
    inputFile.close()
    return rawInputData

def createTrainingData(rawInputData, dataGap):
    #Returns a 2D list of training data
    #dataGap represents the number of rows we want to include
    #dataGap = 1 includes every row, dataGap = 5 includes every fifth row

    trainingData = []
    for i in range(len(rawInputData)):
        #Skip rows that are excluded by gap
        if (i-1) % dataGap == 0:
            #print(rawInputData[i])
            trainingEntry = []
            for j in range(len(rawInputData[i])):
                trainingEntry.append(float(rawInputData[i][j]))
            trainingData.append(trainingEntry)
    return trainingData


# if __name__ == "__main__":
#     print("Main function")
#     rawData = openFile("input.csv")
#     createTrainingData(rawData, 10, 'Hadamard')
