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

def createTrainingData(rawInputData, dataDensity, gateType):
    #Returns a 2D list of training data
    #dataDensity represents the number of rows we want to include
    #dataDensity = 1 includes every row, dataDensity = 5 includes every fifth row
    gate = gateType.upper()

    #print(gate) 

    trainingData = []
    for i in range(len(rawInputData)):
        #Skip the headings and rows that are excluded by density
        if i > 0 and (i-1) % dataDensity == 0:  
            #print(rawInputData[i])
            trainingEntry = []
            for j in range(len(rawInputData[i])):
                #Positions 0 and 1 are universal alpha and beta
                if j < 2:
                    trainingEntry.append(float(rawInputData[i][j]))
                #Positions 4 and 5 are the NOT gate alpha and beta
                if gate == "NOT" and (j == 4 or j == 5):
                    trainingEntry.append(float(rawInputData[i][j]))
                #Positions 8 and 9 are the HADAMARD gate alpha and beta
                if gate == "HADAMARD" and (j == 8 or j == 9):
                    trainingEntry.append(float(rawInputData[i][j]))
            trainingData.append(trainingEntry)
    return trainingData


# if __name__ == "__main__":
#     print("Main function")
#     rawData = openFile("input.csv")
#     createTrainingData(rawData, 10, 'Hadamard')

