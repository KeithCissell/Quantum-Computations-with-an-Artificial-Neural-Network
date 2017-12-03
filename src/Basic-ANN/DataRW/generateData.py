import csv

def generateSWAPData():
    outputFile = open("TrainingData/SWAPGate.csv", 'w')
    csvContent = []
    contentRow = []
    alpha1 = -100
    beta1 = (10000-(alpha1**2))**0.5
    alpha2 = -100
    beta2 = (10000-(alpha2**2))**0.5
    for i in range(0, 51):
        alpha2 = -100
        beta2 = (10000-(alpha2**2))**0.5
        for j in range(0,51):
            contentRow = []
            #Input Values
            contentRow.append(alpha1/100)
            contentRow.append(beta1/100)
            contentRow.append(alpha2/100)
            contentRow.append(beta2/100)
            #Expected Output Values Based on SWAP
            contentRow.append(beta2/100)
            contentRow.append(alpha2/100)
            contentRow.append(beta1/100)
            contentRow.append(alpha1/100)
            csvContent.append(contentRow)
            alpha2 += 4
            beta2 = (10000-(alpha2**2))**0.5
        alpha1 += 4
        beta1 = (10000-(alpha1**2))**0.5
    writer = csv.writer(outputFile)
    writer.writerows(csvContent)
    outputFile.close()

def generateCNOTData():
    outputFile = open("TrainingData/CNOTGate.csv", 'w')
    csvContent = []
    contentRow = []
    alpha1 = -100
    beta1 = (10000-(alpha1**2))**0.5
    alpha2 = -100
    beta2 = (10000-(alpha2**2))**0.5
    for i in range(0, 51):
        alpha2 = -100
        beta2 = (10000-(alpha2**2))**0.5
        for j in range(0,51):
            contentRow = []
            #Input Values
            contentRow.append(alpha1/100)
            contentRow.append(beta1/100)
            contentRow.append(alpha2/100)
            contentRow.append(beta2/100)
            #Expected Output Values Based on CNOT
            contentRow.append(alpha1/100)
            contentRow.append(beta1/100)
            if(beta1/100 == 1.0):
                contentRow.append(beta2/100)
                contentRow.append(alpha2/100)
            else:
                contentRow.append(alpha2/100)
                contentRow.append(beta2/100)
            csvContent.append(contentRow)
            alpha2 += 4
            beta2 = (10000-(alpha2**2))**0.5
        alpha1 += 4
        beta1 = (10000-(alpha1**2))**0.5
    writer = csv.writer(outputFile)
    writer.writerows(csvContent)
    outputFile.close()

def generateCHADAMARDData():
    outputFile = open("TrainingData/CHADAMARDGate.csv", 'w')
    csvContent = []
    contentRow = []
    alpha1 = -100
    beta1 = (10000-(alpha1**2))**0.5
    alpha2 = -100
    beta2 = (10000-(alpha2**2))**0.5
    for i in range(0, 51):
        alpha2 = -100
        beta2 = (10000-(alpha2**2))**0.5
        for j in range(0,51):
            contentRow = []
            #Input Values
            contentRow.append(alpha1/100)
            contentRow.append(beta1/100)
            contentRow.append(alpha2/100)
            contentRow.append(beta2/100)
            #Expected Output Values Based on CHADAMARD
            contentRow.append(alpha1/100)
            contentRow.append(beta1/100)
            if(beta1/100 == 1.0):
                contentRow.append(((alpha2/100)+(beta2/100))/(2**0.5))
                contentRow.append(((alpha2/100)-(beta2/100))/(2**0.5))
            else:
                contentRow.append(alpha2/100)
                contentRow.append(beta2/100)
            csvContent.append(contentRow)
            alpha2 += 4
            beta2 = (10000-(alpha2**2))**0.5
        alpha1 += 4
        beta1 = (10000-(alpha1**2))**0.5
    writer = csv.writer(outputFile)
    writer.writerows(csvContent)
    outputFile.close()

def generateCPFLIPData():
    outputFile = open("TrainingData/CPFLIPGate.csv", 'w')
    csvContent = []
    contentRow = []
    alpha1 = -100
    beta1 = (10000-(alpha1**2))**0.5
    alpha2 = -100
    beta2 = (10000-(alpha2**2))**0.5
    for i in range(0, 51):
        alpha2 = -100
        beta2 = (10000-(alpha2**2))**0.5
        for j in range(0,51):
            contentRow = []
            #Input Values
            contentRow.append(alpha1/100)
            contentRow.append(beta1/100)
            contentRow.append(alpha2/100)
            contentRow.append(beta2/100)
            #Expected Output Values Based on CPFLIP
            contentRow.append(alpha1/100)
            contentRow.append(beta1/100)
            if(beta1/100 == 1.0):
                contentRow.append(alpha2/100)
                contentRow.append(-(beta2/100))
            else:
                contentRow.append(alpha2/100)
                contentRow.append(beta2/100)
            csvContent.append(contentRow)
            alpha2 += 4
            beta2 = (10000-(alpha2**2))**0.5
        alpha1 += 4
        beta1 = (10000-(alpha1**2))**0.5
    writer = csv.writer(outputFile)
    writer.writerows(csvContent)
    outputFile.close()

def generateTOFFOLIData():
    outputFile = open("TrainingData/TOFFOLIGate.csv", 'w')
    csvContent = []
    contentRow = []
    alpha1 = -100
    beta1 = (10000-(alpha1**2))**0.5
    alpha2 = -100
    beta2 = (10000-(alpha2**2))**0.5
    alpha3 = -100
    beta3 = (10000-(alpha3**2))**0.5
    for i in range(0, 11):
        alpha2 = -100
        beta2 = (10000-(alpha2**2))**0.5
        for j in range(0,11):
            alpha3 = -100
            beta3 = (10000-(alpha3**2))**0.5
            for k in range(0,11):
                contentRow = []
                #Input Values
                contentRow.append(alpha1/100)
                contentRow.append(beta1/100)
                contentRow.append(alpha2/100)
                contentRow.append(beta2/100)
                contentRow.append(alpha3/100)
                contentRow.append(beta3/100)
                #Expected Output Values Based on TOFFOLIN
                contentRow.append(alpha1/100)
                contentRow.append(beta1/100)
                contentRow.append(alpha2/100)
                contentRow.append(beta2/100)
                if(beta1/100 == 1.0 and beta2/100 == 1.0):
                    contentRow.append(beta3/100)
                    contentRow.append(alpha3/100)
                else:
                    contentRow.append(alpha3/100)
                    contentRow.append(beta3/100)
                csvContent.append(contentRow)
                alpha3 += 20
                beta3 = (10000-(alpha3**2))**0.5
            alpha2 += 20
            beta2 = (10000-(alpha2**2))**0.5
        alpha1 += 20
        beta1 = (10000-(alpha1**2))**0.5
    writer = csv.writer(outputFile)
    writer.writerows(csvContent)
    outputFile.close()

def generateFREDKINData():
    outputFile = open("TrainingData/FREDKINGate.csv", 'w')
    csvContent = []
    contentRow = []
    alpha1 = -100
    beta1 = (10000-(alpha1**2))**0.5
    alpha2 = -100
    beta2 = (10000-(alpha2**2))**0.5
    alpha3 = -100
    beta3 = (10000-(alpha3**2))**0.5
    for i in range(0, 11):
        alpha2 = -100
        beta2 = (10000-(alpha2**2))**0.5
        for j in range(0,11):
            alpha3 = -100
            beta3 = (10000-(alpha3**2))**0.5
            for k in range(0,11):
                contentRow = []
                #Input Values
                contentRow.append(alpha1/100)
                contentRow.append(beta1/100)
                contentRow.append(alpha2/100)
                contentRow.append(beta2/100)
                contentRow.append(alpha3/100)
                contentRow.append(beta3/100)
                #Expected Output Values Based on FREDKIN
                contentRow.append(alpha1/100)
                contentRow.append(beta1/100)
                if(beta1/100 == 1.0):
                    contentRow.append(beta3/100)
                    contentRow.append(alpha3/100)
                    contentRow.append(beta2/100)
                    contentRow.append(alpha2/100)
                else:
                    contentRow.append(alpha2/100)
                    contentRow.append(beta2/100)
                    contentRow.append(alpha3/100)
                    contentRow.append(beta3/100)
                csvContent.append(contentRow)
                alpha3 += 20
                beta3 = (10000-(alpha3**2))**0.5
            alpha2 += 20
            beta2 = (10000-(alpha2**2))**0.5
        alpha1 += 20
        beta1 = (10000-(alpha1**2))**0.5
    writer = csv.writer(outputFile)
    writer.writerows(csvContent)
    outputFile.close()