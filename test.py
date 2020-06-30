import budd 
from pomdp import *
import datetime
import collins

#Writes a numpy matrix to an xls file. Returns the last row the matrix was written on. Currently supports only 3D numpy matrices.
def writeNumpyMatrixToFile(sheet, matrix, row=0,col=0):
    dimensions = matrix.shape
    rowCount = row
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            
            for k in range(dimensions[2]):
                sheet.write(rowCount,col+k, matrix[i][j][k])
            rowCount = rowCount + 1
        rowCount = rowCount + 1 #Provide an extra space between submatrices
    return rowCount

#Writes a numpy matrix to a csv file. Currently supports only 3D numpy matrices.
def writeNumpyMatrixToCSV(c, matrix):
    dimensions = matrix.shape
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            c.writerow(matrix[i][j][:])
        c.writerow([]) #Provide an extra space between submatrices


#Uses the Test 1 parameters outlined in the SBLTests.docx file with column updates (Collins' method)
def test1_v1(filename,env):
    gainThresh = 0.05 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 25000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = False
    revisedSplitting = False
    haveControl = False
    confidenceFactor = None
    collins.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor)

#Uses the Test 1 parameters outlined in the SBLTests.docx file without column updates (Our method)
def test1_v2(filename,env):
    gainSplitThresh = 0.05 #Threshold of gain to determine if the model should stop learning
    surpriseThresh = 0 #0; used for one-step extension gain splitting
    entropyThresh = 0.55
    numActionsPerExperiment = 50000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    explore = 0.05
    percentTimeofBudd = 0.9
    budd.approximateSPOMDPLearning(env=env, gainThresh=gainSplitThresh, numActions=numActionsPerExperiment, explore=explore, surpriseThresh=surpriseThresh, splitWithEntropy=True, entropyThresh=entropyThresh,writeToFile=True, earlyTermination=False, budd=True, percentTimeofBudd=percentTimeofBudd, filename=filename)

#Uses the Test 1 parameters outlined in the SBLTests.docx file with column updates (Collins' method) with Budd enabled
def test1_v3(filename,env):
    gainThresh = 0.05 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 25000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = True
    revisedSplitting = False
    haveControl = False
    confidenceFactor = None
    collins.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor)

#Uses the Test 2 parameters outlined in the SBLTests.docx file with random actions (no agent control)
def test2_v1(filename,env):
    haveControl = False
    confidenceFactor = 100
    gainThresh = 0.05 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 25000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = True
    revisedSplitting = False
    collins.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor)

#Uses the Test 2 parameters outlined in the SBLTests.docx file with agent control
def test2_v2(filename,env):
    haveControl = True
    confidenceFactor = 100
    gainThresh = 0.05 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 25000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = True
    revisedSplitting = False
    collins.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor)

#Uses the Test 3 parameters outlined in the SBLTests.docx file with Collins' method of SDE splitting
def test3_v1(filename,env):
    gainThresh = 0.05 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 25000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = False
    revisedSplitting = False
    haveControl = False
    confidenceFactor = None
    collins.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor)


#Uses the Test 3 parameters outlined in the SBLTests.docx file with improved SDE splitting
def test3_v3(filename,env):
    gainThresh = 0.05 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 25000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = True
    revisedSplitting = True
    haveControl = False
    confidenceFactor = None
    collins.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor)
    
if __name__ == "__main__":
    testNum = 3
    versionNum = 1
    envNum = 4
    numSubTests = 1
    testString = "test"+str(testNum)+"_v"+str(versionNum)
    envString = "Example"+str(envNum)

    date = datetime.datetime.today()

    for subTest in range(numSubTests):
        filename = "Testing Data/Test" + str(testNum) + "_v" + str(versionNum) + "_env" + str(envNum) + "_" + str(date.month) + "_" + str(date.day) +  "_" + str(date.hour) + "_" + str(date.minute) + "_" + str(subTest) + ".csv"
        print(filename)
        env = locals()[envString]()
        locals()[testString](filename,env)
